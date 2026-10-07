"""Shared machinery of the PINN and DeepONet pricers: inputs, outputs, saving and loading.

Every pricer exposes ``price(x, put)`` on contract rows (``coordinates.FIELDS``), so the
price and all 11 sensitivities come from one autograd routine. Two input layouts:

``raw``      the nine contract inputs plus a put flag; the network returns price / K.
``reduced``  (x, v, tau; kappa, theta, xi, rho) from the exact reduction; the network
             returns the normalized call and puts follow by parity.

Outputs for the reduced layout:

``plain``        c = N(features).
``bs``           c = BS(x, w tau) + tau N(x, v, sqrt(tau), parameters): the payoff holds
                 exactly and the network learns the correction to Black-Scholes at the
                 expected average variance.
``bs_envelope``  c = BS + tau phi(d_2) N: the correction fades like the time value away
                 from the strike.

``bs_vol``        c = BS(x, sqrt(w tau) e^N): the network is a log correction to the base's
                 total volatility (an implied-volatility output). Exact at expiry; to first
                 order the correction is sqrt(w tau) phi(d_2) N, so it scales like the time
                 value near expiry (sqrt(tau), not tau) and away from the strike, and N carries
                 the Heston skew and smile, which stay of order one in volatility as tau -> 0.

``drift_inputs = "time_scaled"`` feeds kappa tau and kappa theta tau instead of kappa and
theta (reduced layout, Black-Scholes base). kappa and theta move the variance only through
its drift over the option's life, so their effect on the correction vanishes as tau -> 0
by construction, while for tau > 0 nothing is lost (kappa and theta are recoverable).

``v_input = "sqrt"`` feeds sqrt(v + 0.01) instead of v (reduced layout), which stretches
the region near v = 0 where the solution changes fastest in v. The shift keeps the input
smooth at v = 0: with plain sqrt(v), c_v would be infinite there unless the network were
even in sqrt(v), and the degenerate equation c_tau = kappa theta c_v needs a finite c_v.
"""

import math

import torch
from torch import nn

from .coordinates import (COLUMN, FIELDS, REDUCED, base_call, base_envelope, base_volatility, black_call,
                          price_from_call, reduced_coordinates)

RAW_INPUTS = FIELDS
V_SHIFT = 0.01   # v_input = "sqrt" feeds sqrt(v + V_SHIFT)


def mlp(inputs, width, depth, outputs):
    """tanh multilayer perceptron with Xavier-normal weights and zero biases."""
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(inputs, width, dtype=torch.float64), nn.Tanh()]
        inputs = width
    layers.append(nn.Linear(inputs, outputs, dtype=torch.float64))
    network = nn.Sequential(*layers)
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return network


class Scaler(nn.Module):
    """Affine map of each input from its [low, high] box onto [-1, 1]."""

    def __init__(self, low, high):
        super().__init__()
        low = torch.as_tensor(low, dtype=torch.float64)
        high = torch.as_tensor(high, dtype=torch.float64)
        self.register_buffer("center", (high + low) / 2)
        self.register_buffer("radius", (high - low) / 2)

    def forward(self, values):
        return (values - self.center) / self.radius


class NeuralPricer(nn.Module):
    """Base class: input scaling, output construction and the price per unit strike.

    ``boxes`` gives [low, high] per input name (``RAW_INPUTS`` or ``REDUCED``).
    Subclasses implement ``network(scaled_inputs, put_flag)``; the flag is None for ``reduced``.
    """

    family = None

    def __init__(self, layout, output, boxes, v_input="v", drift_inputs="plain"):
        super().__init__()
        if drift_inputs not in ("plain", "time_scaled"):
            raise ValueError(f"unknown drift_inputs {drift_inputs!r}")
        if drift_inputs == "time_scaled" and (layout == "raw" or output == "plain"):
            raise ValueError("time-scaled drift inputs need the reduced layout with a Black-Scholes base")
        self.drift_inputs = drift_inputs
        if v_input not in ("v", "sqrt"):
            raise ValueError(f"unknown v_input {v_input!r}")
        if layout == "raw" and v_input != "v":
            raise ValueError("the raw layout takes v0 as it is")
        self.v_input = v_input
        if layout not in ("raw", "reduced") or output not in ("plain", "bs", "bs_envelope", "bs_vol"):
            raise ValueError(f"unknown layout {layout!r} or output {output!r}")
        if layout == "raw" and output != "plain":
            raise ValueError("the raw layout has a plain output")
        self.layout, self.output = layout, output
        names = RAW_INPUTS if layout == "raw" else REDUCED
        low = [boxes[n][0] for n in names]
        high = [boxes[n][1] for n in names]
        if output != "plain":
            low[2], high[2] = math.sqrt(low[2]), math.sqrt(high[2])   # the network sees sqrt(tau)
        if v_input == "sqrt":
            low[1], high[1] = math.sqrt(low[1] + V_SHIFT), math.sqrt(high[1] + V_SHIFT)
        if drift_inputs == "time_scaled":
            # the network sees kappa tau and kappa theta tau in place of kappa and theta
            t_hi = boxes["tau"][1]
            k_hi, th_hi = boxes["kappa"][1], boxes["theta"][1]
            low[3], high[3] = 0.0, k_hi * t_hi
            low[4], high[4] = 0.0, k_hi * th_hi * t_hi
        self.scaler = Scaler(low, high)
        self.inputs = names

    def network(self, inputs, flag):
        raise NotImplementedError

    def call(self, features):
        """Normalized call c(x, v, tau; kappa, theta, xi, rho) (reduced layout)."""
        if self.layout != "reduced":
            raise TypeError("call() needs the reduced layout")
        v = features[:, 1:2]
        if self.v_input == "sqrt":
            v = torch.sqrt(v + V_SHIFT)
        if self.output == "plain":
            return self.network(self.scaler(torch.cat((features[:, :1], v, features[:, 2:]), dim=1)), None)
        tau = features[:, 2]
        drift = features[:, 3:5]
        if self.drift_inputs == "time_scaled":
            kt = features[:, 3] * tau
            drift = torch.stack((kt, kt * features[:, 4]), dim=1)
        scaled = self.scaler(torch.cat((features[:, :1], v, torch.sqrt(tau)[:, None], drift, features[:, 5:]), dim=1))
        if self.output == "bs_vol":
            return black_call(features[:, 0], base_volatility(features) * torch.exp(self.network(scaled, None)))
        correction = tau * self.network(scaled, None)
        if self.output == "bs_envelope":
            correction = correction * base_envelope(features)
        return base_call(features) + correction

    def normalized_price(self, x, put):
        """Price / K from contract rows (raw layout)."""
        return self.network(self.scaler(x), put.to(x.dtype))

    def price(self, x, put):
        if self.layout == "raw":
            return self.normalized_price(x, put)
        features, scale = reduced_coordinates(x)
        return price_from_call(self.call(features), features, scale, put)

    def forward(self, x, put):
        return self.price(x, put)


class Ensemble(nn.Module):
    """Average of member prices; Greeks are member averages by linearity."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.family = members[0].family

    def price(self, x, put):
        return torch.stack([m.price(x, put) for m in self.members]).mean(0)

    def call(self, features):
        return torch.stack([m.call(features) for m in self.members]).mean(0)

    def forward(self, x, put):
        return self.price(x, put)


def build_model(family, **settings):
    from .deeponet import DeepONet
    from .pinn import PINN
    return {"pinn": PINN, "deeponet": DeepONet}[family](**settings)


def _record(model):
    return dict(family=model.family, settings=model.settings, state_dict=model.state_dict())


def save_model(model, path, extra=None):
    if isinstance(model, Ensemble):
        record = dict(family=model.family, ensemble=[_record(m) for m in model.members])
    else:
        record = _record(model)
    torch.save(dict(record, **(extra or {})), path)


def _restore(record):
    model = build_model(record["family"], **record["settings"])
    model.load_state_dict(record["state_dict"])
    return model.eval()


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "ensemble" in checkpoint:
        return Ensemble([_restore(m) for m in checkpoint["ensemble"]]).eval()
    return _restore(checkpoint)
