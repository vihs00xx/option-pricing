"""Shared machinery of the PINN and DeepONet pricers: inputs, outputs, saving and loading.

Every pricer exposes ``price(x, put)`` on contract rows (``Stocks(d).fields``), so
prices and all Greeks come from one autograd routine. Two input layouts:

``raw``      the contract inputs (spots over strike, maturity, rate, dividends,
             volatilities, correlations, the first d - 1 weights) plus a put flag; the
             network returns price over strike.
``reduced``  the exact reduction of ``coordinates``: (z, s, tau; sigma, rho); the network
             returns the normalized call and puts follow by parity.

Two outputs for the reduced layout:

``plain``    c = N(features).
``levy``     c = Levy(features) + tau * N(features); the payoff at tau = 0 holds exactly
             and the network learns the correction to the moment-matched lognormal.
             Its maturity feature is sqrt(tau), which resolves the sqrt(tau) structure
             of option prices near expiry. Optionally the network also sees the
             boundary-layer coordinate tanh(z / (3 total volatility)).
``levy_envelope``  c = Levy + tau * phi(d_2) * N: the correction is scaled by the Levy
             call's sensitivity to total volatility, so it fades with the time value
             deep in and out of the money, where additive errors cross the
             no-arbitrage bounds.

``pinn.PINN`` and ``deeponet.DeepONet`` supply the network N. The module layout and
state-dict keys are those of the two-stock study, so its checkpoints load for d = 2.
"""

import math

import torch
from torch import nn

from .coordinates import Stocks, levy_call, levy_volatility, price_from_call, reduced_coordinates


def raw_inputs(stocks):
    """Raw network inputs: the contract fields without the strike, spots as moneyness."""
    return tuple(f for f in stocks.fields if f != "strike")


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
        self.register_buffer("center", (high+low)/2)
        self.register_buffer("radius", (high-low)/2)

    def forward(self, values):
        return (values-self.center)/self.radius


class NeuralPricer(nn.Module):
    """Base class: input scaling, output construction and the currency price.

    ``boxes`` gives [low, high] per input: for ``raw`` keyed by ``raw_inputs(stocks)``
    (spots as moneyness), for ``reduced`` keyed by ``stocks.features``. Subclasses
    implement ``network(scaled_inputs, put_flag)``; the flag is None for ``reduced``.
    """

    family = None

    def __init__(self, d, layout, output, boxes, similarity=None):
        super().__init__()
        if layout not in ("raw", "reduced") or output not in ("plain", "levy", "levy_envelope"):
            raise ValueError(f"unknown layout {layout!r} or output {output!r}")
        if layout == "raw" and output != "plain":
            raise ValueError("the raw layout has a plain output")
        if similarity is not None and output == "plain":
            raise ValueError("the similarity coordinate needs the levy output")
        self.stocks = Stocks(d)
        self.layout, self.output, self.similarity = layout, output, similarity
        base = raw_inputs(self.stocks) if layout == "raw" else self.stocks.features
        low = [boxes[n][0] for n in base]
        high = [boxes[n][1] for n in base]
        if output != "plain":
            # The network sees sqrt(tau) on [0, sqrt(tau_max)].
            t = self.stocks.tau
            low[t], high[t] = math.sqrt(low[t]), math.sqrt(high[t])
        self.scaler = Scaler(low, high)
        self.inputs = base+(("similarity",) if similarity is not None else ())

    def network(self, inputs, flag):
        raise NotImplementedError

    def similarity_coordinate(self, features):
        """Bounded boundary-layer coordinate tanh(d / 3) with d = z / (total volatility).

        ``similarity = "levy"`` uses the Levy total volatility of the contract; a number
        uses that fixed volatility times sqrt(tau), which depends on the coordinates
        only (for a DeepONet trunk).
        """
        z, tau = features[:, 0], features[:, self.stocks.tau]
        if self.similarity == "levy":
            scale = levy_volatility(self.stocks, features)
        else:
            scale = float(self.similarity)*torch.sqrt(tau)
        return torch.tanh(z/(3*scale))

    def call(self, features):
        """Normalized call c(z, s, tau; sigma, rho) (reduced layout)."""
        if self.layout != "reduced":
            raise TypeError("call() needs the reduced layout")
        if self.output == "plain":
            return self.network(self.scaler(features), None)
        t = self.stocks.tau
        tau = features[:, t]
        # sqrt(tau) has an infinite slope at 0; collocation keeps tau > 0.
        scaled = self.scaler(torch.cat((features[:, :t], torch.sqrt(tau)[:, None], features[:, t+1:]), dim=1))
        if self.similarity is not None:
            scaled = torch.cat((scaled, self.similarity_coordinate(features)[:, None]), dim=1)
        correction = tau*self.network(scaled, None)
        if self.output == "levy_envelope":
            # phi(d_2) = exp(z) phi(d_1) is the Levy call's sensitivity to its total volatility:
            # the correction fades like the time value away from the strike.
            total_vol = levy_volatility(self.stocks, features)
            correction = correction*torch.exp(-0.5*(features[:, 0]/total_vol-total_vol/2)**2)/math.sqrt(2*math.pi)
        return levy_call(self.stocks, features)+correction

    def normalized_price(self, x, put):
        """Price over strike from contract rows (raw layout)."""
        col = self.stocks.column
        strike = x[:, col["strike"]]
        values = torch.stack([x[:, col[n]]/strike if n.startswith("spot") else x[:, col[n]]
                              for n in raw_inputs(self.stocks)], dim=1)
        return self.network(self.scaler(values), put.to(x.dtype))

    def price(self, x, put):
        """Currency price of contract rows ``x``; ``put`` is a boolean tensor."""
        if self.layout == "raw":
            return x[:, self.stocks.column["strike"]]*self.normalized_price(x, put)
        features, scale = reduced_coordinates(self.stocks, x)
        return price_from_call(self.call(features), features, scale, put)

    def forward(self, x, put):
        return self.price(x, put)


class Ensemble(nn.Module):
    """Average of member prices; Greeks are member averages by linearity."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.family = members[0].family
        self.stocks = members[0].stocks

    def price(self, x, put):
        return torch.stack([m.price(x, put) for m in self.members]).mean(0)

    def forward(self, x, put):
        return self.price(x, put)


def build_model(family, **settings):
    from .deeponet import DeepONet
    from .pinn import PINN
    return {"pinn": PINN, "deeponet": DeepONet}[family](**settings)


def _record(model):
    return dict(family=model.family, settings=model.settings, state_dict=model.state_dict())


def save_model(model, path, extra=None):
    """One self-contained checkpoint; an ensemble stores every member in the same file."""
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
        return Ensemble([_restore(member) for member in checkpoint["ensemble"]]).eval()
    return _restore(checkpoint)
