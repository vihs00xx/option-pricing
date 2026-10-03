"""Neural basket pricers. Every model returns the normalized undiscounted call c.

``features`` chooses the network inputs:
    raw      the ten sampled contract fields scaled to [-1, 1]
    reduced  (log F, theta, a_1, a_2, rho) with F = f_1 + f_2 and theta = f_1 / F,
             standardised with training-set statistics
``output`` chooses how the network output y becomes c:
    direct       c = y (raw models: c = y exp(r tau), so y is the call price over K)
    gap          c = Black(F, a_B) + scale * y, a learned correction to the Levy approximation
    implied_vol  c = Black(F, a_B exp(scale * y)), a learned basket implied-volatility ratio
Puts always come from exact put-call parity in ``price_from_normalized``.
"""

import torch
from torch import nn

from .domain import (COLUMN, SAMPLED, black_call, moment_matched_vol,
                     price_from_normalized, reduced_coordinates)

ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "softplus": nn.Softplus, "gelu": nn.GELU}


def mlp(inputs, width, depth, activation, zero_last=False):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(inputs, width), ACTIVATIONS[activation]()]
        inputs = width
    layers.append(nn.Linear(inputs, 1))
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    if zero_last:
        nn.init.zeros_(layers[-1].weight)
    return nn.Sequential(*layers)


def reduced_features(f1, f2, a1, a2, rho):
    forward = f1+f2
    return torch.stack((torch.log(forward), f1/forward, a1, a2, rho), dim=-1)


class BasketPricer(nn.Module):
    def __init__(self, features="reduced", output="direct", width=64, depth=4,
                 activation="tanh", scale=0.1, low=None, high=None, mean=None, std=None):
        super().__init__()
        if features not in ("raw", "reduced") or output not in ("direct", "gap", "implied_vol"):
            raise ValueError("unknown features or output")
        if features == "raw" and output != "direct":
            raise ValueError("raw features only support direct output")
        self.config = dict(features=features, output=output, width=width, depth=depth,
                           activation=activation, scale=scale)
        inputs = len(SAMPLED) if features == "raw" else 5
        self.network = mlp(inputs, width, depth, activation, zero_last=output != "direct")
        self.register_buffer("low", torch.as_tensor(low if low is not None else torch.zeros(inputs)))
        self.register_buffer("high", torch.as_tensor(high if high is not None else torch.ones(inputs)))
        self.register_buffer("mean", torch.as_tensor(mean if mean is not None else torch.zeros(inputs)))
        self.register_buffer("std", torch.as_tensor(std if std is not None else torch.ones(inputs)))

    def call_from_reduced(self, f1, f2, a1, a2, rho):
        """Normalized call from scale-free coordinates; only for reduced-feature models."""
        u = (reduced_features(f1, f2, a1, a2, rho)-self.mean)/self.std
        y = self.network(u)[..., 0]
        if self.config["output"] == "direct":
            return y
        forward = f1+f2
        levy = moment_matched_vol(f1, f2, a1, a2, rho)
        if self.config["output"] == "gap":
            return black_call(forward, levy)+self.config["scale"]*y
        return black_call(forward, levy*torch.exp(self.config["scale"]*y))

    def normalized_call(self, x):
        if self.config["features"] == "reduced":
            return self.call_from_reduced(*reduced_coordinates(x)[:5])
        raw = x[..., [COLUMN[name] for name in SAMPLED]]
        y = self.network(2*(raw-self.low)/(self.high-self.low)-1)[..., 0]
        return y*torch.exp(x[..., COLUMN["rate"]]*x[..., COLUMN["tau"]])

    def forward(self, x, put):
        return price_from_normalized(self.normalized_call(x), x, put)


class EnsemblePricer(nn.Module):
    """Average of member normalized calls; its Greeks are the average Greeks."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def call_from_reduced(self, f1, f2, a1, a2, rho):
        return torch.stack([m.call_from_reduced(f1, f2, a1, a2, rho) for m in self.members]).mean(0)

    def normalized_call(self, x):
        return torch.stack([m.normalized_call(x) for m in self.members]).mean(0)

    def forward(self, x, put):
        return price_from_normalized(self.normalized_call(x), x, put)


def save_pricer(model, path, **extra):
    torch.save(dict(config=model.config, state_dict=model.state_dict(), **extra), path)


def load_pricer(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BasketPricer(**checkpoint["config"]).to(torch.float64)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def load_ensemble(paths):
    return EnsemblePricer([load_pricer(path)[0] for path in paths]).eval()
