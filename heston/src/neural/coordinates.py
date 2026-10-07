"""Contract rows, the exact reduction and the Black-Scholes base for one-stock Heston.

A contract row holds the nine fields in ``FIELDS`` (per unit strike: S is S/K). With
constant r and q the price is

    V = K e^{-r tau} c(x, v, tau; kappa, theta, xi, rho),   x = log(S e^{(r-q) tau} / K),

and the normalized call c solves the driftless reduced PDE

    c_tau = 1/2 v (c_xx - c_x) + rho xi v c_xv + 1/2 xi^2 v c_vv + kappa (theta - v) c_v,

with c(tau = 0) = (e^x - 1)^+; at v = 0 it degenerates to c_tau = kappa theta c_v. Puts
follow from parity, p = c - (e^x - 1). Nine contract inputs become three coordinates and
four parameters.

The base the networks correct is Black-Scholes at the expected average variance,
w = theta + (v - theta)(1 - e^{-kappa tau}) / (kappa tau), exact as xi -> 0.
"""

import math

import numpy as np
import torch
from scipy.stats import qmc

FIELDS = ("S", "tau", "r", "q", "v0", "kappa", "theta", "xi", "rho")
COLUMN = {name: i for i, name in enumerate(FIELDS)}
REDUCED = ("x", "v", "tau", "kappa", "theta", "xi", "rho")
RANGE_KEY = {"S": "moneyness"}


def sample_domain(ranges, count, seed, tau=None):
    """Scrambled Sobol contract rows (count, 9) in the domain box; calls and puts alternate."""
    unit = qmc.Sobol(d=len(FIELDS), scramble=True, seed=seed).random(count)
    x = np.empty((count, len(FIELDS)))
    for j, name in enumerate(FIELDS):
        lo, hi = tau if (name == "tau" and tau is not None) else ranges[RANGE_KEY.get(name, name)]
        x[:, j] = lo + (hi - lo) * unit[:, j]
    return x, np.arange(count) % 2 == 1


def rows_from_contracts(c):
    """Contract rows (numpy) and put flags from a src.heston.model.Contracts batch."""
    return np.stack([getattr(c, n) for n in FIELDS], axis=1), ~c.call


def columns(x):
    return {name: x[..., i] for i, name in enumerate(FIELDS)}


def reduced_coordinates(x):
    """(x, v, tau, kappa, theta, xi, rho) and the discount factor for torch contract rows."""
    c = columns(x)
    logf = torch.log(c["S"]) + (c["r"] - c["q"]) * c["tau"]
    features = torch.stack((logf, c["v0"], c["tau"], c["kappa"], c["theta"], c["xi"], c["rho"]), dim=-1)
    return features, torch.exp(-c["r"] * c["tau"])


def price_from_call(call, features, scale, put):
    parity = torch.where(put, torch.exp(features[..., 0]) - 1, torch.zeros_like(call))
    return scale * (call - parity)


def integrated_variance(features):
    """E[int_0^tau v_t dt] = theta tau + (v - theta)(1 - e^{-kappa tau}) / kappa."""
    _, v, tau, kappa, theta = features.unbind(-1)[:5]
    return theta * tau - (v - theta) * torch.expm1(-kappa * tau) / kappa


def base_volatility(features):
    """Total volatility sqrt(w tau) of the Black-Scholes base."""
    return torch.sqrt(integrated_variance(features))


def black_call(x, total_vol):
    """Undiscounted Black call with unit strike on forward e^x."""
    d1 = x / total_vol + total_vol / 2
    return torch.exp(x) * torch.special.ndtr(d1) - torch.special.ndtr(d1 - total_vol)


def base_call(features):
    return black_call(features[..., 0], base_volatility(features))


def base_envelope(features):
    """phi(d_2): the base call's sensitivity to its total volatility."""
    tv = base_volatility(features)
    return torch.exp(-0.5 * (features[..., 0] / tv - tv / 2) ** 2) / math.sqrt(2 * math.pi)
