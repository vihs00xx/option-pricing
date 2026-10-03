"""Contract rows, the protocol domain, the reduced coordinates and the Levy approximation.

A contract row holds the eleven fields in ``FIELDS``. Under risk-neutral GBM the call
price is homogeneous of degree one in (spots, strike), the weights enter only through
w_i S_i, and rates and dividends only through the forwards and the discount factor:

    C = K exp(-r tau) c(z, s, tau; sigma_1, sigma_2, rho),
    f_i = w_i S_i exp((r - q_i) tau) / K,  z = log(f_1 + f_2),  s = f_1 / (f_1 + f_2).

In y_i = log f_i the normalized call c solves the driftless two-asset PDE

    c_tau = 1/2 sigma_1^2 (c_11 - c_1) + rho sigma_1 sigma_2 c_12 + 1/2 sigma_2^2 (c_22 - c_2)

with c(tau = 0) = (exp(z) - 1)^+. Puts follow from exact put-call parity,
p = c - (exp(z) - 1). The reduction removes the strike, both weights, the rate and
both dividends exactly: eleven contract inputs become three coordinates and three
parameters.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc
import torch

FIELDS = ("spot_1", "spot_2", "strike", "tau", "rate", "dividend_1", "dividend_2",
          "vol_1", "vol_2", "correlation", "weight_1")
COLUMN = {name: i for i, name in enumerate(FIELDS)}
# Test-set parameter names (``experiments/config/protocol/test_set_v1.json``) to row fields.
PROTOCOL_FIELDS = {"tau": "tau", "volatility_1": "vol_1", "volatility_2": "vol_2",
                   "correlation": "correlation", "rate": "rate", "dividend_1": "dividend_1",
                   "dividend_2": "dividend_2", "weight_1": "weight_1"}
PROTOCOL_PATH = Path(__file__).resolve().parents[2]/"experiments/config/protocol/test_set_v1.json"


def protocol_domain(path=PROTOCOL_PATH):
    """Training domain and strike of the frozen test-set protocol."""
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    return protocol["training_domain"], float(protocol["strike"])


def rows_from_parameters(values, strike):
    """Contract rows from protocol-named arrays (moneyness_i, tau, volatility_i, ...)."""
    count = len(np.atleast_1d(values["tau"]))
    x = np.empty((count, len(FIELDS)))
    x[:, COLUMN["strike"]] = strike
    x[:, COLUMN["spot_1"]] = strike*np.asarray(values["moneyness_1"], float)
    x[:, COLUMN["spot_2"]] = strike*np.asarray(values["moneyness_2"], float)
    for name, field in PROTOCOL_FIELDS.items():
        x[:, COLUMN[field]] = np.asarray(values[name], float)
    return x


def sample_domain(domain, strike, count, seed, tau=None):
    """Scrambled Sobol contracts in the protocol box; calls and puts alternate.

    ``tau`` optionally overrides the maturity range (for example a short-maturity panel).
    """
    names = tuple(domain)
    unit = qmc.Sobol(d=len(names), scramble=True, seed=seed).random(count)
    values = {}
    for j, name in enumerate(names):
        low, high = tau if (name == "tau" and tau is not None) else domain[name]
        values[name] = low+(high-low)*unit[:, j]
    return rows_from_parameters(values, strike), np.arange(count) % 2 == 1


def columns(x):
    return {name: x[..., i] for i, name in enumerate(FIELDS)}


def reduced_coordinates(x):
    """(z, s, tau, sigma_1, sigma_2, rho) and the discounted strike for a torch contract tensor."""
    c = columns(x)
    f1 = c["weight_1"]*c["spot_1"]*torch.exp((c["rate"]-c["dividend_1"])*c["tau"])/c["strike"]
    f2 = (1-c["weight_1"])*c["spot_2"]*torch.exp((c["rate"]-c["dividend_2"])*c["tau"])/c["strike"]
    forward = f1+f2
    features = torch.stack((torch.log(forward), f1/forward, c["tau"], c["vol_1"], c["vol_2"],
                            c["correlation"]), dim=-1)
    return features, c["strike"]*torch.exp(-c["rate"]*c["tau"])


def price_from_call(call, features, scale, put):
    """Currency price from the normalized call; ``put`` is a boolean tensor."""
    parity = torch.where(put, torch.exp(features[..., 0])-1, torch.zeros_like(call))
    return scale*(call-parity)


def log_forwards(features):
    """(y_1, y_2) = (z + log s, z + log(1 - s))."""
    z, s = features[..., 0], features[..., 1]
    return z+torch.log(s), z+torch.log1p(-s)


def features_from_log_forwards(y1, y2, rest):
    """Inverse of ``log_forwards``; ``rest`` holds (tau, sigma_1, sigma_2, rho)."""
    z = torch.logaddexp(y1, y2)
    return torch.cat((torch.stack((z, torch.exp(y1-z)), dim=-1), rest), dim=-1)


def levy_volatility(features):
    """Total volatility of the lognormal matching the basket's first two moments (Levy)."""
    z, s, tau, v1, v2, rho = features.unbind(-1)
    a1, a2 = v1*torch.sqrt(tau), v2*torch.sqrt(tau)
    # E[B^2] / E[B]^2 with B the normalized basket; expm1 keeps short maturities exact.
    ratio = (s*s*torch.expm1(a1*a1)+(1-s)*(1-s)*torch.expm1(a2*a2)
             + 2*s*(1-s)*torch.expm1(rho*a1*a2))
    return torch.sqrt(torch.log1p(ratio))


def black_call(z, total_vol):
    """Undiscounted Black call with unit strike on forward exp(z)."""
    d1 = z/total_vol+total_vol/2
    return torch.exp(z)*torch.special.ndtr(d1)-torch.special.ndtr(d1-total_vol)


def levy_call(features):
    return black_call(features[..., 0], levy_volatility(features))

