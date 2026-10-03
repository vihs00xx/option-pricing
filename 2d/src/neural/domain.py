"""Contract boxes, samplers and the scale-free coordinates shared by every neural pricer.

A contract row holds the eleven fields in ``FIELDS``. Prices are homogeneous of
degree one in (spots, strike) and the weights only enter through w_i S_i, so a
call is ``K exp(-r tau) c(f_1, f_2, a_1, a_2, rho)`` with forward moneyness
f_i = w_i S_i exp((r - q_i) tau) / K and total volatility a_i = sigma_i sqrt(tau).
Puts follow from exact put-call parity.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc
import torch

FIELDS = ("spot_1", "spot_2", "strike", "tau", "rate", "dividend_1", "dividend_2",
          "vol_1", "vol_2", "correlation", "weight_1")
COLUMN = {name: i for i, name in enumerate(FIELDS)}
SAMPLED = tuple(name for name in FIELDS if name != "strike")


@dataclass(frozen=True)
class ContractBox:
    spot: tuple = (50.0, 150.0)
    strike: float = 100.0
    tau: tuple = (0.02, 2.0)
    rate: tuple = (0.0, 0.08)
    dividend: tuple = (0.0, 0.05)
    vol: tuple = (0.10, 0.70)
    correlation: tuple = (-0.95, 0.95)
    weight_1: tuple = (0.2, 0.8)

    def bounds(self):
        return {"spot_1": self.spot, "spot_2": self.spot, "tau": self.tau, "rate": self.rate,
                "dividend_1": self.dividend, "dividend_2": self.dividend, "vol_1": self.vol,
                "vol_2": self.vol, "correlation": self.correlation, "weight_1": self.weight_1}

    def contains(self, contracts, tolerance=1e-12):
        x = np.asarray(contracts, float)
        inside = np.isclose(x[:, COLUMN["strike"]], self.strike)
        for name, (low, high) in self.bounds().items():
            column = x[:, COLUMN[name]]
            inside &= (column >= low-tolerance) & (column <= high+tolerance)
        return inside


def box_from_dict(settings):
    return ContractBox(**{key: tuple(value) if isinstance(value, list) else value
                          for key, value in settings.items()})


def sample_contracts(box, count, seed, tau_power=1.0):
    """Scrambled Sobol contracts; tau_power > 1 concentrates samples at short maturities."""
    unit = qmc.Sobol(d=len(SAMPLED), scramble=True, seed=seed).random(count)
    x = np.empty((count, len(FIELDS)))
    x[:, COLUMN["strike"]] = box.strike
    for j, name in enumerate(SAMPLED):
        low, high = box.bounds()[name]
        u = unit[:, j]**tau_power if name == "tau" else unit[:, j]
        x[:, COLUMN[name]] = low+(high-low)*u
    return x


def columns(x):
    return {name: x[..., i] for i, name in enumerate(FIELDS)}


def reduced_coordinates(x):
    """Return (f_1, f_2, a_1, a_2, rho, discounted strike) for a torch contract tensor."""
    c = columns(x)
    weight_2 = 1-c["weight_1"]
    f1 = c["weight_1"]*c["spot_1"]*torch.exp((c["rate"]-c["dividend_1"])*c["tau"])/c["strike"]
    f2 = weight_2*c["spot_2"]*torch.exp((c["rate"]-c["dividend_2"])*c["tau"])/c["strike"]
    root_tau = torch.sqrt(c["tau"])
    scale = c["strike"]*torch.exp(-c["rate"]*c["tau"])
    return f1, f2, c["vol_1"]*root_tau, c["vol_2"]*root_tau, c["correlation"], scale


def price_from_normalized(normalized_call, x, put):
    """Convert c(f, a, rho) to currency prices; ``put`` is a boolean tensor."""
    f1, f2, _, _, _, scale = reduced_coordinates(x)
    parity = torch.where(put, f1+f2-1, torch.zeros_like(f1))
    return scale*(normalized_call-parity)


def moment_matched_vol(f1, f2, a1, a2, rho):
    """Total volatility of the lognormal with the basket's first two moments (Levy)."""
    forward = f1+f2
    second = (f1*f1*torch.exp(a1*a1)+f2*f2*torch.exp(a2*a2)
              + 2*f1*f2*torch.exp(rho*a1*a2))
    return torch.sqrt(torch.log(second/(forward*forward)))


def black_call(forward, total_vol):
    """Undiscounted Black call with unit strike."""
    d1 = torch.log(forward)/total_vol+total_vol/2
    return forward*torch.special.ndtr(d1)-torch.special.ndtr(d1-total_vol)
