"""Heston contracts, the study domain, and the forward reduction.

Everything is per unit strike: K = 1 and S is the moneyness S/K. With
F = S e^{(r-q) tau} and x = log(F/K) the call value is
V = K e^{-r tau} c(x, v0, tau; kappa, theta, xi, rho); puts follow by parity.
"""

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

FIELDS = ("S", "tau", "r", "q", "v0", "theta", "kappa", "xi", "rho")
PARAMETERS = ("kappa", "theta", "xi", "rho")
QUANTITIES = ("price", "delta", "gamma", "vega", "vanna", "volga", "theta_cal",
              "rho_rate", "d_kappa", "d_theta", "d_xi", "d_rho")
DOMAIN_FILE = Path(__file__).resolve().parents[2] / "experiments" / "config" / "domain.json"


@dataclass(frozen=True)
class Contracts:
    """A batch of European options. `call` is boolean; everything else is float64."""

    S: np.ndarray
    tau: np.ndarray
    r: np.ndarray
    q: np.ndarray
    v0: np.ndarray
    theta: np.ndarray
    kappa: np.ndarray
    xi: np.ndarray
    rho: np.ndarray
    call: np.ndarray

    def __post_init__(self):
        n = None
        for f in fields(self):
            dtype = bool if f.name == "call" else np.float64
            value = np.atleast_1d(np.asarray(getattr(self, f.name), dtype=dtype))
            object.__setattr__(self, f.name, value)
            n = value.size if n is None else n
            if value.ndim != 1 or value.size != n:
                raise ValueError("all contract fields must be 1-D with equal length")

    def __len__(self):
        return self.S.size

    @property
    def x(self):
        """Log forward moneyness log(F/K)."""
        return np.log(self.S) + (self.r - self.q) * self.tau

    @property
    def discount(self):
        return np.exp(-self.r * self.tau)

    def take(self, index):
        return Contracts(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def with_values(self, **changes):
        n = len(self)
        changes = {k: np.broadcast_to(np.asarray(v, dtype=bool if k == "call" else float), (n,)).copy()
                   for k, v in changes.items()}
        return replace(self, **changes)

    def records(self):
        return [{**{name: float(getattr(self, name)[i]) for name in FIELDS},
                 "call": bool(self.call[i])} for i in range(len(self))]

    @staticmethod
    def concat(*batches):
        return Contracts(**{f.name: np.concatenate([getattr(b, f.name) for b in batches])
                            for f in fields(Contracts)})

    @staticmethod
    def from_records(records):
        return Contracts(**{name: [rec[name] for rec in records] for name in FIELDS + ("call",)})


def load_domain(path=DOMAIN_FILE):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sample_contracts(n, rng, domain=None, short=False):
    """Independent uniform draws over the domain box (tau from the short split if asked)."""
    ranges = (domain or load_domain())["ranges"]
    draw = {}
    for name in FIELDS:
        key = {"S": "moneyness", "tau": "tau_short" if short else "tau"}.get(name, name)
        lo, hi = ranges[key]
        draw[name] = rng.uniform(lo, hi, n)
    draw["call"] = rng.random(n) < 0.5
    return Contracts(**draw)


def development_contracts(domain=None):
    """The seed-20261006 development set: (domain, short, hard) contracts."""
    domain = domain or load_domain()
    spec = domain["development_sample"]
    rng = np.random.default_rng(spec["seed"])
    main = sample_contracts(spec["n_domain"], rng, domain)
    short = sample_contracts(spec["n_short"], rng, domain, short=True)
    return main, short, hard_contracts()


# name -> (S, tau, r, q, v0, theta, kappa, xi, rho, call)
HARD = {
    "short_deep_otm_call": (0.7, 0.005, 0.10, 0.0, 0.01, 0.01, 0.5, 1.0, -0.9, True),
    "short_deep_otm_put": (1.3, 0.005, 0.0, 0.06, 0.01, 0.36, 5.0, 0.1, 0.3, False),
    "short_otm_call_high_vol": (0.7, 0.005, 0.05, 0.0, 0.36, 0.36, 0.5, 1.0, -0.9, True),
    "short_atm_max_gamma": (1.0, 0.005, 0.0, 0.0, 0.01, 0.01, 0.5, 1.0, -0.9, True),
    "feller_strong_atm": (1.0, 2.0, 0.03, 0.0, 0.01, 0.01, 0.5, 1.0, 0.0, True),
    "feller_strong_skew_put": (0.85, 1.0, 0.03, 0.01, 0.01, 0.01, 0.5, 1.0, -0.9, False),
    "rho_m09_long_itm_call": (1.3, 2.0, 0.03, 0.0, 0.36, 0.36, 0.5, 1.0, -0.9, True),
    "v0_high_theta_low_fast": (1.0, 0.5, 0.03, 0.0, 0.36, 0.01, 5.0, 0.5, -0.5, True),
    "v0_high_theta_low_slow_otm": (0.7, 2.0, 0.03, 0.0, 0.36, 0.01, 0.5, 1.0, -0.9, True),
    "main_short_low_vol_otm": (0.7, 0.05, 0.0, 0.0, 0.01, 0.01, 0.5, 0.1, 0.0, True),
    "long_low_vol_otm_carry": (0.7, 2.0, -0.02, 0.06, 0.01, 0.01, 5.0, 0.1, 0.3, True),
}


def hard_contracts():
    rows = np.array([v[:-1] for v in HARD.values()], dtype=float).T
    return Contracts(*rows, call=[v[-1] for v in HARD.values()])


def feller_violated(c):
    return 2 * c.kappa * c.theta < c.xi**2


def average_variance(v0, theta, kappa, tau):
    """Mean of (1/tau) int_0^tau E[v_t] dt, the Black-Scholes limit variance as xi -> 0."""
    kt = kappa * tau
    return theta + (v0 - theta) * -np.expm1(-kt) / kt


def apply_parity(c, out):
    """Turn call quantities in `out` into put quantities where c.call is False (in place).

    P = C - S e^{-q tau} + K e^{-r tau}; the parameter derivatives, Gamma, Vega,
    Vanna and Volga are unchanged.
    """
    put = ~c.call
    if not put.any():
        return out
    carry = np.exp(-c.q * c.tau)
    out["price"] = np.where(put, out["price"] - c.S * carry + c.discount, out["price"])
    out["delta"] = np.where(put, out["delta"] - carry, out["delta"])
    out["rho_rate"] = np.where(put, out["rho_rate"] - c.tau * c.discount, out["rho_rate"])
    if "theta_cal" in out:
        out["theta_cal"] = np.where(put, out["theta_cal"] - c.q * c.S * carry + c.r * c.discount,
                                    out["theta_cal"])
    return out


def theta_from_pde(c, out):
    """Calendar Theta = -V_tau from the Heston PDE with v = v0."""
    v, S = c.v0, c.S
    v_tau = (0.5 * v * S**2 * out["gamma"] + c.rho * c.xi * v * S * out["vanna"]
             + 0.5 * c.xi**2 * v * out["volga"] + (c.r - c.q) * S * out["delta"]
             + c.kappa * (c.theta - v) * out["vega"] - c.r * out["price"])
    return -v_tau
