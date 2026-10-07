"""Fang-Oosterlee COS method for European calls and puts, vectorised over contracts.

The density of y = log(S_tau / K) = x + X is expanded on [a, b] around c1 + x. Two
truncation rules:
  "c2" (default, frozen for the benchmark): half-width L sqrt(c2), L = 12
  "c4" (Fang and Oosterlee 2008):           half-width L sqrt(c2 + sqrt(c4)), L = 10
with exact c1, c2 (characteristic.cumulants) or contour-integral cumulants of log phi.
Puts are priced by the series and calls by parity, as Fang and Oosterlee recommend.

Quantities: price, Delta, Gamma (x-derivatives term by term), Vega, Vanna, Volga
(phi = exp(C + D v0), so v0-derivatives multiply each term by D), rate rho by the chain
rule through x and the discount, Theta from the PDE identity, and the kappa, theta, xi,
rho sensitivities by central bumps with [a, b] held fixed.

The rule was frozen on the Phase 0 development contracts (score = mean RMSE / RMS over the
12 quantities): c2/L=12 is best at every setting from 32 to 256 terms (score 4.6e-2, 1.3e-2,
3.8e-3, 3.9e-4) and floors at 2.9e-4 from 512 terms on; the c4 rule only wins at 1024 terms.
"""

import numpy as np

from .characteristic import cf_parts, cumulants, numerical_cumulants
from .model import PARAMETERS, apply_parity, theta_from_pde

BUMPS = {"kappa": 1e-4, "theta": 1e-5, "xi": 1e-4, "rho": 1e-4}


def _chi_psi(omega, a, c, d):
    """chi_k(c, d) = int_c^d e^y cos(w (y - a)) dy and psi_k(c, d) = int_c^d cos(w (y - a)) dy."""
    wd, wc = omega * (d - a), omega * (c - a)
    chi = (np.cos(wd) * np.exp(d) - np.cos(wc) * np.exp(c)
           + omega * (np.sin(wd) * np.exp(d) - np.sin(wc) * np.exp(c))) / (1 + omega**2)
    safe = np.where(omega == 0, 1.0, omega)
    psi = np.where(omega == 0, d - c, (np.sin(wd) - np.sin(wc)) / safe)
    return chi, psi


def truncation(c, rule="c2", L=None):
    if rule == "c2":
        m1, m2 = cumulants(c.tau, c.v0, c.kappa, c.theta, c.xi, c.rho)
        half = (L or 12.0) * np.sqrt(np.abs(m2))
    elif rule == "c4":
        m1, m2, m4 = numerical_cumulants(c.tau, c.v0, c.kappa, c.theta, c.xi, c.rho)
        half = (L or 10.0) * np.sqrt(np.abs(m2) + np.sqrt(np.abs(m4)))
    else:
        raise ValueError(rule)
    centre = c.x + m1
    return centre - half, centre + half


def _series(c, terms, a, b, derivatives=True):
    """Normalised put p and its x/v0 derivatives on the given [a, b] (arrays per contract)."""
    a, b = a[:, None], b[:, None]
    x = c.x[:, None]
    omega = np.arange(terms) * np.pi / (b - a)
    upper = np.minimum(b, 0.0)
    chi, psi = _chi_psi(omega, a, np.minimum(a, upper), upper)
    Vk = 2 / (b - a) * (psi - chi)
    col = lambda v: v[:, None]
    C, D = cf_parts(omega, col(c.tau), col(c.kappa), col(c.theta), col(c.xi), col(c.rho))
    t = np.exp(C + D * col(c.v0)) * np.exp(1j * omega * (x - a)) * Vk
    t[:, 0] *= 0.5
    out = {"p": t.real.sum(1)}
    if derivatives:
        iw = 1j * omega
        out.update(p_x=(iw * t).real.sum(1), p_xx=(iw * iw * t).real.sum(1), p_v=(D * t).real.sum(1),
                   p_xv=(iw * D * t).real.sum(1), p_vv=(D * D * t).real.sum(1))
    return out


def cos_quantities(c, terms=256, rule="c2", L=None, parameter_greeks=True):
    """All 12 quantities per unit strike (calls and puts); also returns the interval [a, b]."""
    a, b = truncation(c, rule, L)
    s = _series(c, terms, a, b)
    disc, ex, S = c.discount, np.exp(c.x), c.S
    cc, cx, cxx = s["p"] + ex - 1, s["p_x"] + ex, s["p_xx"] + ex   # call by parity, c = p + e^x - 1
    out = {"price": disc * cc, "delta": disc * cx / S, "gamma": disc * (cxx - cx) / S**2,
           "vega": disc * s["p_v"], "vanna": disc * s["p_xv"] / S, "volga": disc * s["p_vv"]}
    out["rho_rate"] = c.tau * (disc * cx - out["price"])
    for p in PARAMETERS:
        if not parameter_greeks:
            out[f"d_{p}"] = np.full(len(c), np.nan)
            continue
        h = BUMPS[p]
        up = _series(c.with_values(**{p: getattr(c, p) + h}), terms, a, b, False)["p"]
        dn = _series(c.with_values(**{p: getattr(c, p) - h}), terms, a, b, False)["p"]
        out[f"d_{p}"] = disc * (up - dn) / (2 * h)
    out = apply_parity(c, out)
    out["theta_cal"] = theta_from_pde(c, out)
    out["a"], out["b"] = a, b
    return out
