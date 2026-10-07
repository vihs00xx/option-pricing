"""Torch twin of the Lewis reference: batched complex128 quadrature, 12 quantities by autograd.

The integral over [0, U] is mapped by u = U s^2 (nodes cluster near u = 0, where
1/(u^2 + 1/4) varies fastest) and integrated with P panels of 16-point
Gauss-Legendre on s in [0, 1]. U comes from the NumPy truncation scan and is a
constant for autograd; the boundary term it drops is below the truncation tolerance.
The node count is chosen per contract by doubling P until two levels agree.
"""

import math
import time

import numpy as np
import torch

from .model import QUANTITIES, Contracts
from .reference import AGREEMENT, truncation_point

DTYPE = torch.float64
TRUNCATION = 1e-15
SERIES_CUTOFF = 1e-2
GRAD_INPUTS = ("S", "v0", "tau", "r", "kappa", "theta", "xi", "rho")
_GL_T, _GL_W = np.polynomial.legendre.leggauss(16)


def composite_rule(panels):
    """Nodes and weights of `panels` x 16-point Gauss-Legendre on [0, 1]."""
    edges = np.linspace(0.0, 1.0, panels + 1)
    mid, half = 0.5 * (edges[:-1] + edges[1:]), 0.5 * np.diff(edges)
    s = (mid[:, None] + half[:, None] * _GL_T).ravel()
    w = (half[:, None] * _GL_W).ravel()
    return torch.tensor(s, dtype=DTYPE), torch.tensor(w, dtype=DTYPE)


def _expm1_c(z):
    a, b = z.real, z.imag
    s = torch.sin(0.5 * b)
    return torch.complex(torch.expm1(a) * torch.cos(b) - 2 * s * s, torch.exp(a) * torch.sin(b))


def _log1p_c(w):
    a, b = w.real, w.imag
    return torch.complex(0.5 * torch.log1p(a * (2 + a) + b * b), torch.atan2(b, 1 + a))


def _log1p_ratio(w):
    small = w.abs() < SERIES_CUTOFF
    series = 1 + w * (-1 / 2 + w * (1 / 3 + w * (-1 / 4 + w * (1 / 5 + w * (-1 / 6 + w * (1 / 7 - w / 8))))))
    safe = torch.where(small, torch.ones_like(w), w)
    return torch.where(small, series, _log1p_c(safe) / safe)


def cf_parts(u, tau, kappa, theta, xi, rho):
    """Little-trap C and D in the same xi-stable form as characteristic.cf_parts."""
    psi = u * (u + 1j)
    b = kappa - rho * xi * (1j * u)
    xi2 = xi * xi
    d = torch.sqrt(b * b + xi2 * psi)
    bpd = b + d
    b_minus_d_over_xi2 = -psi / bpd
    g_over_xi2 = -psi / (bpd * bpd)
    g = xi2 * g_over_xi2
    one_minus_e = -_expm1_c(-(d * tau))
    e = 1 - one_minus_e
    D = b_minus_d_over_xi2 * one_minus_e / (1 - g * e)
    w_over_xi2 = g_over_xi2 * one_minus_e / (1 - g)
    log_over_xi2 = w_over_xi2 * _log1p_ratio(xi2 * w_over_xi2)
    C = kappa * theta * (b_minus_d_over_xi2 * tau - 2 * log_over_xi2)
    return C, D


def price(t, U, s, w):
    """Option prices for a batch; t maps field names to (B,) tensors, U is (B,), (s, w) the rule."""
    col = {k: v[:, None] for k, v in t.items()}
    u = U[:, None] * s * s
    weight = w * 2 * U[:, None] * s
    z = torch.complex(torch.full_like(u, 0.5), u)
    x = torch.log(col["S"]) + (col["r"] - col["q"]) * col["tau"]
    C, D = cf_parts(u - 0.5j, col["tau"], col["kappa"], col["theta"], col["xi"], col["rho"])
    G = torch.exp(z * x + C + D * col["v0"])
    integral = ((G / (z * (1 - z))).real * weight).sum(dim=1)
    x = x[:, 0]
    disc = torch.exp(-t["r"] * t["tau"])
    call_value = disc * (torch.exp(x) - integral / math.pi)
    put_value = call_value - t["S"] * torch.exp(-t["q"] * t["tau"]) + disc
    return torch.where(t["call"], call_value, put_value)


def _tensors(c):
    t = {}
    for name in ("S", "tau", "r", "q", "v0", "theta", "kappa", "xi", "rho"):
        t[name] = torch.tensor(getattr(c, name), dtype=DTYPE, requires_grad=name in GRAD_INPUTS)
    t["call"] = torch.tensor(c.call)
    return t


def quantities_batch(c, U, panels):
    """All 12 quantities for contracts c at a fixed panel count (one autograd pass)."""
    s, w = composite_rule(panels)
    t = _tensors(c)
    V = price(t, torch.tensor(U, dtype=DTYPE), s, w)
    first = torch.autograd.grad(V.sum(), [t[k] for k in GRAD_INPUTS], create_graph=True)
    d = dict(zip(GRAD_INPUTS, first))
    gamma, vanna = torch.autograd.grad(d["S"].sum(), [t["S"], t["v0"]], retain_graph=True)
    (volga,) = torch.autograd.grad(d["v0"].sum(), [t["v0"]])
    out = {"price": V, "delta": d["S"], "gamma": gamma, "vega": d["v0"], "vanna": vanna,
           "volga": volga, "theta_cal": -d["tau"], "rho_rate": d["r"], "d_kappa": d["kappa"],
           "d_theta": d["theta"], "d_xi": d["xi"], "d_rho": d["rho"]}
    return {q: out[q].detach().numpy() for q in QUANTITIES}


def quantities(c, panels, U=None, max_points=1 << 18):
    """quantities_batch in chunks of at most max_points contract-nodes."""
    U = truncation_point(c, TRUNCATION) if U is None else U
    chunk = max(1, max_points // (16 * panels))
    parts = [quantities_batch(c.take(slice(i, i + chunk)), U[i:i + chunk], panels)
             for i in range(0, len(c), chunk)]
    return {q: np.concatenate([p[q] for p in parts]) for q in QUANTITIES}


def twin(c, tol=AGREEMENT, start_panels=4, max_panels=1024):
    """Doubling ladder per contract: returns (values at the accepted level, info)."""
    n = len(c)
    U = truncation_point(c, TRUNCATION)
    out = {q: np.full(n, np.nan) for q in QUANTITIES}
    info = {"panels": np.zeros(n, int), "change": np.full(n, np.inf), "converged": np.zeros(n, bool),
            "U": U, "seconds": 0.0}
    t0 = time.perf_counter()
    active = np.arange(n)
    panels = start_panels
    prev = quantities(c, panels, U)
    while active.size and panels < max_panels:
        panels *= 2
        sub = c.take(active)
        cur = quantities(sub, panels, U[active])
        change = np.max([np.abs(cur[q] - prev[q]) for q in QUANTITIES], axis=0)
        done = change <= tol
        last = panels >= max_panels
        for q in QUANTITIES:
            out[q][active] = cur[q]
        info["panels"][active] = panels
        info["change"][active] = change
        info["converged"][active] = done
        keep = ~done if not last else np.zeros_like(done)
        active = active[keep]
        prev = {q: cur[q][keep] for q in QUANTITIES}
    info["seconds"] = time.perf_counter() - t0
    info["nodes"] = 16 * info["panels"]
    return out, info
