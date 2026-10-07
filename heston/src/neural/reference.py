"""Reference labels from the torch twin, and the price-and-Greeks routine for any pricer.

The twin (``src.heston.torch_reference``) agrees with the NumPy reference to about 1e-13
on all 12 quantities (Phase 0). Labels of the normalized call come from pricing a call
with S = e^x, r = q = 0 and converting its sensitivities to x-derivatives.
"""

import numpy as np
import torch

from src.heston import torch_reference as tr
from src.heston.model import QUANTITIES, Contracts

from .coordinates import COLUMN, FIELDS

LABELS = ("c", "c_x", "c_v", "c_tau", "c_kappa", "c_theta", "c_xi", "c_rho", "c_xx", "c_xv", "c_vv")


def contracts_from_rows(x, put):
    x = np.asarray(x)
    return Contracts(**{n: x[:, COLUMN[n]] for n in FIELDS}, call=~np.asarray(put, bool))


def reference_values(x, put):
    """All 12 quantities of contract rows from the twin (node ladder per contract)."""
    values, info = tr.twin(contracts_from_rows(x, put))
    if not info["converged"].all():
        raise RuntimeError(f"twin ladder did not converge on {int((~info['converged']).sum())} contracts")
    return values


def reference_labels(features):
    """(N, 11) labels ordered as LABELS at reduced features (x, v, tau, kappa, theta, xi, rho)."""
    f = np.asarray(features)
    S = np.exp(f[:, 0])
    c = Contracts(S=S, tau=f[:, 2], r=0.0 * S, q=0.0 * S, v0=f[:, 1], kappa=f[:, 3], theta=f[:, 4],
                  xi=f[:, 5], rho=f[:, 6], call=np.ones(len(S), bool))
    v, info = tr.twin(c)
    if not info["converged"].all():
        raise RuntimeError("twin ladder did not converge on some label points")
    cx = S * v["delta"]
    labels = np.stack([v["price"], cx, v["vega"], -v["theta_cal"], v["d_kappa"], v["d_theta"], v["d_xi"],
                       v["d_rho"], S * S * v["gamma"] + cx, S * v["vanna"], v["volga"]], axis=1)
    return torch.as_tensor(labels)


def price_and_greeks(price_fn, x, put, chunk=2048):
    """Price and 11 sensitivities of a price-per-unit-strike function of contract rows."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    put = torch.as_tensor(np.asarray(put), dtype=torch.bool)
    out = {q: [] for q in QUANTITIES}
    col = COLUMN
    for start in range(0, len(x), chunk):
        xs = x[start:start + chunk].clone().requires_grad_(True)
        with torch.enable_grad():
            value = price_fn(xs, put[start:start + chunk])
            g, = torch.autograd.grad(value.sum(), xs, create_graph=True)
            rowS, = torch.autograd.grad(g[:, col["S"]].sum(), xs, retain_graph=True)
            rowv, = torch.autograd.grad(g[:, col["v0"]].sum(), xs)
        pieces = dict(price=value, delta=g[:, col["S"]], gamma=rowS[:, col["S"]], vega=g[:, col["v0"]],
                      vanna=rowS[:, col["v0"]], volga=rowv[:, col["v0"]], theta_cal=-g[:, col["tau"]],
                      rho_rate=g[:, col["r"]], d_kappa=g[:, col["kappa"]], d_theta=g[:, col["theta"]],
                      d_xi=g[:, col["xi"]], d_rho=g[:, col["rho"]])
        for q in QUANTITIES:
            out[q].append(pieces[q].detach())
    return {q: torch.cat(v).numpy() for q, v in out.items()}


def derivative_labels(call_fn, features, create_graph=False):
    """The network's c and derivatives at features, stacked as LABELS (Sobolev training)."""
    with torch.enable_grad():
        f = features.detach().clone().requires_grad_(True)
        call = call_fn(f)
        g, = torch.autograd.grad(call.sum(), f, create_graph=True)
        gx, = torch.autograd.grad(g[:, 0].sum(), f, create_graph=True)
        gv, = torch.autograd.grad(g[:, 1].sum(), f, create_graph=create_graph)
    return torch.stack((call, g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4], g[:, 5], g[:, 6],
                        gx[:, 0], gx[:, 1], gv[:, 1]), dim=1)
