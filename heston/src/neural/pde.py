"""Residuals of the Heston pricing PDE by automatic differentiation.

Reduced form, for the normalized call c(x, v, tau; kappa, theta, xi, rho):

    c_tau = 1/2 v (c_xx - c_x) + rho xi v c_xv + 1/2 xi^2 v c_vv + kappa (theta - v) c_v,

which at v = 0 is the degenerate first-order equation c_tau = kappa theta c_v.

Raw form, for V = price / K in (S, v, tau) and the contract parameters:

    V_tau = 1/2 v S^2 V_SS + rho xi v S V_Sv + 1/2 xi^2 v V_vv + (r-q) S V_S + kappa (theta - v) V_v - r V.
"""

import torch

from .coordinates import COLUMN, FIELDS


def _reduced_derivatives(call_fn, features, second=True):
    f = features.detach()
    x = f[:, 0].clone().requires_grad_(True)
    v = f[:, 1].clone().requires_grad_(True)
    tau = f[:, 2].clone().requires_grad_(True)
    call = call_fn(torch.cat((x[:, None], v[:, None], tau[:, None], f[:, 3:]), dim=1))
    cx, cv, ct = torch.autograd.grad(call.sum(), (x, v, tau), create_graph=True)
    out = dict(c=call, x=cx, v=cv, tau=ct)
    if second:
        cxx, cxv = torch.autograd.grad(cx.sum(), (x, v), create_graph=True)
        cvv, = torch.autograd.grad(cv.sum(), v, create_graph=True)
        out.update(xx=cxx, xv=cxv, vv=cvv)
    return out


def reduced_residual(call_fn, features):
    """Residual c_tau - L c at (N, 7) features. Returns (residual, call)."""
    d = _reduced_derivatives(call_fn, features)
    v, kappa, theta, xi, rho = (features[:, k].detach() for k in (1, 3, 4, 5, 6))
    generator = (0.5 * v * (d["xx"] - d["x"]) + rho * xi * v * d["xv"] + 0.5 * xi * xi * v * d["vv"]
                 + kappa * (theta - v) * d["v"])
    return d["tau"] - generator, d["c"]


def zero_variance_residual(call_fn, features):
    """Residual of the degenerate equation c_tau - kappa theta c_v on v = 0 (features[:, 1] set to 0)."""
    d = _reduced_derivatives(call_fn, features, second=False)
    kappa, theta = features[:, 3].detach(), features[:, 4].detach()
    return d["tau"] - kappa * theta * d["v"]


def variance_slope(call_fn, features):
    """c_v, for the V_v = 0 condition at v_max."""
    return _reduced_derivatives(call_fn, features, second=False)["v"]


def raw_residual(normalized_price_fn, x, put):
    """Residual V_tau - L V for price / K in the raw contract inputs."""
    x = x.detach().clone().requires_grad_(True)
    value = normalized_price_fn(x, put)
    g, = torch.autograd.grad(value.sum(), x, create_graph=True)
    VS, Vv = g[:, COLUMN["S"]], g[:, COLUMN["v0"]]
    rowS, = torch.autograd.grad(VS.sum(), x, create_graph=True)
    rowv, = torch.autograd.grad(Vv.sum(), x, create_graph=True)
    c = {name: x[:, i].detach() for i, name in enumerate(FIELDS)}
    S, v = c["S"], c["v0"]
    generator = (0.5 * v * S * S * rowS[:, COLUMN["S"]] + c["rho"] * c["xi"] * v * S * rowS[:, COLUMN["v0"]]
                 + 0.5 * c["xi"] ** 2 * v * rowv[:, COLUMN["v0"]] + (c["r"] - c["q"]) * S * VS
                 + c["kappa"] * (c["theta"] - v) * Vv - c["r"] * value)
    return g[:, COLUMN["tau"]] - generator
