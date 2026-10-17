"""Residuals of the d-stock Heston basket pricing PDE, by automatic differentiation.

Reduced form, for the normalized call c in y_i = log f_i, the variances v_i and tau (see
``coordinates``):

    c_tau = sum_i [1/2 v_i (c_{y_i y_i} - c_{y_i}) + kappa_i (theta_i - v_i) c_{v_i}
                   + 1/2 xi_i^2 v_i c_{v_i v_i}]
          + 1/2 sum_{i != j} sqrt(v_i v_j) [R_ij c_{y_i y_j} + xi_i xi_j C_ij c_{v_i v_j}]
          + sum_{i, j} sqrt(v_i v_j) rho_i xi_j C_ij c_{y_i v_j}   (C_ii = 1).

Raw form, for V = price / K in spots, variances, maturity and the contract parameters:

    V_tau = 1/2 sum_ij sqrt(v_i v_j) R_ij S_i S_j V_{S_i S_j} + sum_i (r - q_i) S_i V_{S_i}
          + sum_i kappa_i (theta_i - v_i) V_{v_i} + 1/2 sum_ij sqrt(v_i v_j) xi_i xi_j C_ij V_{v_i v_j}
          + sum_ij sqrt(v_i v_j) rho_i xi_j C_ij S_i V_{S_i v_j} - r V.
"""

import torch

from .coordinates import features_from_log_forwards, log_forwards


def _full(stocks, entries):
    return stocks.correlation(entries)


def reduced_derivatives(stocks, call_fn, features, second=True):
    """c and its derivatives in (y, v, tau): dict with c, y (N, d), v (N, d), tau (N,) and,
    with ``second``, yy, yv, vv (N, d, d) (yv[:, i, j] = c_{y_i v_j})."""
    d = stocks.d
    f = features.detach()
    y = log_forwards(stocks, f).requires_grad_(True)
    v = f[:, stocks.v].clone().requires_grad_(True)
    tau = f[:, stocks.tau].clone().requires_grad_(True)
    rest = torch.cat((v, tau[:, None], f[:, stocks.tau+1:]), dim=1)
    call = call_fn(features_from_log_forwards(stocks, y, rest))
    cy, cv, ct = torch.autograd.grad(call.sum(), (y, v, tau), create_graph=True)
    out = dict(c=call, y=cy, v=cv, tau=ct)
    if second:
        yy, yv, vv = [], [], []
        for i in range(d):
            row_y, row_v = torch.autograd.grad(cy[:, i].sum(), (y, v), create_graph=True)
            yy.append(row_y)
            yv.append(row_v)
            vv.append(torch.autograd.grad(cv[:, i].sum(), v, create_graph=True)[0])
        out.update(yy=torch.stack(yy, 1), yv=torch.stack(yv, 1), vv=torch.stack(vv, 1))
    return out


def generator(stocks, p, derivatives):
    """The PDE's spatial operator L c from split features ``p`` and ``reduced_derivatives``."""
    v = p["v"].clamp_min(0.0)
    root = torch.sqrt(v)
    outer = root[:, :, None]*root[:, None, :]
    R, C = _full(stocks, p["rho"]), _full(stocks, p["eta"])
    xi, rho = p["xi"], p["leverage"]
    g = derivatives
    diagonal_y = torch.diagonal(g["yy"], dim1=1, dim2=2)
    out = (0.5*v*(diagonal_y-g["y"])).sum(1)+(p["kappa"]*(p["theta"]-v)*g["v"]).sum(1)
    out = out+0.5*torch.einsum("nij,nij->n", outer*R, g["yy"]-torch.diag_embed(diagonal_y))
    vol = xi[:, :, None]*xi[:, None, :]*C
    out = out+0.5*torch.einsum("nij,nij->n", outer*vol, g["vv"])
    cross = rho[:, :, None]*xi[:, None, :]*C
    return out+torch.einsum("nij,nij->n", outer*cross, g["yv"])


def reduced_residual(stocks, call_fn, features):
    """Residual c_tau - L c in (y, v, tau). Returns (residual, call)."""
    g = reduced_derivatives(stocks, call_fn, features)
    p = stocks.split(features.detach())
    return g["tau"]-generator(stocks, p, g), g["c"]


def variance_slope(stocks, call_fn, features):
    """c_{v_i} (N, d), for the c_v = 0 condition at the variance cap."""
    return reduced_derivatives(stocks, call_fn, features, second=False)["v"]


def raw_residual(stocks, normalized_price_fn, x, put):
    """Residual V_tau - L V for price over strike in the raw contract inputs."""
    d, col = stocks.d, stocks.column
    x = x.detach().clone().requires_grad_(True)
    value = normalized_price_fn(x, put)
    gradient, = torch.autograd.grad(value.sum(), x, create_graph=True)
    c = x.detach()
    S, v = stocks.take(c, "spot_"), stocks.take(c, "v0_").clamp_min(0.0)
    kappa, theta, xi, rho = (stocks.take(c, f"{n}_") for n in ("kappa", "theta", "xi", "leverage"))
    R, C = stocks.correlation(stocks.take(c, "correlation_")), stocks.correlation(stocks.take(c, "vol_correlation_"))
    root = torch.sqrt(v)
    rate = c[:, col["rate"]]
    out = -rate*value
    for i in range(d):
        V_S = gradient[:, col[f"spot_{i+1}"]]
        V_v = gradient[:, col[f"v0_{i+1}"]]
        out = out+(rate-c[:, col[f"dividend_{i+1}"]])*S[:, i]*V_S+kappa[:, i]*(theta[:, i]-v[:, i])*V_v
        row_S, = torch.autograd.grad(V_S.sum(), x, create_graph=True)
        row_v, = torch.autograd.grad(V_v.sum(), x, create_graph=True)
        for j in range(d):
            scale = root[:, i]*root[:, j]
            out = out+0.5*scale*R[:, i, j]*S[:, i]*S[:, j]*row_S[:, col[f"spot_{j+1}"]]
            out = out+0.5*scale*xi[:, i]*xi[:, j]*C[:, i, j]*row_v[:, col[f"v0_{j+1}"]]
            out = out+scale*rho[:, i]*xi[:, j]*C[:, i, j]*S[:, i]*row_S[:, col[f"v0_{j+1}"]]
    return gradient[:, col["tau"]]-out
