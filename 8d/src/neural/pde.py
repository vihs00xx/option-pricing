"""Residuals of the d-asset Black-Scholes pricing PDE, by automatic differentiation.

Raw form, for V = price / K in spots, maturity tau and the contract parameters:

    V_tau = 1/2 sum_ij rho_ij s_i s_j S_i S_j V_ij + sum_i (r - q_i) S_i V_i - r V.

Reduced form, for the normalized call c in y_i = log f_i (see ``coordinates``):

    c_tau = 1/2 sum_i s_i^2 (c_ii - c_i) + sum_{i<j} rho_ij s_i s_j c_ij.
"""

import torch

from .coordinates import features_from_log_forwards, log_forwards


def reduced_residual(stocks, call_fn, features):
    """Residual c_tau - L c in (y, tau).

    ``call_fn`` maps (N, F) features to (N,) normalized calls. Returns (residual, call).
    """
    d = stocks.d
    y = log_forwards(stocks, features.detach()).requires_grad_(True)
    tau = features[:, d].detach().clone().requires_grad_(True)
    rest = torch.cat((tau[:, None], features[:, d+1:].detach()), dim=1)
    call = call_fn(features_from_log_forwards(stocks, y, rest))
    first, c_tau = torch.autograd.grad(call.sum(), (y, tau), create_graph=True)
    _, _, _, sigma, rho = stocks.split(features.detach())
    generator = 0
    for i in range(d):
        row, = torch.autograd.grad(first[:, i].sum(), y, create_graph=True)
        generator = generator+0.5*sigma[:, i]**2*(row[:, i]-first[:, i])
        for j in range(i+1, d):
            generator = generator+rho[:, stocks.pairs.index((i, j))]*sigma[:, i]*sigma[:, j]*row[:, j]
    return c_tau-generator, call


def raw_residual(stocks, normalized_price_fn, x, put):
    """Residual V_tau - L V for price over strike in the raw contract inputs."""
    d, col = stocks.d, stocks.column
    x = x.detach().clone().requires_grad_(True)
    value = normalized_price_fn(x, put)
    gradient, = torch.autograd.grad(value.sum(), x, create_graph=True)
    c = x.detach()
    spots, vols = stocks.take(c, "spot_"), stocks.take(c, "vol_")
    rate = c[:, col["rate"]]
    generator = -rate*value
    for i in range(d):
        v_i = gradient[:, col[f"spot_{i+1}"]]
        generator = generator+(rate-c[:, col[f"dividend_{i+1}"]])*spots[:, i]*v_i
        row, = torch.autograd.grad(v_i.sum(), x, create_graph=True)
        for j in range(i, d):
            rho = 1.0 if i == j else c[:, col[f"correlation_{i+1}{j+1}"]]
            factor = 0.5 if i == j else 1.0
            generator = generator+factor*rho*vols[:, i]*vols[:, j]*spots[:, i]*spots[:, j]*row[:, col[f"spot_{j+1}"]]
    return gradient[:, col["tau"]]-generator
