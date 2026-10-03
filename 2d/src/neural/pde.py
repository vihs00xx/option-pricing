"""Residuals of the two-asset Black-Scholes pricing PDE, by automatic differentiation.

Raw form, for V = price / K in spots, maturity tau and the contract parameters:

    V_tau = 1/2 s1^2 S_1^2 V_11 + rho s1 s2 S_1 S_2 V_12 + 1/2 s2^2 S_2^2 V_22
            + (r - q_1) S_1 V_1 + (r - q_2) S_2 V_2 - r V.

Reduced form, for the normalized call c in y_i = log f_i (see ``coordinates``):

    c_tau = 1/2 s1^2 (c_11 - c_1) + rho s1 s2 c_12 + 1/2 s2^2 (c_22 - c_2).
"""

import torch

from .coordinates import COLUMN, FIELDS, features_from_log_forwards, log_forwards


def reduced_residual(call_fn, features):
    """Residual c_tau - L c in (y_1, y_2, tau).

    ``call_fn`` maps (N, 6) features to (N,) normalized calls. Returns (residual, call).
    """
    y1, y2 = log_forwards(features.detach())
    y1, y2 = y1.requires_grad_(True), y2.requires_grad_(True)
    tau = features[:, 2].detach().clone().requires_grad_(True)
    rest = torch.cat((tau[:, None], features[:, 3:].detach()), dim=1)
    call = call_fn(features_from_log_forwards(y1, y2, rest))
    c1, c2, ct = torch.autograd.grad(call.sum(), (y1, y2, tau), create_graph=True)
    c11, c12 = torch.autograd.grad(c1.sum(), (y1, y2), create_graph=True)
    c22, = torch.autograd.grad(c2.sum(), y2, create_graph=True)
    v1, v2, rho = features[:, 3].detach(), features[:, 4].detach(), features[:, 5].detach()
    generator = 0.5*v1*v1*(c11-c1)+rho*v1*v2*c12+0.5*v2*v2*(c22-c2)
    return ct-generator, call


def raw_residual(normalized_price_fn, x, put):
    """Residual V_tau - L V for price over strike in the raw contract inputs."""
    x = x.detach().clone().requires_grad_(True)
    value = normalized_price_fn(x, put)
    gradient, = torch.autograd.grad(value.sum(), x, create_graph=True)
    v1, v2 = gradient[:, COLUMN["spot_1"]], gradient[:, COLUMN["spot_2"]]
    row_1, = torch.autograd.grad(v1.sum(), x, create_graph=True)
    row_2, = torch.autograd.grad(v2.sum(), x, create_graph=True)
    c = {name: x[:, i].detach() for i, name in enumerate(FIELDS)}
    s1, s2, a1, a2 = c["spot_1"], c["spot_2"], c["vol_1"], c["vol_2"]
    generator = (0.5*a1*a1*s1*s1*row_1[:, COLUMN["spot_1"]]
                 + c["correlation"]*a1*a2*s1*s2*row_1[:, COLUMN["spot_2"]]
                 + 0.5*a2*a2*s2*s2*row_2[:, COLUMN["spot_2"]]
                 + (c["rate"]-c["dividend_1"])*s1*v1+(c["rate"]-c["dividend_2"])*s2*v2
                 - c["rate"]*value)
    return gradient[:, COLUMN["tau"]]-generator
