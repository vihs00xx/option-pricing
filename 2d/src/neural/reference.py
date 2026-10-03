"""Batched conditional-quadrature reference with automatic-differentiation Greeks.

Carried over from the earlier two-stock neural study: the vectorised torch twin of
``src.basket.integration.conditional_samples``. The shared Gaussian factor is
integrated in closed form and the residual factor with a fixed trapezoidal rule on
[-limit, limit]. The nodes do not depend on the contract, so derivatives of the rule
are rules for the derivatives and the Greeks come from autograd. Tests compare it with
the adaptive ``quadrature_price`` and ``arithmetic_greeks``.

It labels DeepONet training data, anchors and validation sets. The frozen test set
keeps its own adaptive-quadrature references.
"""

import math

import numpy as np
import torch

from .coordinates import COLUMN, features_from_log_forwards, log_forwards, price_from_call, reduced_coordinates

# Order and names of the frozen test set's ``references.csv``.
QUANTITIES = ("price", "delta_1", "delta_2", "gamma_11", "gamma_12", "gamma_22",
              "vega_1", "vega_2", "theta", "rho", "correlation_12")
GREEKS = QUANTITIES[1:]


def trapezoid_rule(nodes=1001, limit=10.0, dtype=torch.float64):
    z = torch.linspace(-limit, limit, nodes, dtype=dtype)
    weights = torch.exp(-z*z/2)/math.sqrt(2*math.pi)*(z[1]-z[0])
    return z, weights


def normalized_call(f1, f2, a1, a2, rho, rule=None):
    """E[(f1 e^{a1 Z1 - a1^2/2} + f2 e^{a2 Z2 - a2^2/2} - 1)^+] with corr(Z1, Z2) = rho.

    Requires positive total volatilities and |rho| < 1 (the regular case).
    """
    z, weights = rule if rule is not None else trapezoid_rule(dtype=f1.dtype)
    cross = rho*a1*a2
    difference = a1*a1+a2*a2-2*cross
    root = torch.sqrt(difference)
    shared = a1*a1*a2*a2*(1-rho*rho)/difference
    b1, b2 = (a1*a1-cross)/root, (cross-a2*a2)/root
    sd = torch.sqrt(shared)[:, None]
    level = (f1[:, None]*torch.exp(b1[:, None]*z-(a1*a1/2)[:, None])
             + f2[:, None]*torch.exp(b2[:, None]*z-(a2*a2/2)[:, None]))
    d2 = torch.log(level)/sd
    inner = level*torch.exp(sd*sd/2)*torch.special.ndtr(d2+sd)-torch.special.ndtr(d2)
    return inner@weights


def reference_call(features, rule=None):
    """Normalized call c(z, s, tau; sigma_1, sigma_2, rho) of ``coordinates``."""
    z, s, tau, v1, v2, rho = features.unbind(-1)
    forward, root_tau = torch.exp(z), torch.sqrt(tau)
    return normalized_call(forward*s, forward*(1-s), v1*root_tau, v2*root_tau, rho, rule)


def call_pricer(call_fn):
    """Currency price function (x, put) -> price from a normalized-call function of features."""
    def price(x, put):
        features, scale = reduced_coordinates(x)
        return price_from_call(call_fn(features), features, scale, put)
    return price


def price_and_greeks(price_fn, x, put, chunk=2048, greeks=True):
    """Evaluate a currency price function and its Greeks in the test set's units.

    Vega, rho and correlation are per unit decimal change; theta is -dV/dtau per year.
    Returns a dict of float64 numpy arrays keyed by ``QUANTITIES``.
    """
    x = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    put = torch.as_tensor(np.asarray(put), dtype=torch.bool)
    names = QUANTITIES if greeks else ("price",)
    out = {name: [] for name in names}
    for start in range(0, len(x), chunk):
        xs = x[start:start+chunk].clone().requires_grad_(greeks)
        ps = put[start:start+chunk]
        with torch.set_grad_enabled(greeks):
            value = price_fn(xs, ps)
            out["price"].append(value.detach())
            if not greeks:
                continue
            gradient, = torch.autograd.grad(value.sum(), xs, create_graph=True)
            first = {name: gradient[:, COLUMN[field]] for name, field in (
                ("delta_1", "spot_1"), ("delta_2", "spot_2"), ("vega_1", "vol_1"),
                ("vega_2", "vol_2"), ("rho", "rate"), ("correlation_12", "correlation"))}
            first["theta"] = -gradient[:, COLUMN["tau"]]
            row_1, = torch.autograd.grad(first["delta_1"].sum(), xs, retain_graph=True)
            row_2, = torch.autograd.grad(first["delta_2"].sum(), xs)
        for name, value in first.items():
            out[name].append(value.detach())
        out["gamma_11"].append(row_1[:, COLUMN["spot_1"]])
        out["gamma_12"].append(row_1[:, COLUMN["spot_2"]])
        out["gamma_22"].append(row_2[:, COLUMN["spot_2"]])
    return {name: torch.cat(values).numpy() for name, values in out.items()}


def reference_values(x, put, greeks=True, nodes=1001, limit=10.0, chunk=1024):
    rule = trapezoid_rule(nodes, limit)
    return price_and_greeks(call_pricer(lambda f: reference_call(f, rule)), x, put, chunk, greeks)


# Derivative labels of the normalized call in the reduced coordinates, in this order.
LABELS = ("c", "c_1", "c_2", "c_11", "c_12", "c_22", "c_tau", "c_sigma_1", "c_sigma_2", "c_rho")


def derivative_labels(call_fn, features, chunk=4096, create_graph=False):
    """c and its derivatives in (y_1, y_2, tau, sigma_1, sigma_2, rho), stacked as ``LABELS``.

    With ``create_graph`` the result stays differentiable in the parameters of
    ``call_fn`` (Sobolev training); otherwise it is detached and chunked.
    """
    if create_graph:
        return _derivatives(call_fn, features, True)
    return torch.cat([_derivatives(call_fn, features[i:i+chunk], False).detach()
                      for i in range(0, len(features), chunk)])


def _derivatives(call_fn, features, create_graph):
    with torch.enable_grad():
        y1, y2 = log_forwards(features.detach())
        y1, y2 = y1.requires_grad_(True), y2.requires_grad_(True)
        rest = features[:, 2:].detach().clone().requires_grad_(True)
        call = call_fn(features_from_log_forwards(y1, y2, rest))
        c1, c2, cr = torch.autograd.grad(call.sum(), (y1, y2, rest), create_graph=True)
        c11, c12 = torch.autograd.grad(c1.sum(), (y1, y2), create_graph=True)
        c22, = torch.autograd.grad(c2.sum(), y2, create_graph=create_graph)
    return torch.stack((call, c1, c2, c11, c12, c22, cr[:, 0], cr[:, 1], cr[:, 2], cr[:, 3]), dim=1)


def reference_labels(features, nodes=1001, limit=10.0, chunk=4096):
    rule = trapezoid_rule(nodes, limit)
    return derivative_labels(lambda f: reference_call(f, rule), features, chunk)
