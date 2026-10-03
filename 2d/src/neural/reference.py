"""Batched conditional-quadrature reference prices with automatic-differentiation Greeks.

This is the vectorised torch twin of ``src.basket.integration.conditional_samples``:
the shared Gaussian factor is integrated in closed form and the residual factor
with a fixed trapezoidal rule on [-limit, limit]. Because the nodes do not depend
on the contract, derivatives of the rule are rules for the derivatives, so
Greeks come from autograd rather than bumps. Tests compare it with the adaptive
``quadrature_price`` and ``arithmetic_greeks``.
"""

import math

import numpy as np
import torch

from .domain import COLUMN, FIELDS, price_from_normalized, reduced_coordinates

GREEK_NAMES = ("delta_1", "delta_2", "gamma_11", "gamma_22", "gamma_12",
               "vega_1", "vega_2", "theta", "rho", "correlation")
QUANTITIES = ("price",)+GREEK_NAMES


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


def reference_price(x, put, rule=None):
    f1, f2, a1, a2, rho, _ = reduced_coordinates(x)
    return price_from_normalized(normalized_call(f1, f2, a1, a2, rho, rule), x, put)


def price_and_greeks(price_fn, x, put, chunk=2048, greeks=True):
    """Evaluate a currency price function and its Greeks in the repository's units.

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
                ("vega_2", "vol_2"), ("rho", "rate"), ("correlation", "correlation"))}
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
    return price_and_greeks(lambda xs, ps: reference_price(xs, ps, rule), x, put, chunk, greeks)


def contract_row(contract):
    """Map a ``BasketContract`` with positive weights summing to one onto ``FIELDS``."""
    values = dict(spot_1=contract.spots[0], spot_2=contract.spots[1], strike=contract.strike,
                  tau=contract.tau, rate=contract.rate, dividend_1=contract.dividends[0],
                  dividend_2=contract.dividends[1], vol_1=contract.volatilities[0],
                  vol_2=contract.volatilities[1], correlation=contract.correlation,
                  weight_1=contract.weights[0])
    return np.array([values[name] for name in FIELDS], float), contract.option == "put"
