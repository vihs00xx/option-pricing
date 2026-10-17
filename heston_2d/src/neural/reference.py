"""References, training labels and autograd Greeks for the Heston basket pricers.

There is no closed form or quadrature for a Heston basket, so every reference is the
conditional QE simulation of ``src/heston_basket/engine.py``:

``reference_values``  per-contract RQMC estimates (``heston_estimate``) with standard errors,
                      all quantities, run in worker processes; for monitor, validation and
                      the test set.
``reference_labels``  the normalized call and its first and second derivatives in the log
                      forwards for a batch of contracts on one shared Sobol net (fixed
                      normals and steps), so the labels' simulation error is a smooth
                      function of the contract, as with a fixed quadrature rule. These
                      derivatives are analytic per path; parameter derivatives would need
                      bumps of the QE paths and are not labelled.

``price_and_greeks`` evaluates any currency price function (the networks) and its Greeks
by automatic differentiation, in the engine's units and order.
"""

import os
from multiprocessing import Pool

import numpy as np
import torch

from src.heston_basket.engine import draw_normals, heston_estimate, normalized_paths
from src.heston_basket.model import quantity_names

from .coordinates import contract_of, features_from_log_forwards, log_forwards, matrices


def quantities(stocks):
    """Order and names of the engine's quantities (``src.heston_basket.model.quantity_names``)."""
    return tuple(quantity_names(stocks.d))


def labels(stocks):
    """Derivative labels of the normalized call in the log forwards y, in this order."""
    d, n = stocks.d, range(1, stocks.d+1)
    return (("c",)+tuple(f"c_{i}" for i in n)+tuple(f"c_{i+1}{j+1}" for i in range(d) for j in range(i, d)))


# ---------------------------------------------------------------- engine references
def _estimate(task):
    row, put, settings, seed, d = task
    from .coordinates import Stocks
    c = contract_of(Stocks(d), row, put)
    e = heston_estimate(c, settings["points"], settings["replicates"], seed, dt=settings["dt"],
                        greeks=settings.get("greeks", True), sampler=settings.get("sampler", "sobol"))
    return [e.values[q] for q in e.names], [e.standard_errors[q] for q in e.names]


def reference_values(stocks, x, put, settings, seed, workers=None):
    """Engine estimates for contract rows: (values, standard_errors), dicts of arrays.

    ``settings`` holds points, replicates and dt (and optionally greeks, sampler); contract
    k uses seed + k. ``workers`` processes share the contracts (default: all cores).
    """
    x, put = np.asarray(x), np.asarray(put)
    tasks = [(x[k], bool(put[k]), settings, seed+k, stocks.d) for k in range(len(x))]
    workers = workers or os.cpu_count()
    if workers > 1 and len(tasks) > 1:
        with Pool(workers) as pool:
            rows = pool.map(_estimate, tasks, chunksize=max(1, len(tasks)//(8*workers)))
    else:
        rows = [_estimate(task) for task in tasks]
    names = quantities(stocks) if settings.get("greeks", True) else ("price",)
    values = {q: np.array([r[0][k] for r in rows]) for k, q in enumerate(names)}
    errors = {q: np.array([r[1][k] for r in rows]) for k, q in enumerate(names)}
    return values, errors


def _parameter_arrays(stocks, features):
    p = stocks.split(torch.as_tensor(features))
    out = {k: v.detach().numpy() for k, v in p.items()}
    d = stocks.d
    C = matrices(out["eta"], d)
    R = matrices(out["rho"], d)
    rho = out["leverage"]
    rbar = np.sqrt(1-rho**2)
    P = (R-rho[:, :, None]*rho[:, None, :]*C)/(rbar[:, :, None]*rbar[:, None, :])
    P[:, np.arange(d), np.arange(d)] = 1.0
    return out, C, P


def reference_labels(stocks, features, settings, chunk=None):
    """Labels (``labels(stocks)`` order) of the normalized call for reduced features (N, F).

    ``settings`` holds points (a power of two), steps and seed: one Sobol net with a fixed
    number of QE steps for every contract, so contract k's maturity sets its step size.
    """
    d = stocks.d
    features = torch.as_tensor(features).detach()
    U, increments = draw_normals(int(settings["points"]), d, int(settings["steps"]), int(settings["seed"]))
    chunk = chunk or max(1, (1 << 21)//(int(settings["points"])*d*(int(settings["steps"])+1)))
    rows = np.triu_indices(d)
    out = []
    for start in range(0, len(features), chunk):
        part = features[start:start+chunk]
        p, C, P = _parameter_arrays(stocks, part)
        log_f = log_forwards(stocks, part).numpy()
        value, first, second = normalized_paths(log_f, p["v"], p["kappa"], p["theta"], p["xi"], p["leverage"],
                                                C, P, p["tau"], U, increments, 1, derivatives=True)
        value, first, second = value.mean(1), first.mean(1), second.mean(1)
        f = np.exp(log_f)
        c_y = f*first
        c_yy = f[:, :, None]*f[:, None, :]*second+np.einsum("ni,ij->nij", c_y, np.eye(d))
        out.append(np.column_stack((value, c_y, c_yy[:, rows[0], rows[1]])))
    return torch.as_tensor(np.concatenate(out))


# ---------------------------------------------------------------- autograd Greeks of a price function
def price_and_greeks(stocks, price_fn, x, put, chunk=256, greeks=True):
    """Evaluate a currency price function and its Greeks in the engine's units.

    Vega is d/dv0 per stock; the parameter sensitivities are per unit change; theta is
    -dV/dtau per year; rho is per unit rate. Returns a dict of float64 numpy arrays keyed
    by ``quantities(stocks)``.
    """
    d, col = stocks.d, stocks.column
    x = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    put = torch.as_tensor(np.asarray(put), dtype=torch.bool)
    names = quantities(stocks) if greeks else ("price",)
    out = {name: [] for name in names}
    n = range(1, d+1)
    first_order = ([(f"delta_{i}", f"spot_{i}") for i in n]+[("rho", "rate")]
                   + [(f"vega_{i}", f"v0_{i}") for i in n]
                   + [(f"d_{p}_{i}", f"{p}_{i}") for p in ("kappa", "theta", "xi", "leverage") for i in n]
                   + [(f"correlation_{i+1}{j+1}",)*2 for i, j in stocks.pairs]
                   + [(f"vol_correlation_{i+1}{j+1}",)*2 for i, j in stocks.pairs])
    for start in range(0, len(x), chunk):
        xs = x[start:start+chunk].clone().requires_grad_(greeks)
        ps = put[start:start+chunk]
        with torch.set_grad_enabled(greeks):
            value = price_fn(xs, ps)
            out["price"].append(value.detach())
            if not greeks:
                continue
            gradient, = torch.autograd.grad(value.sum(), xs, create_graph=True)
            for name, field in first_order:
                out[name].append(gradient[:, col[field]].detach())
            out["theta"].append(-gradient[:, col["tau"]].detach())
            for i in range(d):
                row, = torch.autograd.grad(gradient[:, col[f"spot_{i+1}"]].sum(), xs, retain_graph=i < d-1)
                for j in range(i, d):
                    out[f"gamma_{i+1}{j+1}"].append(row[:, col[f"spot_{j+1}"]])
    return {name: torch.cat(values).numpy() for name, values in out.items()}


def derivative_labels(stocks, call_fn, features, chunk=4096, create_graph=False):
    """c and its first and second derivatives in y, stacked as ``labels(stocks)``.

    With ``create_graph`` the result stays differentiable in the parameters of
    ``call_fn`` (Sobolev training); otherwise it is detached and chunked.
    """
    if create_graph:
        return _derivatives(stocks, call_fn, features, True)
    return torch.cat([_derivatives(stocks, call_fn, features[i:i+chunk], False).detach()
                      for i in range(0, len(features), chunk)])


def _derivatives(stocks, call_fn, features, create_graph):
    d = stocks.d
    with torch.enable_grad():
        y = log_forwards(stocks, features.detach()).requires_grad_(True)
        rest = features[:, d:].detach()
        call = call_fn(features_from_log_forwards(stocks, y, rest))
        first, = torch.autograd.grad(call.sum(), y, create_graph=True)
        second = []
        for i in range(d):
            row, = torch.autograd.grad(first[:, i].sum(), y, create_graph=create_graph, retain_graph=True)
            second += [row[:, j] for j in range(i, d)]
    return torch.cat((call[:, None], first, torch.stack(second, dim=1)), dim=1)


def symmetric(stocks, upper):
    """(N, d, d) symmetric matrices from upper-triangle columns with the diagonal."""
    d = stocks.d
    index = [(i, j) for i in range(d) for j in range(i, d)]
    out = upper.new_zeros(len(upper), d, d)
    for k, (i, j) in enumerate(index):
        out[:, i, j] = out[:, j, i] = upper[:, k]
    return out, index


def permute_labels(stocks, values, order):
    """Labels of ``coordinates.permute``d features: new stock k is old order[k]."""
    d = stocks.d
    rows = order[:, :, None].expand(-1, -1, d)
    columns = order[:, None, :].expand(-1, d, -1)
    full, index = symmetric(stocks, values[:, 1+d:])
    full = torch.gather(torch.gather(full, 1, rows), 2, columns)
    index = torch.tensor(index)
    return torch.cat((values[:, :1], torch.gather(values[:, 1:1+d], 1, order), full[:, index[:, 0], index[:, 1]]),
                     dim=1)
