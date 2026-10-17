"""Batched conditional reference with automatic-differentiation Greeks.

The torch twin of the numpy engine's conditional estimator (``src.pricing.conditional``,
direction "volatility", arithmetic basket) in the reduced coordinates. The log forwards
are written as X = log f - a^2/2 + b Y + A U with a_i = sigma_i sqrt(tau); the factor Y is
integrated in closed form given U, and U ~ N(0, I_{d-1}) by a fixed rule ``(U, mass)``:

``gauss_hermite``  the tensor Gauss-Hermite rule of ``gauss_hermite_values``; n^(d-1)
                   points, the four-stock reference, infeasible for eight stocks.
``sobol``          one scrambled Sobol net of 2^m normal points (equal masses), with the
                   geometric control: the conditional value of the geometric basket
                   G = exp(sum_i s_i log(X_i / s_i)) <= B on the same points is
                   subtracted and its exact mean added back, which removes most of the
                   rule's error (B - G is small and smooth).

The nodes do not depend on the contract, so derivatives of the rule are rules for the
derivatives and the Greeks come from autograd. A Sobol rule is one fixed draw: its error
is a smooth function of the contract, not noise, and it is shared by every contract
priced with it.

The strike root y* is solved without gradients and then refined by one Newton step
with gradients. The step leaves the value at the root and gives dy*/dtheta exactly;
the conditional value is stationary in y at the root, so first and second derivatives
of the price are exact as well.

It labels training data, anchors and validation sets. The frozen test set keeps the
numpy engine's RQMC references (independent replicates, bumped parameter Greeks).
"""

import numpy as np
import torch

from .coordinates import features_from_log_forwards, log_forwards, price_from_call, reduced_coordinates


def quantities(stocks):
    """Order and names of the frozen test set's ``references.csv`` (engine order)."""
    d, n = stocks.d, range(1, stocks.d+1)
    return (("price",)+tuple(f"delta_{i}" for i in n)
            + tuple(f"gamma_{i+1}{j+1}" for i in range(d) for j in range(i, d))
            + tuple(f"vega_{i}" for i in n)+("theta", "rho")
            + tuple(f"correlation_{i+1}{j+1}" for i, j in stocks.pairs))


def labels(stocks):
    """Derivative labels of the normalized call in (y, tau, sigma, rho), in this order."""
    d, n = stocks.d, range(1, stocks.d+1)
    return (("c",)+tuple(f"c_{i}" for i in n)
            + tuple(f"c_{i+1}{j+1}" for i in range(d) for j in range(i, d))
            + ("c_tau",)+tuple(f"c_sigma_{i}" for i in n)
            + tuple(f"c_rho_{i+1}{j+1}" for i, j in stocks.pairs))


def gauss_hermite_rule(nodes, k, min_mass=1e-24, dtype=torch.float64):
    """Tensor rule for E[g(U)], U ~ N(0, I_k), in the engine's node order.

    Corner nodes whose product weight is at most ``min_mass`` are dropped: at 32 nodes
    in three dimensions this keeps 60% of the grid and drops a total mass of 4e-22.
    """
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w/np.sqrt(2*np.pi)
    grids = np.meshgrid(*([x]*k), indexing="ij")
    U = np.column_stack([g.ravel() for g in grids])
    mass = np.prod(np.meshgrid(*([w]*k), indexing="ij"), axis=0).ravel()
    keep = mass > min_mass
    return torch.as_tensor(U[keep], dtype=dtype), torch.as_tensor(mass[keep], dtype=dtype)


def factorization(stocks, a, rho):
    """Loadings b (N, d) on the smoothing factor and A (N, d, d-1) on the residual.

    Direction t = a, which gives the same b and A as the engine's t = sigma.
    """
    covariance = a[:, :, None]*stocks.correlation(rho)*a[:, None, :]
    lower = torch.linalg.cholesky(covariance)
    u = torch.linalg.solve_triangular(lower, a[:, :, None], upper=False)[:, :, 0]
    norm = torch.linalg.vector_norm(u, dim=1, keepdim=True)
    v = u/norm
    h = v+torch.nn.functional.one_hot(torch.zeros(len(a), dtype=torch.long), stocks.d).to(a.dtype)
    reflection = torch.eye(stocks.d, dtype=a.dtype)-2*h[:, :, None]*h[:, None, :]/(h*h).sum(1)[:, None, None]
    return a/norm, lower@reflection[:, :, 1:]


@torch.no_grad()
def strike_root(log_c, b):
    """Solve sum_i exp(log_c_i + b_i y) = 1 along the last axis; Newton from the right.

    The engine's ``_root`` with unit strike: the old stopping test first, then a
    repeated iterate ends the loop and steps within the rounding floor are accepted.
    """
    y = ((-log_c)/b).min(-1).values
    recent = []
    for _ in range(100):
        e = log_c+b*y[..., None]
        top = e.max(-1, keepdim=True).values
        scaled = torch.exp(e-top)
        total = scaled.sum(-1)
        slope = (scaled*b).sum(-1)/total
        step = (top[..., 0]+torch.log(total))/slope
        y = y-step
        if bool((step.abs() <= 1e-13*(1+y.abs())).all()):
            return y
        if any(torch.equal(y, old) for old in recent):
            break
        recent = recent[-1:]+[y]
    floor = 8*torch.finfo(y.dtype).eps*(1+top[..., 0].abs())/slope
    if bool((step.abs() <= torch.maximum(1e-13*(1+y.abs()), floor)).all()):
        return y
    raise RuntimeError("Conditional strike root did not converge")


def sobol_rule(points, k, seed, dtype=torch.float64):
    """One scrambled Sobol net of ``points`` standard normal points in k dimensions."""
    from src.pricing.rqmc import normal_points
    U = normal_points(int(points), k, int(seed))
    return torch.as_tensor(U, dtype=dtype), torch.full((int(points),), 1.0/int(points), dtype=dtype)


def make_rule(stocks, rule):
    """``(U, mass, control)`` from settings: an int is a Gauss-Hermite node count (no control);
    a dict is {"kind": "gauss_hermite", "nodes": n} or {"kind": "sobol", "points": 2^m, "seed": s}
    with an optional "control" (default True for Sobol)."""
    if isinstance(rule, (int, np.integer)):
        rule = dict(kind="gauss_hermite", nodes=int(rule))
    k = stocks.d-1
    if rule["kind"] == "gauss_hermite":
        U, mass = gauss_hermite_rule(rule["nodes"], k, rule.get("min_mass", 1e-24))
        return U, mass, bool(rule.get("control", False))
    if rule["kind"] == "sobol":
        U, mass = sobol_rule(rule["points"], k, rule["seed"])
        return U, mass, bool(rule.get("control", True))
    raise ValueError(f"unknown rule {rule!r}")


def black_value(log_mean, sd):
    """E[(exp(N(log_mean, sd^2)) - 1)^+] for sd > 0."""
    d1 = (log_mean+sd*sd)/sd
    return torch.exp(log_mean+sd*sd/2)*torch.special.ndtr(d1)-torch.special.ndtr(d1-sd)


def normalized_call(stocks, features, rule):
    """c(z, s, tau; sigma, rho) by the conditional rule ``(U, mass[, control])``."""
    U, mass = rule[0], rule[1]
    control = len(rule) > 2 and rule[2]
    z, s, tau, sigma, rho = stocks.split(features)
    a = sigma*torch.sqrt(tau)[:, None]
    b, A = factorization(stocks, a, rho)
    shift = torch.einsum("mk,nik->nmi", U, A)
    log_c = (z[:, None]+torch.log(s)-a*a/2)[:, None, :]+shift
    b = b[:, None, :]
    y = strike_root(log_c.detach(), b.detach())
    exponent = log_c+b*y[..., None]
    y = y-torch.logsumexp(exponent, -1)/(torch.softmax(exponent, -1)*b).sum(-1)
    value = (torch.exp(log_c+b*b/2)*torch.special.ndtr(b-y[..., None])).sum(-1)-torch.special.ndtr(-y)
    if not control:
        return value@mass
    # Geometric control: log G = z - sum s_i a_i^2/2 + s.(A U) + (s.b) Y, so given U it is a
    # Black call; unconditionally log G ~ N(z - sum s_i a_i^2/2, s' Sigma s).
    centre = z-(s*a*a).sum(1)/2
    loading = (s[:, None, :]*b).sum(-1)
    conditional = black_value(centre[:, None]+(s[:, None, :]*shift).sum(-1), loading)
    covariance = a[:, :, None]*stocks.correlation(rho)*a[:, None, :]
    exact = black_value(centre, torch.sqrt(torch.einsum("ni,nij,nj->n", s, covariance, s)))
    return (value-conditional)@mass+exact


def call_function(stocks, nodes=32, min_mass=1e-24):
    """Normalized-call function of features for a rule: a Gauss-Hermite node count or rule settings."""
    rule = make_rule(stocks, dict(kind="gauss_hermite", nodes=nodes, min_mass=min_mass)
                     if isinstance(nodes, (int, np.integer)) else nodes)
    return lambda features: normalized_call(stocks, features, rule)


def default_chunk(stocks, nodes, min_mass=1e-24):
    """Contracts per batch so that a batch holds about 2^17 rule points."""
    if isinstance(nodes, (int, np.integer)):
        size = len(gauss_hermite_rule(nodes, stocks.d-1, min_mass)[1])
    elif nodes["kind"] == "sobol":
        size = int(nodes["points"])
    else:
        size = len(gauss_hermite_rule(nodes["nodes"], stocks.d-1, nodes.get("min_mass", min_mass))[1])
    return max(1, (1 << 17)//size)


def call_pricer(stocks, call_fn):
    """Currency price function (x, put) -> price from a normalized-call function of features."""
    def price(x, put):
        features, scale = reduced_coordinates(stocks, x)
        return price_from_call(call_fn(features), features, scale, put)
    return price


def price_and_greeks(stocks, price_fn, x, put, chunk=256, greeks=True):
    """Evaluate a currency price function and its Greeks in the test set's units.

    Vega, rho and correlation are per unit decimal change; theta is -dV/dtau per year.
    Returns a dict of float64 numpy arrays keyed by ``quantities(stocks)``.
    """
    d, col = stocks.d, stocks.column
    x = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    put = torch.as_tensor(np.asarray(put), dtype=torch.bool)
    names = quantities(stocks) if greeks else ("price",)
    out = {name: [] for name in names}
    first_order = ([(f"delta_{i}", f"spot_{i}") for i in range(1, d+1)]
                   + [(f"vega_{i}", f"vol_{i}") for i in range(1, d+1)] + [("rho", "rate")]
                   + [(f"correlation_{i+1}{j+1}",)*2 for i, j in stocks.pairs])
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


def reference_values(stocks, x, put, greeks=True, nodes=32, chunk=None):
    chunk = chunk or default_chunk(stocks, nodes)
    return price_and_greeks(stocks, call_pricer(stocks, call_function(stocks, nodes)), x, put, chunk, greeks)


def derivative_labels(stocks, call_fn, features, chunk=4096, create_graph=False):
    """c and its derivatives in (y, tau, sigma, rho), stacked as ``labels(stocks)``.

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
        rest = features[:, d:].detach().clone().requires_grad_(True)
        call = call_fn(features_from_log_forwards(stocks, y, rest))
        first, parameters = torch.autograd.grad(call.sum(), (y, rest), create_graph=True)
        second = []
        for i in range(d):
            row, = torch.autograd.grad(first[:, i].sum(), y, create_graph=create_graph, retain_graph=True)
            second += [row[:, j] for j in range(i, d)]
    return torch.cat((call[:, None], first, torch.stack(second, dim=1), parameters), dim=1)


def reference_labels(stocks, features, nodes=32, chunk=None):
    chunk = chunk or default_chunk(stocks, nodes)
    return derivative_labels(stocks, call_function(stocks, nodes), features, chunk)


def symmetric(stocks, upper, diagonal=True):
    """(N, d, d) symmetric matrices from upper-triangle columns (with or without the diagonal)."""
    d = stocks.d
    index = [(i, j) for i in range(d) for j in range(i if diagonal else i+1, d)]
    out = upper.new_zeros(len(upper), d, d)
    for k, (i, j) in enumerate(index):
        out[:, i, j] = out[:, j, i] = upper[:, k]
    return out, index


def permute_labels(stocks, values, order):
    """Labels of ``coordinates.permute``d features: new stock k is old order[k]."""
    d = stocks.d
    rows = order[:, :, None].expand(-1, -1, d)
    columns = order[:, None, :].expand(-1, d, -1)

    def matrix(upper, diagonal):
        full, index = symmetric(stocks, upper, diagonal)
        full = torch.gather(torch.gather(full, 1, rows), 2, columns)
        index = torch.tensor(index)
        return full[:, index[:, 0], index[:, 1]]

    first, second = values[:, 1:1+d], values[:, 1+d:1+d+d*(d+1)//2]
    rest = values[:, 1+d+d*(d+1)//2:]
    tau, sigma, rho = rest[:, :1], rest[:, 1:1+d], rest[:, 1+d:]
    return torch.cat((values[:, :1], torch.gather(first, 1, order), matrix(second, True), tau,
                      torch.gather(sigma, 1, order), matrix(rho, False)), dim=1)
