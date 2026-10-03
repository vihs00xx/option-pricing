"""Conditional (one-factor smoothed) basket estimators for d correlated stocks.

The log returns X ~ N(drift, C) are written as X = drift + b Y + A U, with
Y ~ N(0, 1) independent of U ~ N(0, I_{d-1}), b = s t for a positive direction t,
and s^2 = 1 / (t' C^{-1} t), the largest multiple of t t' that can be removed
from C while leaving a valid covariance. Every stock increases with Y, so given
U the basket crosses the strike at one root y* and the conditional option value
is a sum of Black-type terms. Only U is sampled or integrated numerically.

With t = 1 (equal loadings) the root is explicit and, for two stocks, the
construction is the shared-factor estimator of the two-asset study. The default
t = sigma keeps the smoothing factor large when volatilities differ strongly;
engine_validation_v1 records the comparison. The residual
coordinates use a Cholesky factor and a Householder reflection, both smooth in
the parameters, so common-random-number bumps stay smooth.
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr

from .contract import parameter_bumps, spot_names

DIRECTIONS = ("equal", "volatility")
KINDS = ("arithmetic", "geometric")


def factorization(c, direction="volatility"):
    """Loadings b (d,) on the smoothing factor and A (d, d-1) on the residual."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if c.tau <= 0:
        raise ValueError("Conditional estimators require positive maturity")
    d = c.dimension
    lower = np.linalg.cholesky(c.covariance())
    t = np.ones(d) if direction == "equal" else np.asarray(c.volatilities)
    u = solve_triangular(lower, t, lower=True)
    norm = np.linalg.norm(u)
    v = u / norm
    # v[0] = t[0] / (lower[0, 0] * norm) > 0, so h never vanishes.
    h = v.copy()
    h[0] += 1.0
    reflection = np.eye(d) - 2.0 * np.outer(h, h) / (h @ h)
    return t / norm, lower @ reflection[:, 1:]


def _root(log_c, b, log_strike):
    """Solve sum_i exp(log_c_i + b_i y) = K for each row; Newton from the right.

    f(y) = logsumexp(log_c + b y) - log K is convex and increasing, and the
    smallest single-stock crossing satisfies f >= 0, so iterates fall
    monotonically onto the root.
    """
    with np.errstate(invalid="ignore"):
        y = np.min((log_strike - log_c) / b, axis=1)
    for _ in range(100):
        a = log_c + b * y[:, None]
        top = a.max(axis=1, keepdims=True)
        scaled = np.exp(a - top)
        total = scaled.sum(axis=1)
        step = (top[:, 0] + np.log(total) - log_strike) / ((scaled @ b) / total)
        y = y - step
        if np.all(np.abs(step) <= 1e-13 * (1.0 + np.abs(y))):
            return y
    raise RuntimeError("Conditional strike root did not converge")


def _spot_values(c, U, kind, direction, greeks):
    b, A = factorization(c, direction)
    d = c.dimension
    w, spots = np.asarray(c.weights), np.asarray(c.spots)
    shifts = c.log_drift() + U @ A.T
    discount = np.exp(-c.rate * c.tau)
    sign, strike = c.sign, c.strike
    rows = np.triu_indices(d)
    if kind == "arithmetic":
        with np.errstate(divide="ignore"):
            log_scaled = np.log(w) + shifts                    # log(c_i / S_i)
        log_c = log_scaled + np.log(spots)
        y = _root(log_c, b, np.log(strike))
        tails = np.exp(log_c + 0.5 * b**2) * ndtr(sign * (b - y[:, None]))
        price = discount * sign * (tails.sum(axis=1) - strike * ndtr(-sign * y))
        if not greeks:
            return price
        delta = discount * sign * np.exp(log_scaled + 0.5 * b**2) * ndtr(sign * (b - y[:, None]))
        g = np.exp(log_scaled + b * y[:, None])
        slope = np.exp(log_c + b * y[:, None]) @ b
        density = np.exp(-0.5 * y**2) / np.sqrt(2 * np.pi)
        gamma = (discount * density / slope)[:, None] * g[:, rows[0]] * g[:, rows[1]]
        return np.column_stack((price, delta, gamma))
    if kind != "geometric":
        raise ValueError(f"kind must be one of {KINDS}")
    active = w > 0
    log_mean = (np.log(spots[active]) + shifts[:, active]) @ w[active]
    sd = float(w @ b)
    d1 = (log_mean - np.log(strike) + sd**2) / sd
    mean = np.exp(log_mean + 0.5 * sd**2)
    price = discount * sign * (mean * ndtr(sign * d1) - strike * ndtr(sign * (d1 - sd)))
    if not greeks:
        return price
    first = sign * discount * mean * ndtr(sign * d1)
    second = first + discount * mean * np.exp(-0.5 * d1**2) / (np.sqrt(2 * np.pi) * sd)
    delta = first[:, None] * w / spots
    diagonal = (rows[0] == rows[1]).astype(float)
    gamma = ((second[:, None] * w[rows[0]] * w[rows[1]] - first[:, None] * w[rows[0]] * diagonal)
             / (spots[rows[0]] * spots[rows[1]]))
    return np.column_stack((price, delta, gamma))


def sample_values(c, U, kind="arithmetic", direction="volatility", greeks=True, bump_scale=1.0):
    """Conditional values for each residual draw: price, or every quantity in
    contract.quantity_names order when greeks=True.

    Bumped sensitivities re-evaluate each bumped contract at the same U, so
    their sampling error is shared with the base price; bump bias is excluded.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[1] != c.dimension - 1 or not np.all(np.isfinite(U)):
        raise ValueError(f"U must have shape (n, {c.dimension - 1}) and be finite")
    if not greeks:
        return _spot_values(c, U, kind, direction, False)[:, None]
    columns = [_spot_values(c, U, kind, direction, True)]
    for _, field, index, step, sign in parameter_bumps(c, bump_scale):
        upper = _spot_values(c.bumped(field, step, index), U, kind, direction, False)
        lower = _spot_values(c.bumped(field, -step, index), U, kind, direction, False)
        columns.append((sign * (upper - lower) / (2 * step))[:, None])
    return np.column_stack(columns)


def gauss_hermite_values(c, n_nodes=64, kind="arithmetic", direction="volatility", greeks=True, bump_scale=1.0):
    """Tensor Gauss-Hermite integration of the conditional values over U.

    The conditional values are smooth, so this converges quickly in low
    residual dimension; cost grows as n_nodes ** (d - 1).
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or not 1 <= n_nodes <= 320:
        raise ValueError("n_nodes must be an integer from 1 to 320; NumPy's Hermite weights overflow above about 360")
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    weights = weights / np.sqrt(2 * np.pi)
    k = c.dimension - 1
    if k == 0:
        U, mass = np.zeros((1, 0)), np.ones(1)
    else:
        grids = np.meshgrid(*([nodes] * k), indexing="ij")
        U = np.column_stack([g.ravel() for g in grids])
        mass = np.prod(np.meshgrid(*([weights] * k), indexing="ij"), axis=0).ravel()
    return mass @ sample_values(c, U, kind, direction, greeks, bump_scale)


def value_names(c, greeks=True):
    if not greeks:
        return ["price"]
    return spot_names(c.dimension) + [bump[0] for bump in parameter_bumps(c)]
