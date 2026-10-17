"""Conditional QE simulation of the Heston basket, and its RQMC and Monte Carlo estimates.

Each path simulates the d variance processes only, by Andersen's (2008) quadratic-exponential
(QE) scheme driven by the correlated variance normals (central weights 1/2, psi_c = 1.5,
martingale correction). Given the variance paths, the log returns X_i = log(S_i(T) / F_i)
are jointly Gaussian:

    X_i | Z ~ N(A_i, Sigma_ij),  A_i = sum over steps of K0*_i + K1_i v_i(t) + K2_i v_i(t + dt),
    Sigma_ij = rbar_i rbar_j P_ij int sqrt(v_i v_j) dt  (trapezoid),

with Andersen's constants K, whose martingale correction makes E[exp(X_i)] = 1 exactly for
the discrete scheme. The basket option given the variance paths is then the Gaussian-basket
value of ``src/pricing``'s conditional estimator: one factor of Sigma in closed form, the
other d - 1 residual factors sampled with the path. The payoff kink is integrated out, so
spot Deltas and Gammas are analytic per path, and common-random-number bumps of the variance
parameters give smooth-enough sensitivities.

Sampling: replicated scrambled Sobol nets (sampler "sobol"), the first d - 1 coordinates
for the residual factors and then a Brownian bridge per stock with the stocks interleaved,
or pseudo-random normals (sampler "mc"). Intervals use independent replicate means.

Everything here also works on a batch of contracts in the reduced coordinates
(``normalized_paths``), which is how training labels are made.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc, t as student

from .model import parameter_bumps, quantity_names, spot_names

PSI_C = 1.5
JITTER = 1e-14
PATH_FIELDS = ("v0", "kappa", "theta", "xi", "leverage")


# ---------------------------------------------------------------- normals
def brownian_bridge(z):
    """Unit-variance increments on 0, 1, ..., n from normals z (..., n) in bridge order."""
    n = z.shape[-1]
    W = np.zeros(z.shape[:-1] + (n + 1,))
    W[..., n] = np.sqrt(n) * z[..., 0]
    k = 1
    stack = [(0, n)]
    while stack:
        following = []
        for lo, hi in stack:
            if hi - lo < 2:
                continue
            mid = (lo + hi) // 2
            a, b = mid - lo, hi - mid
            W[..., mid] = (b * W[..., lo] + a * W[..., hi]) / (hi - lo) + np.sqrt(a * b / (hi - lo)) * z[..., k]
            k += 1
            following += [(lo, mid), (mid, hi)]
        stack = following
    return np.diff(W, axis=-1)


def draw_normals(n_points, d, steps, seed, sampler="sobol"):
    """(residual (n, d-1), increments (n, d, steps)) of unit-variance normals."""
    dims = d - 1 + d * steps
    if sampler == "mc":
        z = np.random.default_rng(seed).standard_normal((n_points, dims))
        return z[:, :d - 1], z[:, d - 1:].reshape(n_points, steps, d).transpose(0, 2, 1)
    if sampler != "sobol":
        raise ValueError("sampler must be sobol or mc")
    if n_points & (n_points - 1):
        raise ValueError("Sobol points per replicate must be a power of two")
    u = qmc.Sobol(d=dims, scramble=True, seed=seed).random_base2(int(n_points).bit_length() - 1)
    z = ndtri(np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))
    bridge = z[:, d - 1:].reshape(n_points, steps, d).transpose(0, 2, 1)   # bridge order, stocks interleaved
    return z[:, :d - 1], brownian_bridge(bridge)


# ---------------------------------------------------------------- QE variance paths
def correlate(C, increments):
    """Variance normals z = L_C e per time step; increments (..., n, d, steps), C (..., d, d)."""
    L = np.linalg.cholesky(np.asarray(C, dtype=float))[..., None, :, :]
    return np.einsum("...ij,...jk->...ik", L, increments)


def variance_paths(v0, kappa, theta, xi, leverage, dt, z, martingale=True):
    """QE paths of the stocks in z: (drift (..., n, d), sqrt(v) (..., n, d, steps + 1)).

    Parameters are (..., d) arrays (dt (...)) with a leading contract batch ``...``
    (possibly empty) and z (..., n, d, steps) the correlated variance normals.
    """
    expand = lambda a: np.asarray(a, dtype=float)[..., None, :]          # (..., 1, d)
    v0, kappa, theta, xi, rho = map(expand, (v0, kappa, theta, xi, leverage))
    dt = np.asarray(dt, dtype=float)[..., None, None]                      # (..., 1, 1)
    steps = z.shape[-1]
    e = np.exp(-kappa * dt)
    k0 = -rho * kappa * theta * dt / xi
    k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = 0.5 * dt * (1 - rho ** 2)
    A_coef = k2 + 0.5 * k3
    c1 = xi ** 2 * e * (1 - e) / kappa
    c2 = theta * xi ** 2 * (1 - e) ** 2 / (2 * kappa)
    v = np.broadcast_to(v0, z.shape[:-1]).copy()
    root = np.empty(z.shape[:-1] + (steps + 1,))
    root[..., 0] = np.sqrt(v)
    drift = np.zeros_like(v)
    for k in range(steps):
        m = theta + (v - theta) * e
        psi = (v * c1 + c2) / (m * m)
        quad = psi <= PSI_C
        inv = 2 / np.minimum(psi, PSI_C)
        b2 = inv - 1 + np.sqrt(inv * (inv - 1))
        a = m / (1 + b2)
        zk = z[..., k]
        psi_e = np.maximum(psi, 1.0)
        p = (psi_e - 1) / (psi_e + 1)
        beta = (1 - p) / m
        # u = Phi(z) and 1 - u = Phi(-z), so the exponential branch keeps its tail accurate.
        tail = ndtr(-zk)
        v_exp = np.where(1 - tail <= p, 0.0, np.log((1 - p) / np.maximum(tail, 1e-300)) / beta)
        v_new = np.where(quad, a * (np.sqrt(b2) + zk) ** 2, v_exp)
        if martingale:
            fine_q = A_coef < 1 / (2 * a)
            fine_e = A_coef < beta
            ok = np.where(quad, fine_q, fine_e)
            safe = np.where(fine_q, 1 - 2 * A_coef * a, 1.0)
            log_m = np.where(quad, A_coef * b2 * a / safe - 0.5 * np.log(safe),
                             np.log(p + beta * (1 - p) / np.where(fine_e, beta - A_coef, 1.0)))
            drift += np.where(ok, -log_m - (k1 + 0.5 * k3) * v, k0) + k1 * v + k2 * v_new
        else:
            drift += k0 + k1 * v + k2 * v_new
        v = v_new
        root[..., k + 1] = np.sqrt(v)
    return drift, root


def moments(drift, root, leverage, P, dt):
    """A = drift and Sigma_ij = rbar_i rbar_j P_ij int sqrt(v_i v_j) dt (trapezoid)."""
    steps = root.shape[-1] - 1
    weights = np.full(steps + 1, 1.0)
    weights[[0, -1]] = 0.5
    dt = np.asarray(dt, dtype=float)[..., None, None, None]
    K = np.einsum("...it,...jt->...ij", root, root * weights) * dt
    rbar = np.sqrt(1 - np.asarray(leverage, dtype=float) ** 2)
    scale = (rbar[..., :, None] * rbar[..., None, :] * np.asarray(P, dtype=float))[..., None, :, :]
    return drift, scale * K


def conditional_moments(v0, kappa, theta, xi, leverage, C, P, tau, increments, martingale=True):
    """A (..., n, d) and Sigma (..., n, d, d) of the log returns given the variance paths.

    Parameters are (..., d) arrays (C, P (..., d, d), tau (...)) with a leading contract
    batch ``...`` (possibly empty); ``increments`` are (..., n, d, steps) unit normals,
    correlated across stocks here by the Cholesky factor of C.
    """
    dt = np.asarray(tau, dtype=float) / increments.shape[-1]
    drift, root = variance_paths(v0, kappa, theta, xi, leverage, dt, correlate(C, increments), martingale)
    return moments(drift, root, leverage, P, dt)


# ---------------------------------------------------------------- conditional basket
def _root(log_c, b):
    """Solve sum_i exp(log_c_i + b_i y) = 1 along the last axis (b > 0); Newton from the right.

    The function is convex and increasing in y and the smallest single-term crossing lies
    to the right of the root, so the iterates fall monotonically onto it.
    """
    with np.errstate(invalid="ignore"):
        y = np.min(-log_c / b, axis=-1)
    for _ in range(100):
        e = log_c + b * y[..., None]
        top = e.max(axis=-1, keepdims=True)
        scaled = np.exp(e - top)
        total = scaled.sum(axis=-1)
        slope = (scaled * b).sum(axis=-1) / total
        step = (top[..., 0] + np.log(total)) / slope
        y = y - step
        if np.all(np.abs(step) <= 1e-12 * (1.0 + np.abs(y))):
            return y
    floor = 64 * np.finfo(float).eps * (1.0 + np.abs(top[..., 0])) / slope
    if np.all(np.abs(step) <= np.maximum(1e-12 * (1.0 + np.abs(y)), floor)):
        return y
    raise RuntimeError("Conditional strike root did not converge")


def gaussian_basket(log_f, A, Sigma, U, sign, derivatives=False):
    """E[(sign (sum_i f_i e^{X_i} - 1))^+ | U] for X ~ N(A, Sigma), at residual draws U.

    One factor of Sigma along t = sqrt(diag Sigma) is integrated in closed form, as in
    src/pricing/conditional.py; U (..., n, d-1) are the residual factors. Returns the value
    (..., n), and with ``derivatives`` also d/df_i (..., n, d) and d2/df_i df_j (..., n, d, d).
    """
    d = A.shape[-1]
    Sigma = Sigma + JITTER * np.eye(d)
    t = np.sqrt(np.diagonal(Sigma, axis1=-2, axis2=-1))
    lower = np.linalg.cholesky(Sigma)
    u = np.linalg.solve(lower, t[..., None])[..., 0]
    norm = np.linalg.norm(u, axis=-1, keepdims=True)
    h = u / norm
    h[..., 0] += 1.0
    reflection = np.eye(d) - 2.0 * h[..., :, None] * h[..., None, :] / (h * h).sum(-1)[..., None, None]
    b = t / norm
    shift = A + np.einsum("...ij,...jk,...k->...i", lower, reflection[..., :, 1:], U)
    log_c = np.asarray(log_f)[..., None, :] + shift
    y = _root(log_c, b)
    tails = np.exp(log_c + 0.5 * b ** 2) * ndtr(sign * (b - y[..., None]))
    value = sign * (tails.sum(-1) - ndtr(-sign * y))
    if not derivatives:
        return value
    first = sign * np.exp(shift + 0.5 * b ** 2) * ndtr(sign * (b - y[..., None]))
    g = np.exp(shift + b * y[..., None])
    slope = (np.exp(log_c + b * y[..., None]) * b).sum(-1)
    density = np.exp(-0.5 * y ** 2) / np.sqrt(2 * np.pi)
    second = (density / slope)[..., None, None] * g[..., :, None] * g[..., None, :]
    return value, first, second


def normalized_paths(log_f, v0, kappa, theta, xi, leverage, C, P, tau, U, increments, sign=1,
                     derivatives=False):
    """Per-path normalized values (and f-derivatives) for a batch of contracts."""
    A, Sigma = conditional_moments(v0, kappa, theta, xi, leverage, C, P, tau, increments)
    return gaussian_basket(log_f, A, Sigma, U, sign, derivatives)


# ---------------------------------------------------------------- one contract
def steps_for(tau, dt):
    return max(2, int(np.ceil(tau / dt - 1e-12)))


class _Paths:
    """One contract's variance paths on fixed normals; a bump re-simulates only what it moves."""

    def __init__(self, c, U, increments):
        self.U, self.increments = U, increments
        self.dt = c.tau / increments.shape[-1]
        self.z = correlate(c.vol_correlation_matrix(), increments)
        self.drift, self.root = variance_paths(c.v0, c.kappa, c.theta, c.xi, c.leverage, self.dt, self.z)

    def value(self, c, field=None, index=None, derivatives=False):
        """Normalized values for ``c``, which differs from the base contract in ``field`` only.

        A stock's own variance parameter re-simulates that stock; the stock correlation
        changes P only; the variance correlation and the maturity re-simulate every stock.
        """
        drift, root, dt = self.drift, self.root, self.dt
        if field in PATH_FIELDS:
            own = [np.asarray(getattr(c, name))[[index]] for name in PATH_FIELDS]
            d_i, r_i = variance_paths(*own, dt, self.z[:, [index]])
            drift, root = drift.copy(), root.copy()
            drift[:, index], root[:, index] = d_i[:, 0], r_i[:, 0]
        elif field in ("vol_correlation", "tau"):
            dt = c.tau / self.increments.shape[-1]
            drift, root = variance_paths(c.v0, c.kappa, c.theta, c.xi, c.leverage, dt,
                                         correlate(c.vol_correlation_matrix(), self.increments))
        A, Sigma = moments(drift, root, c.leverage, c.residual_correlation_matrix(), dt)
        with np.errstate(divide="ignore"):
            log_f = np.log(c.forwards())
        return gaussian_basket(log_f, A, Sigma, self.U, c.sign, derivatives)


def path_values(c, U, increments, greeks=True, bump_scale=1.0):
    """Per-path estimators of every quantity in ``quantity_names`` order (or the price).

    Spot Greeks are analytic per path; theta and the parameter sensitivities are central
    differences on the same normals; rho is -tau V + tau sum_i S_i dV/dS_i.
    """
    d = c.dimension
    K = c.discounted_strike()
    paths = _Paths(c, U, increments)
    if not greeks:
        return (K * paths.value(c))[:, None]
    value, first, second = paths.value(c, derivatives=True)
    f = c.forwards()
    spots = np.asarray(c.spots)
    rows = np.triu_indices(d)
    price = K * value
    delta = K * first * f / spots
    gamma = K * second[:, rows[0], rows[1]] * f[rows[0]] * f[rows[1]] / (spots[rows[0]] * spots[rows[1]])
    named = {}
    for name, field, index, step in parameter_bumps(c, bump_scale):
        up, down = c.bumped(field, step, index), c.bumped(field, -step, index)
        upper, lower = paths.value(up, field, index), paths.value(down, field, index)
        if field == "tau":
            # Full value at tau +- step (discount, forwards and the paths), the steps held fixed.
            named[name] = -(up.discounted_strike() * upper - down.discounted_strike() * lower) / (2 * step)
        else:
            named[name] = K * (upper - lower) / (2 * step)
    named["rho"] = c.tau * (-price + (delta * spots).sum(1))
    columns = [price[:, None], delta, gamma] + [named[q][:, None] for q in quantity_names(d)[len(spot_names(d)):]]
    return np.hstack(columns)


@dataclass(frozen=True)
class HestonEstimate:
    """Intervals use independent replicate means and exclude bump and time-step bias."""

    names: tuple
    values: dict
    standard_errors: dict
    intervals: dict
    replicate_means: np.ndarray
    points: int
    replicates: int
    steps: int
    sampler: str


def heston_estimate(c, n_points=4096, n_replicates=16, seed=42, dt=1 / 64, greeks=True, sampler="sobol",
                    bump_scale=1.0):
    """Price, and optionally every sensitivity, of a Heston basket option.

    ``dt`` sets the number of QE steps, ceil(tau / dt) and at least 2.
    """
    if n_replicates < 2:
        raise ValueError("need at least two replicates")
    d = c.dimension
    steps = steps_for(c.tau, dt)
    names = tuple(quantity_names(d)) if greeks else ("price",)
    replicates = []
    for child in np.random.SeedSequence(seed).spawn(n_replicates):
        U, increments = draw_normals(n_points, d, steps, int(child.generate_state(1)[0]), sampler)
        replicates.append(path_values(c, U, increments, greeks, bump_scale).mean(axis=0))
    replicates = np.array(replicates)
    mean = replicates.mean(axis=0)
    se = replicates.std(axis=0, ddof=1) / np.sqrt(n_replicates)
    radius = float(student.ppf(0.975, n_replicates - 1)) * se
    return HestonEstimate(names, dict(zip(names, map(float, mean))), dict(zip(names, map(float, se))),
                          {q: (float(m - r), float(m + r)) for q, m, r in zip(names, mean, radius)},
                          replicates, n_points, n_replicates, steps, sampler)
