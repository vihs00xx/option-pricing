"""Andersen (2008) quadratic-exponential scheme with the martingale correction.

The simulated quantity is log(S_t / F_t), a martingale under Q, so payoffs are
(e^{x + X} - 1)^+ per unit strike. Central weights gamma1 = gamma2 = 1/2,
switching value psi_c = 1.5. `qe_paths` takes its normals and uniforms from a
caller-supplied source so the same code serves pseudo-random and RQMC runs.
"""

import numpy as np
from scipy.special import ndtr, ndtri

from .model import apply_parity

PSI_C = 1.5


def qe_step_constants(kappa, theta, xi, rho, dt):
    e = np.exp(-kappa * dt)
    k0 = -rho * kappa * theta * dt / xi
    k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = 0.5 * dt * (1 - rho**2)
    k4 = k3
    return e, k0, k1, k2, k3, k4


def qe_step(v, logx, z_v, u_v, z_x, kappa, theta, xi, rho, dt, consts, martingale=True):
    """Advance (v, log X) by dt. Returns new arrays and the count of correction failures.

    Both branches are evaluated on every path and selected with `where` (no masking).
    """
    e, k0, k1, k2, k3, k4 = consts
    m = theta + (v - theta) * e
    s2 = v * (xi**2 * e * (1 - e) / kappa) + theta * xi**2 * (1 - e) ** 2 / (2 * kappa)
    psi = s2 / (m * m)
    quad = psi <= PSI_C
    A = k2 + 0.5 * k4
    # quadratic branch (psi clipped so the unused lanes stay finite)
    inv = 2 / np.minimum(psi, PSI_C)
    b2 = inv - 1 + np.sqrt(inv * (inv - 1))
    a = m / (1 + b2)
    v_quad = a * (np.sqrt(b2) + z_v) ** 2
    # exponential branch
    psi_e = np.maximum(psi, 1.0)
    p = (psi_e - 1) / (psi_e + 1)
    beta = (1 - p) / m
    v_exp = np.where(u_v <= p, 0.0, np.log((1 - p) / np.maximum(1 - u_v, 1e-300)) / beta)
    v_new = np.where(quad, v_quad, v_exp)
    failures = 0
    if martingale:
        ok_q = A < 1 / (2 * a)
        ok_e = A < beta
        ok = np.where(quad, ok_q, ok_e)
        failures = int(v.size - np.count_nonzero(ok))
        safe = np.where(ok_q, 1 - 2 * A * a, 1.0)
        lnM_q = A * b2 * a / safe - 0.5 * np.log(safe)
        lnM_e = np.log(p + beta * (1 - p) / np.where(ok_e, beta - A, 1.0))
        lnM = np.where(quad, lnM_q, lnM_e)
        k0_star = np.where(ok, -lnM - (k1 + 0.5 * k3) * v, k0)
    else:
        k0_star = k0
    logx = logx + k0_star + k1 * v + k2 * v_new + np.sqrt(k3 * v + k4 * v_new) * z_x
    return v_new, logx, failures


def qe_paths(v0, kappa, theta, xi, rho, tau, steps, draws, martingale=True):
    """Terminal log X for paths driven by draws(step) -> (z_v, u_v, z_x) arrays."""
    dt = tau / steps
    consts = qe_step_constants(kappa, theta, xi, rho, dt)
    v = None
    logx = None
    failures = 0
    for k in range(steps):
        z_v, u_v, z_x = draws(k)
        if v is None:
            v = np.full(z_v.shape, float(v0))
            logx = np.zeros(z_v.shape)
        v, logx, f = qe_step(v, logx, z_v, u_v, z_x, kappa, theta, xi, rho, dt, consts, martingale)
        failures += f
    return logx, v, failures


def qe_price(c1, n_paths, steps, rng, chunk=1 << 18, martingale=True):
    """Plain Monte Carlo price for one contract: (mean, standard error, correction failures).

    The variance uniform u_v also drives the quadratic-branch normal (z_v = Phi^{-1}(u_v)),
    as in Andersen's scheme, so each step uses two random numbers.
    """
    x, disc = float(c1.x[0]), float(c1.discount[0])
    args = [float(getattr(c1, n)[0]) for n in ("v0", "kappa", "theta", "xi", "rho", "tau")]
    total, total2, failures, done = 0.0, 0.0, 0, 0
    while done < n_paths:
        m = min(chunk, n_paths - done)

        def draws(_k, m=m):
            u = rng.random(m)
            return ndtri(u), u, rng.standard_normal(m)

        logx, _, f = qe_paths(*args, steps, draws, martingale)
        payoff = np.maximum(np.exp(x + logx) - 1, 0.0) if c1.call[0] else np.maximum(1 - np.exp(x + logx), 0.0)
        total += payoff.sum()
        total2 += (payoff**2).sum()
        failures += f
        done += m
    mean = total / n_paths
    var = max(total2 / n_paths - mean**2, 0.0) * n_paths / (n_paths - 1)
    return disc * mean, disc * np.sqrt(var / n_paths), failures


# ---- Greeks with common random numbers; pseudo-random and RQMC drivers ------------------

# Relative bump sizes for the CRN finite differences (absolute for rho).
MC_BUMPS = {"S": 0.01, "v0": 0.02, "kappa": 0.02, "theta": 0.02, "xi": 0.02, "rho": 0.02, "tau": 0.02}
_ESTIMATORS = ("price", "delta", "gamma", "vega", "vanna", "volga", "theta_cal", "rho_rate",
               "d_kappa", "d_theta", "d_xi", "d_rho")


def _terminal(pars, tau, Zv, Zx, martingale=True):
    v0, kappa, theta, xi, rho = pars
    steps = Zv.shape[1]
    draws = lambda k: (Zv[:, k], ndtr(Zv[:, k]), Zx[:, k])
    logx, _, fails = qe_paths(v0, kappa, theta, xi, rho, tau, steps, draws, martingale)
    return logx, fails


def path_estimators(c1, Zv, Zx):
    """Per-path call estimators of all 12 quantities (CRN bumps) for normals of shape (n, steps)."""
    S, tau, r, q = (float(getattr(c1, k)[0]) for k in ("S", "tau", "r", "q"))
    base = {k: float(getattr(c1, k)[0]) for k in ("v0", "kappa", "theta", "xi", "rho")}
    names = ("v0", "kappa", "theta", "xi", "rho")

    def sim(**chg):
        p = dict(base, **{k: v for k, v in chg.items() if k != "tau"})
        return _terminal(tuple(p[k] for k in names), chg.get("tau", tau), Zv, Zx)[0]

    def pay(x, L):
        return np.maximum(np.exp(x + L) - 1, 0.0)

    def dpay(x, L):
        return np.where(x + L > 0, np.exp(x + L), 0.0)

    x = np.log(S) + (r - q) * tau
    disc = np.exp(-r * tau)
    L0 = sim()
    est = {"price": disc * pay(x, L0), "delta": disc * dpay(x, L0) / S}
    hS = MC_BUMPS["S"] * S
    dp = disc * dpay(np.log(S + hS) + (r - q) * tau, L0) / (S + hS)
    dm = disc * dpay(np.log(S - hS) + (r - q) * tau, L0) / (S - hS)
    est["gamma"] = (dp - dm) / (2 * hS)
    est["rho_rate"] = tau * (disc * dpay(x, L0) - est["price"])
    hv = MC_BUMPS["v0"] * base["v0"]
    Lp, Lm = sim(v0=base["v0"] + hv), sim(v0=base["v0"] - hv)
    est["vega"] = disc * (pay(x, Lp) - pay(x, Lm)) / (2 * hv)
    est["volga"] = disc * (pay(x, Lp) - 2 * pay(x, L0) + pay(x, Lm)) / hv**2
    est["vanna"] = disc * (dpay(x, Lp) - dpay(x, Lm)) / (S * 2 * hv)
    for p in ("kappa", "theta", "xi", "rho"):
        h = MC_BUMPS[p] * (1.0 if p == "rho" else base[p])
        Lp, Lm = sim(**{p: base[p] + h}), sim(**{p: base[p] - h})
        est[f"d_{p}"] = disc * (pay(x, Lp) - pay(x, Lm)) / (2 * h)
    ht = MC_BUMPS["tau"] * tau
    vals = []
    for t in (tau + ht, tau - ht):
        Lt = sim(tau=t)
        vals.append(np.exp(-r * t) * pay(np.log(S) + (r - q) * t, Lt))
    est["theta_cal"] = -(vals[0] - vals[1]) / (2 * ht)
    return est


def _finish(c1, mean, se):
    out = {k: np.array([mean[k]]) for k in _ESTIMATORS}
    out = apply_parity(c1, out)
    return {k: float(out[k][0]) for k in _ESTIMATORS}, {k: float(se[k]) for k in _ESTIMATORS}


def steps_for(tau, dt):
    return max(2, int(np.ceil(tau / dt - 1e-12)))


def mc_quantities(c1, n_paths, dt, seed, chunk=1 << 16):
    """Pseudo-random QE Monte Carlo: (values, standard errors) of the 12 quantities."""
    steps = steps_for(float(c1.tau[0]), dt)
    rng = np.random.default_rng(seed)
    s1 = {k: 0.0 for k in _ESTIMATORS}
    s2 = {k: 0.0 for k in _ESTIMATORS}
    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        Zv, Zx = rng.standard_normal((m, steps)), rng.standard_normal((m, steps))
        est = path_estimators(c1, Zv, Zx)
        for k in _ESTIMATORS:
            s1[k] += est[k].sum()
            s2[k] += (est[k] ** 2).sum()
        done += m
    mean = {k: s1[k] / n_paths for k in _ESTIMATORS}
    se = {k: np.sqrt(max(s2[k] / n_paths - mean[k] ** 2, 0.0) / (n_paths - 1)) for k in _ESTIMATORS}
    return _finish(c1, mean, se)


def brownian_bridge(z):
    """Unit-variance increments of W on 0, 1, ..., n built from normals z (m, n) in bridge order."""
    m, n = z.shape
    W = np.zeros((m, n + 1))
    W[:, n] = np.sqrt(n) * z[:, 0]
    k = 1
    stack = [(0, n)]
    while stack:
        nxt = []
        for lo, hi in stack:
            if hi - lo < 2:
                continue
            mid = (lo + hi) // 2
            a, b = mid - lo, hi - mid
            W[:, mid] = (b * W[:, lo] + a * W[:, hi]) / (hi - lo) + np.sqrt(a * b / (hi - lo)) * z[:, k]
            k += 1
            nxt += [(lo, mid), (mid, hi)]
        stack = nxt
    return np.diff(W, axis=1)


def rqmc_quantities(c1, log2_points, dt, seed, scrambles=16):
    """Scrambled Sobol + Brownian bridge QE: (values, standard errors over scrambles)."""
    from scipy.stats import qmc
    steps = steps_for(float(c1.tau[0]), dt)
    reps = {k: [] for k in _ESTIMATORS}
    for s in range(scrambles):
        u = qmc.Sobol(d=2 * steps, scramble=True, seed=seed * 1000 + s).random_base2(log2_points)
        z = ndtri(np.clip(u, 1e-16, 1 - 1e-16))
        # interleave: the two Brownian motions take alternating Sobol coordinates
        Zv = brownian_bridge(z[:, 0::2])
        Zx = brownian_bridge(z[:, 1::2])
        est = path_estimators(c1, Zv, Zx)
        for k in _ESTIMATORS:
            reps[k].append(est[k].mean())
    mean = {k: float(np.mean(v)) for k, v in reps.items()}
    se = {k: float(np.std(v, ddof=1) / np.sqrt(scrambles)) for k, v in reps.items()}
    return _finish(c1, mean, se)
