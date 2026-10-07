"""NumPy reference: Lewis single integral with closed-form integrands for all 12 quantities.

With z = 1/2 + i u, u^2 + 1/4 = z (1 - z) and G(u) = exp(z x) phi(u - i/2),

    c        = e^x - (1/pi) int_0^inf Re[G / (z (1 - z))] du
    c_x      = e^x - (1/pi) int Re[G / (1 - z)] du
    c_xx-c_x =       (1/pi) int Re[G] du
    d^n c / dv0^n (x-derivatives) multiply the integrand by D^n (phi = exp(C + D v0))
    dc/dp    = -(1/pi) int Re[G (C_p + D_p v0) / (z (1 - z))] du,  p in (kappa, theta, xi, rho)

where C_p, D_p come from forward-mode differentiation of the little-trap formulas.
The integrals run over [0, U]; U is the first point of a 2^{k/4} grid beyond which
every integrand magnitude times u stays below the truncation tolerance. The
quadrature is adaptive 16-point Gauss-Legendre: a panel is accepted when its
two halves agree with the whole panel to eps * width / U on every component, and
the halves' sum is kept. Differences at the rounding level of the panel's integral of |f| are accepted.
"""

import numpy as np
from scipy import integrate

from .characteristic import Dual, cf, cf_parts
from .model import Contracts, PARAMETERS, QUANTITIES, apply_parity, theta_from_pde

N_FULL = 10
# (panel tolerance eps, truncation tolerance, initial panel width) per convergence level
LEVELS = ((1e-10, 1e-13, 32.0), (1e-12, 1e-15, 16.0), (1e-14, 1e-17, 8.0), (1e-15, 1e-19, 4.0))
AGREEMENT = 1e-9
# The phase u x + Im(C + D v0) grows linearly in u, so the integrand's rounding
# error is about eps_mach * u relative; differences below ROUNDOFF (4 + u) int |f|
# over a panel are rounding, not quadrature error.
ROUNDOFF = 1e-15
MAX_PANELS = 200_000
_GL_T, _GL_W = np.polynomial.legendre.leggauss(16)
_SCAN = 2.0 ** (np.arange(-8, 101) / 4)


def _column(a):
    return np.asarray(a, dtype=float).reshape(-1, 1)


def lewis_integrands(u, c, full=True):
    """Complex integrands (n_comp, n_contracts, n_u); take .real for the integral.

    u has shape (n_u,) or (n_contracts, n_u).
    """
    u = np.atleast_2d(u)
    x, tau, v0 = _column(c.x), _column(c.tau), _column(c.v0)
    z = 0.5 + 1j * u
    uu = u - 0.5j
    w0 = 1 / (z * (1 - z))
    if not full:
        C, D = cf_parts(uu, tau, _column(c.kappa), _column(c.theta), _column(c.xi), _column(c.rho))
        G = np.exp(z * x + C + D * v0)
        return (G * w0)[None]
    params = Dual.variables([_column(getattr(c, p)) for p in PARAMETERS])
    C, D = cf_parts(uu, tau, *params)
    G = np.exp(z * x + C.v + D.v * v0)
    Gw = G * w0
    out = [Gw, G / (1 - z), G, Gw * D.v, G * D.v / (1 - z), Gw * D.v * D.v]
    for k in range(len(PARAMETERS)):
        out.append(Gw * (C.g[k] + D.g[k] * v0))
    return np.stack(out)


def truncation_point(c, tol, full=True, chunk=256):
    """Per-contract U: beyond U every |integrand| * u is below tol on the 2^{k/4} grid."""
    U = np.empty(len(c))
    for start in range(0, len(c), chunk):
        part = c.take(slice(start, start + chunk))
        mag = np.abs(lewis_integrands(_SCAN, part, full)).max(axis=0) * _SCAN
        above = mag > tol
        last = np.where(above.any(axis=1), above.shape[1] - 1 - np.argmax(above[:, ::-1], axis=1), 0)
        U[start:start + chunk] = _SCAN[np.minimum(last + 1, _SCAN.size - 1)]
    return np.maximum(U, 2.0)


def _panel_rule(f, lo, hi):
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    u = mid[:, None] + half[:, None] * _GL_T
    vals = f(u.ravel()).reshape(-1, lo.size, _GL_T.size)
    return (vals * _GL_W).sum(-1) * half, (np.abs(vals) * _GL_W).sum(-1) * half


def initial_breakpoints(U, width=32.0):
    geometric = 0.25 * 2.0 ** np.arange(0, 64)
    pts = np.concatenate([[0.0], geometric[geometric < U], np.arange(width, U, width), [U]])
    return np.unique(pts)


def adaptive_integrals(f, U, eps, width=32.0, max_iter=60, min_width=1e-10):
    """Integrate the real vector function f over [0, U]. Returns (totals, nodes, weights, info)."""
    pts = initial_breakpoints(U, width)
    lo, hi = pts[:-1], pts[1:]
    whole, _ = _panel_rule(f, lo, hi)
    total = np.zeros(whole.shape[0])
    kept_lo, kept_hi = [], []
    n_eval = lo.size * _GL_T.size
    err_total = 0.0
    converged = True
    for it in range(max_iter):
        mid = 0.5 * (lo + hi)
        halves, size = _panel_rule(f, np.concatenate([lo, mid]), np.concatenate([mid, hi]))
        n_eval += 2 * lo.size * _GL_T.size
        left, right = halves[:, :lo.size], halves[:, lo.size:]
        err = np.abs(left + right - whole)
        floor = ROUNDOFF * (4 + hi) * (size[:, :lo.size] + size[:, lo.size:])
        ok = (err <= np.maximum(eps * (hi - lo) / U, floor)).all(axis=0) | (hi - lo < min_width)
        err = err.max(axis=0)
        if it == max_iter - 1 or 2 * (~ok).sum() > MAX_PANELS:
            converged = bool(ok.all())
            ok[:] = True
        total += (left + right)[:, ok].sum(axis=1)
        err_total += err[ok].sum()
        kept_lo += [lo[ok], mid[ok]]
        kept_hi += [mid[ok], hi[ok]]
        bad = ~ok
        if not bad.any() or not converged:
            break
        lo, hi = np.concatenate([lo[bad], mid[bad]]), np.concatenate([mid[bad], hi[bad]])
        whole = np.concatenate([left[:, bad], right[:, bad]], axis=1)
    klo, khi = np.concatenate(kept_lo), np.concatenate(kept_hi)
    order = np.argsort(klo)
    klo, khi = klo[order], khi[order]
    mid, half = 0.5 * (klo + khi), 0.5 * (khi - klo)
    nodes = (mid[:, None] + half[:, None] * _GL_T).ravel()
    weights = (half[:, None] * _GL_W).ravel()
    info = {"panels": int(klo.size), "evaluations": int(n_eval), "converged": converged,
            "error_estimate": float(err_total), "iterations": it + 1}
    return total, nodes, weights, info


def quantities_from_integrals(c, I):
    """Map the 10 Lewis integrals (shape (10, n)) to the 12 quantities, with parity for puts."""
    disc, ex, S = c.discount, np.exp(c.x), c.S
    cx = ex - I[1] / np.pi
    out = {
        "price": disc * (ex - I[0] / np.pi),
        "delta": disc * cx / S,
        "gamma": disc * I[2] / (np.pi * S**2),
        "vega": -disc * I[3] / np.pi,
        "vanna": -disc * I[4] / (np.pi * S),
        "volga": -disc * I[5] / np.pi,
    }
    out["rho_rate"] = c.tau * (disc * cx - out["price"])
    for k, p in enumerate(PARAMETERS):
        out[f"d_{p}"] = -disc * I[6 + k] / np.pi
    out = apply_parity(c, out)
    out["theta_cal"] = theta_from_pde(c, out)
    return {q: out[q] for q in QUANTITIES}


def price_from_integral(c, I0):
    out = {"price": c.discount * (np.exp(c.x) - I0 / np.pi), "delta": np.zeros(len(c)),
           "rho_rate": np.zeros(len(c))}
    return apply_parity(c, out)["price"]


def integrate_contract(c1, level, full=True):
    """Lewis integrals for a single contract (len(c1) == 1) at one convergence level."""
    eps, trunc, width = LEVELS[level]
    U = float(truncation_point(c1, trunc, full)[0])
    f = lambda u: lewis_integrands(u, c1, full)[:, 0, :].real
    total, nodes, weights, info = adaptive_integrals(f, U, eps, width)
    info["U"] = U
    return total, nodes, weights, info


def reference(c, full=True, tol=AGREEMENT, max_level=len(LEVELS) - 1, keep_nodes=False):
    """All 12 quantities (or the price only) with a per-contract convergence ladder.

    Levels l and l+1 are computed until every quantity agrees to `tol`; the
    finer level's values are returned. info holds the level used, the last
    change, U, panel counts and (optionally) the final node set.
    """
    n = len(c)
    names = QUANTITIES if full else ("price",)
    out = {q: np.empty(n) for q in names}
    info = {"level": np.zeros(n, int), "change": np.zeros(n), "converged": np.ones(n, bool),
            "U": np.zeros(n), "panels": np.zeros(n, int), "evaluations": np.zeros(n, int),
            "worst_quantity": np.empty(n, object)}
    nodes = []
    for i in range(n):
        ci = c.take([i])
        prev = None
        for level in range(max_level + 1):
            total, nd, wt, inf = integrate_contract(ci, level, full)
            vals = (quantities_from_integrals(ci, total[:, None]) if full
                    else {"price": price_from_integral(ci, total[:1])})
            info["evaluations"][i] += inf["evaluations"]
            if prev is not None:
                diffs = {q: abs(float(vals[q][0] - prev[q][0])) for q in names}
                worst = max(diffs, key=diffs.get)
                if diffs[worst] <= tol or level == max_level:
                    info["converged"][i] = diffs[worst] <= tol and inf["converged"]
                    info["change"][i] = diffs[worst]
                    info["worst_quantity"][i] = worst
                    break
            prev = vals
        for q in names:
            out[q][i] = vals[q][0]
        info["level"][i], info["U"][i], info["panels"][i] = level, inf["U"], inf["panels"]
        if keep_nodes:
            nodes.append((nd, wt))
    if keep_nodes:
        info["nodes"] = nodes
    return out, info


def price_on_nodes(c, nodes, weights):
    """Price of every contract in c with one fixed quadrature rule (smooth in all inputs)."""
    vals = lewis_integrands(nodes, c, full=False)[0].real
    return price_from_integral(c, vals @ weights)


def richardson(c1, name, h, nodes, weights):
    """Richardson-extrapolated central difference of the price in field `name`, rule fixed."""
    base = getattr(c1, name)[0]
    bumps = base + np.array([h, -h, h / 2, -h / 2])
    batch = Contracts.concat(*[c1] * 4).with_values(**{name: bumps})
    p = price_on_nodes(batch, nodes, weights)
    d1 = (p[0] - p[1]) / (2 * h)
    d2 = (p[2] - p[3]) / h
    return (4 * d2 - d1) / 3


# ---- Independent check: Heston's two-probability form (Gil-Pelaez) -----------------

def gil_pelaez_price(c1, epsabs=1e-14, epsrel=1e-13):
    """c = e^x P1 - P2 with P_j = 1/2 + (1/pi) int Re[e^{iux} phi_j(u) / (i u)] du (scipy quad)."""
    x, tau, v0 = float(c1.x[0]), float(c1.tau[0]), float(c1.v0[0])
    pars = tuple(float(getattr(c1, p)[0]) for p in PARAMETERS)

    def phis(u):
        u = np.atleast_1d(u)
        p2 = cf(u, tau, v0, *pars)
        p1 = cf(u - 1j, tau, v0, *pars)
        return p1, p2

    p1, p2 = phis(_SCAN)
    mag = (np.abs(p1) * np.exp(x) + np.abs(p2))
    above = np.nonzero(mag > 1e-17)[0]
    U = max(_SCAN[min(above[-1] + 1, _SCAN.size - 1)] if above.size else 2.0, 2.0)
    points = initial_breakpoints(U, width=max(U / 64, 32.0))[1:-1][:90]

    def integrand(u, j):
        p1, p2 = phis(u)
        p = p1 if j == 1 else p2
        return float((np.exp(1j * u * x) * p / (1j * u)).real[0])

    probs = []
    for j in (1, 2):
        val, _ = integrate.quad(integrand, 0.0, U, args=(j,), points=points, limit=20000,
                                epsabs=epsabs, epsrel=epsrel)
        probs.append(0.5 + val / np.pi)
    call = np.exp(x) * probs[0] - probs[1]
    out = {"price": np.array([c1.discount[0] * call]), "delta": np.zeros(1), "rho_rate": np.zeros(1)}
    return float(apply_parity(c1, out)["price"][0])
