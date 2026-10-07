"""Hundsdorfer-Verwer ADI for the Heston PDE on nonuniform (S, v) grids (In 't Hout and Foulon 2010).

The v-convection uses second-order backward (upwind) differences where v > 1.

    V_tau = 1/2 v S^2 V_SS + rho xi v S V_Sv + 1/2 xi^2 v V_vv + (r-q) S V_S + kappa (theta - v) V_v - r V

Grids (per unit strike, K = 1):
  S = K + c sinh(xi_i) on [0, S_max], clustered at the strike; the spot S0 is a node.
  v = d sinh(eta_j) on [0, v_max], clustered near v = 0; v0 is a node.
Boundaries:
  S = 0     V = 0 (calls; puts by parity).
  S = S_max V_SS = 0 (linear in S); V_S and V_Sv one-sided.
  v = 0     the degenerate PDE itself: V_tau = (r-q) S V_S + kappa theta V_v - r V with a
            second-order forward difference for V_v; no extra condition.
  v = v_max V_v = 0 (reflected ghost node for V_vv; no mixed term), or optionally (far_v =
            "base_value") the Black-Scholes price at the expected average variance from v_max.
Splitting: A0 = mixed term, A1 = S-direction terms + r V/2, A2 = v-direction terms + r V/2.
Time: `damping` Douglas (theta = 1) half-steps first (Rannacher-type start for the payoff
kink), then HV with theta = 1/2 + sqrt(3)/6. The payoff is cell-averaged at the nodes next
to the strike. Greeks: Delta, Gamma, Vega, Vanna, Volga from the grid at (S0, v0); Theta
from the PDE; rate rho and the parameter sensitivities by central bumps re-solved on the
same grid.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.special import ndtr

from .model import Contracts, apply_parity, theta_from_pde

HV_THETA = 0.5 + np.sqrt(3) / 6
BUMPS = {"kappa": 1e-3, "theta": 1e-4, "xi": 1e-3, "rho": 1e-3, "r": 1e-4}


@dataclass(frozen=True)
class GridSize:
    m1: int        # S intervals
    m2: int        # v intervals
    steps: int     # time steps
    damping: int = 2


def _grid(anchor_value, lower, upper, scale, centre, target_intervals):
    """Nodes x = centre + scale sinh(t) on [lower, upper] with anchor_value exactly on a node."""
    t_lo = np.arcsinh((lower - centre) / scale)
    t_hi = np.arcsinh((upper - centre) / scale)
    t_a = np.arcsinh((anchor_value - centre) / scale)
    target = (t_hi - t_lo) / target_intervals
    k = max(1, int(round((t_a - t_lo) / target)))
    step = (t_a - t_lo) / k
    n = int(np.ceil((t_hi - t_lo) / step - 1e-9))
    t = t_lo + step * np.arange(n + 1)
    x = centre + scale * np.sinh(t)
    x[0], x[-1], x[k] = lower, upper, anchor_value
    if x[-1] - x[-2] < 0.25 * (x[-2] - x[-3]):   # merge a sliver at the far end
        x = np.delete(x, -2)
    return x, k


def stencils(x):
    """Central first/second derivative weights at interior nodes of a nonuniform grid."""
    hm, hp = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    first = np.stack([-hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp))])
    second = np.stack([2 / (hm * (hm + hp)), -2 / (hm * hp), 2 / (hp * (hm + hp))])
    return first, second


def _forward_first(x):
    """Second-order forward difference weights at x[0] on nodes 0, 1, 2."""
    h1, h2 = x[1] - x[0], x[2] - x[1]
    return np.array([-(2 * h1 + h2) / (h1 * (h1 + h2)), (h1 + h2) / (h1 * h2), -h1 / (h2 * (h1 + h2))])


def _backward_first(x):
    """Second-order backward difference weights at x[-1] on nodes -3, -2, -1."""
    h1, h2 = x[-2] - x[-3], x[-1] - x[-2]
    return np.array([h2 / (h1 * (h1 + h2)), -(h1 + h2) / (h1 * h2), (2 * h2 + h1) / (h2 * (h1 + h2))])


def first_derivative_matrix(x):
    """Sparse D1 on all nodes: central inside, one-sided second order at both ends."""
    n = len(x)
    f, _ = stencils(x)
    rows, cols, vals = [], [], []
    for k in range(3):
        rows.append(np.arange(1, n - 1)); cols.append(np.arange(k, n - 2 + k)); vals.append(f[k])
    rows.append(np.zeros(3, int)); cols.append(np.arange(3)); vals.append(_forward_first(x))
    rows.append(np.full(3, n - 1)); cols.append(np.arange(n - 3, n)); vals.append(_backward_first(x))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def upwind_first_matrix(x, above=1.0):
    """D1 with second-order backward differences at interior nodes x_j > above (j >= 2),
    central elsewhere: In 't Hout and Foulon's upwinding of the v-convection kappa (theta - v) V_v,
    whose coefficient is negative there."""
    D = first_derivative_matrix(x).tolil()
    for j in range(2, len(x) - 1):
        if x[j] > above:
            h1, h2 = x[j - 1] - x[j - 2], x[j] - x[j - 1]
            D[j, :] = 0
            D[j, j - 2] = h2 / (h1 * (h1 + h2))
            D[j, j - 1] = -(h1 + h2) / (h1 * h2)
            D[j, j] = (2 * h2 + h1) / (h2 * (h1 + h2))
    return D.tocsr()


def second_derivative_matrix(x, reflect_upper=False):
    """Sparse D2: central inside, zero at the lower node; zero (linear) or reflected (V' = 0) at the top."""
    n = len(x)
    _, s = stencils(x)
    rows, cols, vals = [], [], []
    for k in range(3):
        rows.append(np.arange(1, n - 1)); cols.append(np.arange(k, n - 2 + k)); vals.append(s[k])
    if reflect_upper:
        h = x[-1] - x[-2]
        rows.append(np.array([n - 1, n - 1])); cols.append(np.array([n - 2, n - 1]))
        vals.append(np.array([2 / h**2, -2 / h**2]))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


class HestonADI:
    """Grid and operators for one contract's model parameters; solve() returns the grid values."""

    def __init__(self, c1, size, v_max, grid=None, far_v="zero_slope"):
        """`grid` = (S, iS, v, jv) reuses another solve's nodes (for bumped parameters).

        far_v: "zero_slope" (V_v = 0 at v_max) or "base_value" (V on the v_max row equals the
        Black-Scholes price at the expected average variance from v_max, a Dirichlet row).
        """
        if far_v not in ("zero_slope", "base_value"):
            raise ValueError(far_v)
        self.c = c1
        self.size = size
        self.far_v = far_v
        if grid is None:
            S0, v0, tau = float(c1.S[0]), float(c1.v0[0]), float(c1.tau[0])
            width = np.sqrt(max(v0, float(c1.theta[0])) * tau)
            s_max = max(4.0, 2 * S0) * np.exp(min(8 * width, 3.0))
            S, iS = _grid(S0, 0.0, s_max, float(np.clip(width, 0.02, 0.25)), 1.0, size.m1)
            v, jv = _grid(v0, 0.0, v_max, v_max / 500, 0.0, size.m2)
            grid = (S, iS, v, jv)
        self.grid = grid
        self.S, self.iS, self.v, self.jv = grid
        self._build()

    def _build(self):
        c = self.c
        r, q = float(c.r[0]), float(c.q[0])
        kappa, theta, xi, rho = (float(getattr(c, p)[0]) for p in ("kappa", "theta", "xi", "rho"))
        S, v = self.S, self.v
        n1, n2 = len(S), len(v)
        D1S, D2S = first_derivative_matrix(S), second_derivative_matrix(S)       # V_SS = 0 at S_max
        D1v = first_derivative_matrix(v)
        D1v_conv = upwind_first_matrix(v) if kappa * theta < kappa * v[-1] else D1v
        D1v_conv = D1v_conv.tolil(); D1v_conv[n2 - 1, :] = 0; D1v_conv = D1v_conv.tocsr()
        D2v = second_derivative_matrix(v, reflect_upper=True)
        # V_v = 0 at v_max: zero the first-derivative row there.
        D1v = D1v.tolil(); D1v[n2 - 1, :] = 0; D1v = D1v.tocsr()
        I1, I2 = sp.identity(n1, format="csr"), sp.identity(n2, format="csr")
        # unknowns ordered V[i, j] -> i * n2 + j (S index major)
        Sd, vd = sp.diags(S), sp.diags(v)
        A1 = (sp.kron(sp.diags(0.5 * S**2) @ D2S, vd) + sp.kron((r - q) * Sd @ D1S, I2)
              - 0.5 * r * sp.identity(n1 * n2)).tocsr()
        A2 = (sp.kron(I1, 0.5 * xi**2 * vd @ D2v + sp.diags(kappa * (theta - v)) @ D1v_conv)
              - 0.5 * r * sp.identity(n1 * n2)).tocsr()
        D1v_mixed = D1v.tolil(); D1v_mixed[0, :] = 0; D1v_mixed = D1v_mixed.tocsr()   # v = 0: no mixed term
        A0 = rho * xi * sp.kron(Sd @ D1S, vd @ D1v_mixed, format="csr")
        # Dirichlet V = 0 at S = 0: drop those unknowns (and the v_max row for "base_value")
        idx = np.arange(n1 * n2).reshape(n1, n2)
        if self.far_v == "base_value":
            keep, bnd = idx[1:, :-1].ravel(), idx[1:, -1]
        else:
            keep, bnd = idx[1:, :].ravel(), np.zeros(0, int)
        self.keep, self.bnd = keep, bnd
        self.A0, self.A1, self.A2 = (M[keep][:, keep].tocsc() for M in (A0, A1, A2))
        self.B = [M[keep][:, bnd].tocsr() for M in (A0, A1, A2)]
        self.A = (self.A0 + self.A1 + self.A2).tocsr()
        self.n = len(keep)

    def boundary_values(self, tau):
        """Call value on the v_max row: Black-Scholes at the expected average variance from v_max."""
        c = self.c
        r, q = float(c.r[0]), float(c.q[0])
        kappa, theta = float(c.kappa[0]), float(c.theta[0])
        S = self.S[1:]
        if tau <= 0:
            return np.maximum(S - 1.0, 0.0)
        vm = self.v[-1]
        sd = np.sqrt(theta * tau - (vm - theta) * np.expm1(-kappa * tau) / kappa)
        x = np.log(S) + (r - q) * tau
        d1 = x / sd + 0.5 * sd
        return np.exp(-r * tau) * (np.exp(x) * ndtr(d1) - ndtr(d1 - sd))

    def forcing(self, tau):
        """(b0, b1, b2): the boundary row's contribution to each split operator."""
        if self.bnd.size == 0:
            return (0.0, 0.0, 0.0)
        g = self.boundary_values(tau)
        return tuple(B @ g for B in self.B)

    def payoff(self):
        S = self.S
        mid = np.concatenate([[S[0]], 0.5 * (S[1:] + S[:-1]), [S[-1]]])
        lo, hi = mid[:-1], mid[1:]
        avg = np.maximum(S - 1.0, 0.0)
        near = (lo < 1.0) & (hi > 1.0)
        # cell average of (S - 1)^+ over [lo, hi]
        avg[near] = 0.5 * (hi[near] - 1.0) ** 2 / (hi[near] - lo[near])
        return np.repeat(avg, len(self.v))[self.keep]

    def solve(self):
        size = self.size
        tau = float(self.c.tau[0])
        dt = tau / size.steps
        U = self.payoff()
        I = sp.identity(self.n, format="csc")
        lu = {}

        def solver(h, th):
            key = (round(h, 15), th)
            if key not in lu:
                lu[key] = (splu((I - th * h * self.A1).tocsc()), splu((I - th * h * self.A2).tocsc()))
            return lu[key]

        # F(t, U) = A U + b(t); b = b0 + b1 + b2 comes from a Dirichlet v_max row (zero otherwise).
        def douglas(U, t, h, th):
            s1, s2 = solver(h, th)
            b_old, b_new = self.forcing(t), self.forcing(t + h)
            Y0 = U + h * (self.A @ U + sum(b_old))
            Y1 = s1.solve(Y0 - th * h * (self.A1 @ U) + th * h * (b_new[1] - b_old[1]))
            return s2.solve(Y1 - th * h * (self.A2 @ U) + th * h * (b_new[2] - b_old[2]))

        t = 0.0
        steps = 0
        for _ in range(size.damping):
            for _ in range(2):
                U = douglas(U, t, dt / 2, 1.0)
                t += dt / 2
            steps += 1
        th = HV_THETA
        s1, s2 = solver(dt, th)
        for _ in range(size.steps - steps):
            b_old, b_new = self.forcing(t), self.forcing(t + dt)
            FU = self.A @ U + sum(b_old)
            Y0 = U + dt * FU
            A1U, A2U = self.A1 @ U, self.A2 @ U
            Y1 = s1.solve(Y0 - th * dt * A1U + th * dt * (b_new[1] - b_old[1]))
            Y2 = s2.solve(Y1 - th * dt * A2U + th * dt * (b_new[2] - b_old[2]))
            Z0 = Y0 + 0.5 * dt * (self.A @ Y2 + sum(b_new) - FU)
            A1Y, A2Y = self.A1 @ Y2, self.A2 @ Y2
            Z1 = s1.solve(Z0 - th * dt * A1Y)
            U = s2.solve(Z1 - th * dt * A2Y)
            t += dt
        V = np.zeros(len(self.S) * len(self.v))
        V[self.keep] = U
        if self.bnd.size:
            V[self.bnd] = self.boundary_values(tau)
        return V.reshape(len(self.S), len(self.v))

    def spot_values(self, V):
        """Call price and grid Greeks at (S0, v0)."""
        i, j = self.iS, self.jv
        S, v = self.S, self.v
        fS, sS = (w[:, i - 1] for w in stencils(S))
        if j == 0:
            raise ValueError("v0 must be an interior node")
        fv, sv = (w[:, j - 1] for w in stencils(v))
        rowsS = V[i - 1:i + 2]                     # (3, n2)
        dS = fS @ rowsS                            # V_S along v
        out = {"price": V[i, j], "delta": dS[j], "gamma": sS @ V[i - 1:i + 2, j],
               "vega": fv @ V[i, j - 1:j + 2], "volga": sv @ V[i, j - 1:j + 2],
               "vanna": fv @ dS[j - 1:j + 2]}
        return {k: float(x) for k, x in out.items()}


def adi_quantities(c, size, v_max, parameter_greeks=True, far_v="zero_slope"):
    """All 12 quantities per contract (puts by parity)."""
    n = len(c)
    names = ("price", "delta", "gamma", "vega", "vanna", "volga")
    out = {k: np.empty(n) for k in names + ("rho_rate", "d_kappa", "d_theta", "d_xi", "d_rho")}
    for i in range(n):
        ci = c.take([i]).with_values(call=True)
        grid = HestonADI(ci, size, v_max, far_v=far_v)
        vals = grid.spot_values(grid.solve())
        for k in names:
            out[k][i] = vals[k]
        if parameter_greeks:
            for name, h in BUMPS.items():
                prices = []
                for sign in (1, -1):
                    cb = ci.with_values(**{name: getattr(ci, name)[0] + sign * h})
                    g = HestonADI(cb, size, v_max, grid=grid.grid, far_v=far_v)
                    prices.append(g.spot_values(g.solve())["price"])
                key = "rho_rate" if name == "r" else f"d_{name}"
                out[key][i] = (prices[0] - prices[1]) / (2 * h)
        else:
            for key in ("rho_rate", "d_kappa", "d_theta", "d_xi", "d_rho"):
                out[key][i] = np.nan
    if not parameter_greeks:
        out["rho_rate"] = np.zeros(n)
    out = apply_parity(c, out)
    out["theta_cal"] = theta_from_pde(c, out)
    if not parameter_greeks:
        out["rho_rate"][:] = np.nan
    return out
