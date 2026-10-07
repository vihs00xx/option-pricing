"""Little-trap characteristic function of X = log(S_tau / F) (Albrecher et al. 2007).

    b = kappa - rho xi i u,  d = sqrt(b^2 + xi^2 (i u + u^2)),  g = (b - d)/(b + d)
    C = kappa theta / xi^2 ((b - d) tau - 2 log((1 - g e^{-d tau}) / (1 - g)))
    D = (b - d) / xi^2 (1 - e^{-d tau}) / (1 - g e^{-d tau}),   phi(u) = exp(C + D v0)

The code evaluates the same expressions in a form that stays accurate as xi -> 0:
(b - d)/xi^2 = -psi/(b + d) with psi = i u + u^2, g/xi^2 = -psi/(b + d)^2, the log
as log1p, and 1 - e^{-d tau} as -expm1(-d tau). Nothing is divided by xi^2 explicitly.

`cf_parts` runs on plain complex arrays or on `Dual` numbers, which carry exact
first derivatives with respect to (kappa, theta, xi, rho).
"""

import math

import numpy as np

SERIES_CUTOFF = 1e-2


class Dual:
    """Forward-mode value with gradient `g` of shape (n_params,) + value.shape; g=None is zero."""

    __slots__ = ("v", "g")
    __array_ufunc__ = None  # make ndarray (op) Dual defer to the reflected Dual method

    def __init__(self, v, g=None):
        self.v = v
        self.g = g

    @staticmethod
    def variables(values):
        """One Dual per entry of `values`, seeded with unit gradients."""
        n = len(values)
        out = []
        for k, value in enumerate(values):
            value = np.asarray(value)
            g = np.zeros((n,) + value.shape, dtype=value.dtype)
            g[k] = 1.0
            out.append(Dual(value, g))
        return out

    def __add__(self, o):
        o = _lift(o)
        return Dual(self.v + o.v, _add(self.g, o.g))

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, None if self.g is None else -self.g)

    def __sub__(self, o):
        return self + (-_lift(o))

    def __rsub__(self, o):
        return _lift(o) + (-self)

    def __mul__(self, o):
        o = _lift(o)
        return Dual(self.v * o.v, _add(_scale(self.g, o.v), _scale(o.g, self.v)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _lift(o)
        q = self.v / o.v
        return Dual(q, _scale(_add(self.g, _scale(o.g, -q)), 1 / o.v))

    def __rtruediv__(self, o):
        return _lift(o) / self


def _lift(a):
    return a if isinstance(a, Dual) else Dual(a)


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _scale(g, x):
    return None if g is None else g * x


def _unary(a, value, slope):
    if isinstance(a, Dual):
        return Dual(value, _scale(a.g, slope))
    return value


def _val(a):
    return a.v if isinstance(a, Dual) else a


def csqrt(a):
    v = np.sqrt(_val(a))
    return _unary(a, v, 0.5 / v if isinstance(a, Dual) else None)


def cexp(a):
    v = np.exp(_val(a))
    return _unary(a, v, v)


def expm1_c(z):
    """Complex expm1 without cancellation: (e^a - 1) cos b - 2 sin^2(b/2) + i e^a sin b."""
    a, b = z.real, z.imag
    s = np.sin(0.5 * b)
    return (np.expm1(a) * np.cos(b) - 2 * s * s) + 1j * (np.exp(a) * np.sin(b))


def cexpm1(a):
    z = _val(a)
    return _unary(a, expm1_c(z), np.exp(z) if isinstance(a, Dual) else None)


def log1p_c(w):
    """Complex log(1 + w) with an accurate real part for small w."""
    a, b = w.real, w.imag
    return 0.5 * np.log1p(a * (2 + a) + b * b) + 1j * np.arctan2(b, 1 + a)


def log1p_ratio(w):
    """log(1 + w) / w, by series when |w| < SERIES_CUTOFF."""
    w = np.asarray(w)
    small = np.abs(w) < SERIES_CUTOFF
    series = 1 + w * (-1 / 2 + w * (1 / 3 + w * (-1 / 4 + w * (1 / 5 + w * (-1 / 6 + w * (1 / 7 - w / 8))))))
    safe = np.where(small, 1.0, w)
    return np.where(small, series, log1p_c(safe) / safe)


def log1p_ratio_slope(w):
    """d/dw [log(1 + w) / w]."""
    w = np.asarray(w)
    small = np.abs(w) < SERIES_CUTOFF
    series = -1 / 2 + w * (2 / 3 + w * (-3 / 4 + w * (4 / 5 + w * (-5 / 6 + w * (6 / 7 - w * 7 / 8)))))
    safe = np.where(small, 1.0, w)
    direct = (1 / (1 + safe) - log1p_c(safe) / safe) / safe
    return np.where(small, series, direct)


def clog1p_ratio(a):
    w = _val(a)
    return _unary(a, log1p_ratio(w), log1p_ratio_slope(w) if isinstance(a, Dual) else None)


def cf_parts(u, tau, kappa, theta, xi, rho):
    """Return (C, D) at complex u; parameters may be arrays or Dual numbers (broadcasting)."""
    psi = u * (u + 1j)
    b = kappa - rho * xi * (1j * u)
    xi2 = xi * xi
    d = csqrt(b * b + xi2 * psi)
    bpd = b + d
    b_minus_d_over_xi2 = -psi / bpd
    g_over_xi2 = -psi / (bpd * bpd)
    g = xi2 * g_over_xi2
    one_minus_e = -cexpm1(-(d * tau))
    e = 1 - one_minus_e
    D = b_minus_d_over_xi2 * one_minus_e / (1 - g * e)
    w_over_xi2 = g_over_xi2 * one_minus_e / (1 - g)
    log_over_xi2 = w_over_xi2 * clog1p_ratio(xi2 * w_over_xi2)
    C = kappa * theta * (b_minus_d_over_xi2 * tau - 2 * log_over_xi2)
    return C, D


def cf(u, tau, v0, kappa, theta, xi, rho):
    """phi(u) = E[exp(i u X)], X = log(S_tau / F)."""
    C, D = cf_parts(u, tau, kappa, theta, xi, rho)
    return np.exp(C + D * v0)


def cumulants(tau, v0, kappa, theta, xi, rho, nodes=64):
    """Mean and variance of X = log(S_tau / F).

    With I = int_0^tau v dt and int sqrt(v) dW2 = (v_tau - v0 - kappa theta tau + kappa I)/xi,
    X = a I + (rho/xi) v_tau + const + sqrt(1 - rho^2) int sqrt(v) dW_perp, a = rho kappa/xi - 1/2, so
    Var X = a^2 Var I + 2 a (rho/xi) Cov(I, v_tau) + (rho/xi)^2 Var v_tau + (1 - rho^2) E I.
    The CIR covariances are integrated over [0, tau] by Gauss-Legendre (smooth integrands).
    """
    tau, v0, kappa, theta, xi, rho = np.broadcast_arrays(
        *[np.asarray(a, dtype=float) for a in (tau, v0, kappa, theta, xi, rho)])
    om = -np.expm1(-kappa * tau)
    mean_I = theta * tau + (v0 - theta) * om / kappa

    def var_v(t, kappa=kappa, theta=theta, xi=xi, v0=v0):
        e = np.exp(-kappa * t)
        return xi**2 * (v0 * (e - e * e) / kappa + theta * (1 - e) ** 2 / (2 * kappa))

    t_gl, w_gl = np.polynomial.legendre.leggauss(nodes)
    s = 0.5 * tau[..., None] * (t_gl + 1)
    w = 0.5 * tau[..., None] * w_gl
    k = kappa[..., None]
    vs = var_v(s, k, theta[..., None], xi[..., None], v0[..., None])
    cov_I_vT = (w * np.exp(-k * (tau[..., None] - s)) * vs).sum(-1)
    var_I = (w * 2 * vs * -np.expm1(-k * (tau[..., None] - s)) / k).sum(-1)
    a = rho * kappa / xi - 0.5
    var_X = (a**2 * var_I + 2 * a * (rho / xi) * cov_I_vT + (rho / xi) ** 2 * var_v(tau)
             + (1 - rho**2) * mean_I)
    return -0.5 * mean_I, var_X


def numerical_cumulants(tau, v0, kappa, theta, xi, rho, orders=(1, 2, 4), radius=0.1, points=64):
    """Cumulants of X from log phi on a small circle: c_n = n! [u^n] log phi(u) / i^n.

    log phi is analytic near u = 0 (moments of S_tau^p exist for small |p|), so the
    trapezoid rule on |u| = radius converges geometrically.
    """
    args = [np.asarray(a, dtype=float)[..., None] for a in (tau, v0, kappa, theta, xi, rho)]
    t = 2 * np.pi * np.arange(points) / points
    u = radius * np.exp(1j * t)
    C, D = cf_parts(u, args[0], *args[2:])
    logphi = C + D * args[1]
    out = []
    for n in orders:
        coef = (logphi * np.exp(-1j * n * t)).mean(-1) / radius**n
        out.append((math.factorial(n) * coef / (1j) ** n).real)
    return out
