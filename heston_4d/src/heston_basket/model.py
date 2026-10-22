"""European arithmetic basket options on d stocks, each with its own Heston variance.

For stock i, under the risk-neutral measure,

    dS_i / S_i = (r - q_i) dt + sqrt(v_i) dW_i,
    dv_i = kappa_i (theta_i - v_i) dt + xi_i sqrt(v_i) dZ_i.

The variance drivers Z have correlation matrix C (``vol_correlation``). Each stock's
Brownian motion is W_i = rho_i Z_i + sqrt(1 - rho_i^2) Y_i, with rho_i the stock's
leverage and Y independent of Z with correlation matrix P. The contract gives the stock
correlation R (``correlation``) = d<W_i, W_j> / dt and C; P follows from

    R_ij = rho_i rho_j C_ij + rbar_i rbar_j P_ij,   rbar_i = sqrt(1 - rho_i^2),

and must be a valid correlation matrix. Then d<W_i, Z_j> = rho_i C_ij: a stock moves with
its own variance through its leverage and with the other variances through C. Every
(W, Z) correlation is valid by construction, and for one stock this is the Heston model.

Time is in years; rates, dividend yields, variances and volatilities of variance are
decimals.
"""

from dataclasses import dataclass, replace

import numpy as np

PER_STOCK = ("spots", "dividends", "weights", "v0", "kappa", "theta", "xi", "leverage")
MIN_EIGENVALUE = 1e-10


def correlation_matrix(value, d, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim == 0:
        matrix = np.full((d, d), float(matrix))
        np.fill_diagonal(matrix, 1.0)
    if matrix.shape != (d, d) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a scalar or a finite {d}x{d} matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-14, rtol=0):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-14, rtol=0) or np.any(np.abs(matrix) > 1):
        raise ValueError(f"{name} must have a unit diagonal and entries in [-1, 1]")
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    return matrix


def residual_correlation(R, C, leverage):
    """P from R = (rho rho') o C + (rbar rbar') o P."""
    rho = np.asarray(leverage, dtype=float)
    rbar = np.sqrt(1 - rho ** 2)
    P = (np.asarray(R) - np.outer(rho, rho) * np.asarray(C)) / np.outer(rbar, rbar)
    np.fill_diagonal(P, 1.0)
    return P


@dataclass(frozen=True)
class HestonBasket:
    """A scalar correlation means equal correlation between every pair."""

    spots: tuple
    strike: float
    tau: float
    rate: float
    dividends: tuple
    weights: tuple
    v0: tuple
    kappa: tuple
    theta: tuple
    xi: tuple
    leverage: tuple
    correlation: object = 0.0
    vol_correlation: object = 0.0
    option: str = "call"

    def __post_init__(self):
        spots = np.atleast_1d(np.asarray(self.spots, dtype=float))
        if spots.ndim != 1 or spots.size < 1:
            raise ValueError("spots must be a non-empty vector")
        d = spots.size
        for name in PER_STOCK:
            values = np.atleast_1d(np.asarray(getattr(self, name), dtype=float))
            if values.shape != (d,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain {d} finite values")
            object.__setattr__(self, name, tuple(float(x) for x in values))
        R = correlation_matrix(self.correlation, d, "correlation")
        C = correlation_matrix(self.vol_correlation, d, "vol_correlation")
        if not np.all(np.isfinite([self.strike, self.tau, self.rate])):
            raise ValueError("strike, tau and rate must be finite")
        if self.strike <= 0 or self.tau < 0:
            raise ValueError("strike must be positive and tau non-negative")
        if min(self.spots) <= 0 or min(self.v0) < 0 or min(self.kappa) <= 0 or min(self.theta) <= 0 \
                or min(self.xi) <= 0:
            raise ValueError("spots, kappa, theta and xi must be positive and v0 non-negative")
        if max(abs(x) for x in self.leverage) >= 1:
            raise ValueError("leverage must lie in (-1, 1)")
        if min(self.weights) < 0 or not np.isclose(sum(self.weights), 1, atol=1e-12, rtol=0):
            raise ValueError("weights must be non-negative and sum to one")
        if self.option not in ("call", "put"):
            raise ValueError("option must be call or put")
        for name, matrix in (("correlation", R), ("vol_correlation", C),
                             ("residual correlation P", residual_correlation(R, C, self.leverage))):
            if np.linalg.eigvalsh(matrix)[0] <= MIN_EIGENVALUE:
                raise ValueError(f"{name} must be positive definite")
            if name.startswith("residual") and np.any(np.abs(matrix) > 1 + 1e-12):
                raise ValueError("the residual correlation P has entries outside [-1, 1]")
        object.__setattr__(self, "correlation", tuple(tuple(float(x) for x in row) for row in R))
        object.__setattr__(self, "vol_correlation", tuple(tuple(float(x) for x in row) for row in C))

    @property
    def dimension(self):
        return len(self.spots)

    @property
    def sign(self):
        return 1 if self.option == "call" else -1

    def correlation_matrix(self):
        return np.array(self.correlation)

    def vol_correlation_matrix(self):
        return np.array(self.vol_correlation)

    def residual_correlation_matrix(self):
        return residual_correlation(self.correlation_matrix(), self.vol_correlation_matrix(), self.leverage)

    def forwards(self):
        """f_i = w_i S_i exp((r - q_i) tau) / K, the normalized forward of each basket term."""
        w, s, q = (np.asarray(x) for x in (self.weights, self.spots, self.dividends))
        return w * s * np.exp((self.rate - q) * self.tau) / self.strike

    def discounted_strike(self):
        return self.strike * np.exp(-self.rate * self.tau)

    def changed(self, **kwargs):
        return replace(self, **kwargs)

    def bumped(self, name, shift, index=None):
        """Shift one input; a correlation index (i, j) moves both symmetric entries."""
        if name in ("correlation", "vol_correlation"):
            i, j = index
            matrix = np.array(getattr(self, name))
            matrix[i, j] += shift
            matrix[j, i] += shift
            return self.changed(**{name: matrix})
        if index is None:
            return self.changed(**{name: getattr(self, name) + shift})
        values = list(getattr(self, name))
        values[index] += shift
        return self.changed(**{name: tuple(values)})

    def forward_value(self):
        """Discounted expected basket value, the parity term."""
        return float(np.dot(self.weights, np.asarray(self.spots) * np.exp(-np.asarray(self.dividends) * self.tau)))

    def payoff(self, terminal):
        terminal = np.asarray(terminal, dtype=float)
        return np.maximum(self.sign * (terminal @ np.asarray(self.weights) - self.strike), 0)


def pairs(d):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def spot_names(d):
    """Price, Deltas and the upper triangle of the spot Hessian."""
    return (["price"] + [f"delta_{i + 1}" for i in range(d)]
            + [f"gamma_{i + 1}{j + 1}" for i in range(d) for j in range(i, d)])


def quantity_names(d):
    """Every reported quantity: spot Greeks, then theta and rho, then the parameter sensitivities."""
    n = range(1, d + 1)
    return (spot_names(d) + ["theta", "rho"] + [f"vega_{i}" for i in n] + [f"d_kappa_{i}" for i in n]
            + [f"d_theta_{i}" for i in n] + [f"d_xi_{i}" for i in n] + [f"d_leverage_{i}" for i in n]
            + [f"correlation_{i + 1}{j + 1}" for i, j in pairs(d)]
            + [f"vol_correlation_{i + 1}{j + 1}" for i, j in pairs(d)])


BUMPS = {"v0": 0.02, "kappa": 0.02, "theta": 0.02, "xi": 0.02, "leverage": 0.02, "correlation": 0.01,
         "vol_correlation": 0.01, "tau": 0.02}


def _valid(contract, field, step, index):
    """Whether the contract bumped by ``step`` is valid (the bump itself validates it)."""
    try:
        contract.bumped(field, step, index)
        return True
    except ValueError:
        return False


def parameter_bumps(c, scale=1.0):
    """(name, field, index, step) for the central common-random-number sensitivities.

    Variance-path parameters move by 2% of their value (leverage by 0.02, correlations by
    0.01), the sizes of the one-stock study's Monte Carlo Greeks: QE paths are only
    piecewise smooth in the parameters, so smaller steps add variance without removing
    bias that matters. A step is halved until both bumped contracts are valid; theta's
    step is 2% of tau with the number of time steps held fixed.
    """
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("bump scale must lie in (0, 1]")
    d = c.dimension
    out = [("theta", "tau", None, BUMPS["tau"] * c.tau * scale)]
    for field, prefix in (("v0", "vega"), ("kappa", "d_kappa"), ("theta", "d_theta"), ("xi", "d_xi"),
                          ("leverage", "d_leverage")):
        for i in range(d):
            base = getattr(c, field)[i]
            step = BUMPS[field] * (1.0 if field == "leverage" else max(base, 1e-3)) * scale
            out.append((f"{prefix}_{i + 1}", field, i, step))
    for field in ("correlation", "vol_correlation"):
        for i, j in pairs(d):
            out.append((f"{field}_{i + 1}{j + 1}", field, (i, j), BUMPS[field] * scale))
    checked = []
    for name, field, index, step in out:
        for _ in range(30):
            if _valid(c, field, step, index) and _valid(c, field, -step, index):
                break
            step /= 2
        else:
            raise ValueError(f"no valid bump for {name}")
        checked.append((name, field, index, step))
    return checked
