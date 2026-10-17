"""European arithmetic basket options on d correlated stocks under risk-neutral GBM.

Time is in years; rates, dividend yields and volatilities are decimals.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Basket:
    """A scalar correlation means equal correlation between every pair."""

    spots: tuple
    strike: float
    tau: float
    rate: float
    volatilities: tuple
    dividends: tuple
    weights: tuple
    correlation: object = 0.0
    option: str = "call"

    def __post_init__(self):
        spots = np.atleast_1d(np.asarray(self.spots, dtype=float))
        if spots.ndim != 1 or spots.size < 1:
            raise ValueError("spots must be a non-empty vector")
        d = spots.size
        for name in ("spots", "volatilities", "dividends", "weights"):
            values = np.atleast_1d(np.asarray(getattr(self, name), dtype=float))
            if values.shape != (d,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain {d} finite values")
            object.__setattr__(self, name, tuple(float(x) for x in values))
        matrix = np.asarray(self.correlation, dtype=float)
        if matrix.ndim == 0:
            matrix = np.full((d, d), float(matrix))
            np.fill_diagonal(matrix, 1.0)
        if matrix.shape != (d, d) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"correlation must be a scalar or a finite {d}x{d} matrix")
        if not np.allclose(matrix, matrix.T, atol=1e-14, rtol=0):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(matrix), 1.0, atol=1e-14, rtol=0) or np.any(np.abs(matrix) > 1):
            raise ValueError("correlation must have a unit diagonal and entries in [-1, 1]")
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 1.0)
        if np.linalg.eigvalsh(matrix)[0] <= 1e-12:
            raise ValueError("correlation must be positive definite")
        object.__setattr__(self, "correlation", tuple(tuple(float(x) for x in row) for row in matrix))
        if not np.all(np.isfinite([self.strike, self.tau, self.rate])):
            raise ValueError("strike, tau and rate must be finite")
        if self.strike <= 0 or self.tau < 0:
            raise ValueError("strike must be positive and tau non-negative")
        if min(self.spots) <= 0 or min(self.volatilities) <= 0:
            raise ValueError("spots and volatilities must be positive")
        if min(self.weights) < 0 or not np.isclose(sum(self.weights), 1, atol=1e-12, rtol=0):
            raise ValueError("weights must be non-negative and sum to one")
        if self.option not in ("call", "put"):
            raise ValueError("option must be call or put")

    @property
    def dimension(self):
        return len(self.spots)

    @property
    def sign(self):
        return 1 if self.option == "call" else -1

    def correlation_matrix(self):
        return np.array(self.correlation)

    def covariance(self):
        """Covariance of the log returns over the remaining life."""
        s = np.asarray(self.volatilities)
        return np.outer(s, s) * self.correlation_matrix() * self.tau

    def log_drift(self):
        """Mean log return of each stock over the remaining life."""
        s = np.asarray(self.volatilities)
        return (self.rate - np.asarray(self.dividends) - 0.5 * s**2) * self.tau

    def changed(self, **kwargs):
        return replace(self, **kwargs)

    def bumped(self, name, shift, index=None):
        """Shift one input; a correlation index (i, j) moves both symmetric entries."""
        if name == "correlation":
            i, j = index
            matrix = self.correlation_matrix()
            matrix[i, j] += shift
            matrix[j, i] += shift
            return self.changed(correlation=matrix)
        if index is None:
            return self.changed(**{name: getattr(self, name) + shift})
        values = list(getattr(self, name))
        values[index] += shift
        return self.changed(**{name: tuple(values)})

    def forward_value(self):
        """Discounted expected basket value, the parity term."""
        return float(np.dot(self.weights, np.asarray(self.spots)
                            * np.exp(-np.asarray(self.dividends) * self.tau)))

    def payoff(self, terminal):
        terminal = np.asarray(terminal, dtype=float)
        if terminal.ndim == 0 or terminal.shape[-1] != self.dimension:
            raise ValueError(f"terminal stocks must have last dimension {self.dimension}")
        return np.maximum(self.sign * (terminal @ np.asarray(self.weights) - self.strike), 0)


def spot_names(d):
    """Price, Deltas and the upper triangle of the spot Hessian."""
    return (["price"] + [f"delta_{i + 1}" for i in range(d)]
            + [f"gamma_{i + 1}{j + 1}" for i in range(d) for j in range(i, d)])


def parameter_bumps(contract, scale=1.0):
    """(name, field, index, step, sign) for central-difference sensitivities.

    Vega and correlation sensitivities use unit decimal changes; theta is
    calendar time (minus the maturity derivative) per year; rho is per unit rate.
    """
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("bump scale must lie in (0, 1]")
    c = contract
    d = c.dimension
    smallest = np.linalg.eigvalsh(c.correlation_matrix())[0]
    bumps = [(f"vega_{i + 1}", "volatilities", i, min(1e-4, c.volatilities[i] / 4), 1) for i in range(d)]
    bumps.append(("theta", "tau", None, min(1e-5, c.tau / 1000), -1))
    bumps.append(("rho", "rate", None, 1e-4, 1))
    for i in range(d):
        for j in range(i + 1, d):
            # A symmetric unit bump moves each eigenvalue by at most its size.
            step = min(1e-4, (1 - abs(c.correlation[i][j])) / 4, smallest / 4)
            bumps.append((f"correlation_{i + 1}{j + 1}", "correlation", (i, j), step, 1))
    return [(name, field, index, step * scale, sign) for name, field, index, step, sign in bumps]


def quantity_names(contract):
    return spot_names(contract.dimension) + [bump[0] for bump in parameter_bumps(contract)]
