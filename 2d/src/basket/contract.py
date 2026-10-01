from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class BasketContract:
    spots: tuple = (100.0, 100.0)
    strike: float = 100.0
    tau: float = 1.0
    rate: float = 0.05
    volatilities: tuple = (0.2, 0.3)
    dividends: tuple = (0.0, 0.0)
    weights: tuple = (0.5, 0.5)
    correlation: float = 0.5
    option: str = "call"

    def __post_init__(self):
        for name in ("spots", "volatilities", "dividends", "weights"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain two finite values")
            object.__setattr__(self, name, tuple(float(x) for x in values))
        scalars = (self.strike, self.tau, self.rate, self.correlation)
        if not np.all(np.isfinite(scalars)):
            raise ValueError("Contract parameters must be finite")
        if self.strike <= 0 or self.tau < 0:
            raise ValueError("strike must be positive and tau non-negative")
        if min(self.spots) < 0 or min(self.volatilities) < 0:
            raise ValueError("spots and volatilities must be non-negative")
        if min(self.weights) < 0 or not np.isclose(sum(self.weights), 1, atol=1e-12, rtol=0):
            raise ValueError("weights must be non-negative and sum to one")
        if abs(self.correlation) > 1:
            raise ValueError("correlation must lie in [-1, 1]")
        if self.option not in ("call", "put"):
            raise ValueError("option must be call or put")

    def changed(self, **kwargs):
        return replace(self, **kwargs)

    def bumped(self, name, shift, index=None):
        if index is None:
            return self.changed(**{name: getattr(self, name) + shift})
        values = list(getattr(self, name))
        values[index] += shift
        return self.changed(**{name: tuple(values)})

    def forward_value(self):
        return float(np.dot(self.weights, np.asarray(self.spots)
                            * np.exp(-np.asarray(self.dividends) * self.tau)))

    def payoff(self, terminal, kind="arithmetic"):
        terminal = np.asarray(terminal, dtype=float)
        if terminal.ndim == 0 or terminal.shape[-1] != 2 or not np.all(np.isfinite(terminal)) or np.any(terminal < 0):
            raise ValueError("terminal stocks must be non-negative with last dimension two")
        if kind == "arithmetic":
            basket = terminal @ np.asarray(self.weights)
        elif kind == "geometric":
            active = np.asarray(self.weights) > 0
            with np.errstate(divide="ignore"):
                basket = np.exp(np.log(terminal[..., active]) @ np.asarray(self.weights)[active])
        else:
            raise ValueError("kind must be arithmetic or geometric")
        sign = 1 if self.option == "call" else -1
        return np.maximum(sign * (basket - self.strike), 0)
