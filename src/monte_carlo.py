from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.sampling import estimate, validate_sampling


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    ci_low: float
    ci_high: float
    method: str
    payoff_evaluations: int
    independent_observations: int
    pilot_samples: int
    beta: float | None
    elapsed_seconds: float


def validate_inputs(S, K, tau, r, sigma, q, option):
    if any(np.ndim(value) != 0 or not np.isfinite(value) for value in (S, K, tau, r, sigma, q)):
        raise ValueError("Financial inputs must be finite scalars")
    if S < 0 or K <= 0 or tau < 0 or sigma < 0:
        raise ValueError("Require S >= 0, K > 0, tau >= 0, and sigma >= 0")
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")


def monte_carlo_price(S, K, tau, r, sigma, q=0.0, option="call", *,
                      n_samples=16384, method="plain", seed=42,
                      batch_size=65536, pilot_samples=1024, n_scrambles=8):
    """Total payoff budget includes pairs and independent pilots; intervals are approximate 95%."""
    start = perf_counter()
    validate_inputs(S, K, tau, r, sigma, q, option)
    validate_sampling(n_samples, method, batch_size, pilot_samples, n_scrambles)
    sign = 1 if option == "call" else -1
    if tau == 0 or sigma == 0 or S == 0:
        price = float(max(sign * (S * np.exp(-q * tau) - K * np.exp(-r * tau)), 0))
        return MonteCarloResult(price, 0.0, price, price, method, 0, 0, 0, None,
                                perf_counter() - start)
    discount = np.exp(-r * tau)
    drift = (r - q - 0.5 * sigma**2) * tau
    spread = sigma * np.sqrt(tau)

    def payoffs(z):
        terminal = S * np.exp(drift + spread * z)
        return (discount * terminal)[:, None], (discount * np.maximum(sign * (terminal - K), 0))[:, None]

    result = estimate(payoffs, np.array([S * np.exp(-q * tau)]), n_samples,
                      method, seed, batch_size, pilot_samples, n_scrambles)
    price, se = float(result.mean[0]), float(result.se[0])
    return MonteCarloResult(
        price, se, price - result.critical * se, price + result.critical * se,
        method, int(n_samples), result.observations, result.pilot_samples,
        float(result.beta[0]) if result.beta is not None else None, perf_counter() - start,
    )
