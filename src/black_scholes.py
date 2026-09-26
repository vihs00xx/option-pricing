"""Time in years; annual rates, dividend yield, and volatility as decimals."""

import numpy as np
from scipy.special import ndtr


def _prepare_inputs(S, K, tau, r, sigma, q, option):
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")

    values = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in (S, K, tau, r, sigma, q)]
    )
    for name, value in zip(("S", "K", "tau", "r", "sigma", "q"), values):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")

    S, K, tau, r, sigma, q = values
    if np.any(S < 0) or np.any(K <= 0):
        raise ValueError("S must be non-negative and K must be positive")
    if np.any(tau < 0) or np.any(sigma < 0):
        raise ValueError("tau and sigma must be non-negative")
    return S, K, tau, r, sigma, q


def _as_output(value):
    return float(value) if value.ndim == 0 else value


def _d1_d2(S, K, tau, r, sigma, q):
    spread = sigma * np.sqrt(tau)
    d1 = (np.log(S) - np.log(K) + (r - q + 0.5 * sigma**2) * tau) / spread
    return d1, d1 - spread


def black_scholes_price(S, K, tau, r, sigma, q=0.0, option="call"):
    S, K, tau, r, sigma, q = _prepare_inputs(S, K, tau, r, sigma, q, option)
    discounted_spot = S * np.exp(-q * tau)
    discounted_strike = K * np.exp(-r * tau)
    sign = 1 if option == "call" else -1

    # This also gives the correct price at expiry or zero volatility.
    price = np.array(np.maximum(sign * (discounted_spot - discounted_strike), 0.0))
    regular = (S > 0) & (tau > 0) & (sigma > 0)
    if np.any(regular):
        d1, d2 = _d1_d2(
            S[regular], K[regular], tau[regular], r[regular],
            sigma[regular], q[regular],
        )
        price[regular] = sign * (
            discounted_spot[regular] * ndtr(sign * d1)
            - discounted_strike[regular] * ndtr(sign * d2)
        )
    return _as_output(price)


def black_scholes_greeks(S, K, tau, r, sigma, q=0.0, option="call"):
    """Vega/rho are per unit decimal change; theta is -dV/dtau per year.
    Requires positive S, tau, and sigma.
    """
    S, K, tau, r, sigma, q = _prepare_inputs(S, K, tau, r, sigma, q, option)
    if np.any(S == 0) or np.any(tau == 0) or np.any(sigma == 0):
        raise ValueError("Greeks require positive S, tau, and sigma")

    d1, d2 = _d1_d2(S, K, tau, r, sigma, q)
    density = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    spot_discount = np.exp(-q * tau)
    strike_discount = np.exp(-r * tau)
    sign = 1 if option == "call" else -1

    delta = sign * spot_discount * ndtr(sign * d1)
    gamma = spot_discount * density / (S * sigma * np.sqrt(tau))
    vega = S * spot_discount * density * np.sqrt(tau)
    theta = (
        -S * spot_discount * density * sigma / (2 * np.sqrt(tau))
        + sign * q * S * spot_discount * ndtr(sign * d1)
        - sign * r * K * strike_discount * ndtr(sign * d2)
    )
    rho = sign * K * tau * strike_discount * ndtr(sign * d2)

    return {
        name: _as_output(value)
        for name, value in zip(
            ("delta", "gamma", "vega", "theta", "rho"),
            (delta, gamma, vega, theta, rho),
        )
    }
