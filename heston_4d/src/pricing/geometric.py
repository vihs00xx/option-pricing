"""Closed-form geometric-basket prices and sensitivities for d stocks.

log G_T = sum_i w_i log S_i(T) is Gaussian, so the geometric basket option is a
lognormal (Black) payoff. It is the exact control variate and validation target
for the arithmetic estimators.
"""

import numpy as np
from scipy.special import ndtr

from .contract import parameter_bumps, spot_names


def lognormal_value(log_mean, variance, strike, sign):
    """E[(sign * (X - K))^+] for log X ~ N(log_mean, variance), undiscounted."""
    log_mean = np.asarray(log_mean, dtype=float)
    mean = np.exp(log_mean + 0.5 * variance)
    if variance <= 0:
        return np.maximum(sign * (mean - strike), 0)
    sd = np.sqrt(variance)
    d2 = (log_mean - np.log(strike)) / sd
    return sign * (mean * ndtr(sign * (d2 + sd)) - strike * ndtr(sign * d2))


def geometric_parameters(c):
    """Mean and variance of log G_T."""
    w = np.asarray(c.weights)
    active = w > 0
    log_mean = float(w[active] @ (np.log(np.asarray(c.spots)[active]) + c.log_drift()[active]))
    return log_mean, float(w @ c.covariance() @ w)


def geometric_price(c):
    if c.tau == 0:
        return _expiry(c)
    m, v = geometric_parameters(c)
    return float(np.exp(-c.rate * c.tau) * lognormal_value(m, v, c.strike, c.sign))


def _expiry(c):
    w = np.asarray(c.weights)
    active = w > 0
    value = np.exp(w[active] @ np.log(np.asarray(c.spots)[active]))
    return float(max(c.sign * (value - c.strike), 0))


def _log_mean_derivatives(c, m, v):
    """First and second derivatives of the discounted price in log_mean."""
    sd = np.sqrt(v)
    d1 = (m - np.log(c.strike) + v) / sd
    discounted_mean = np.exp(-c.rate * c.tau + m + 0.5 * v)
    first = c.sign * discounted_mean * ndtr(c.sign * d1)
    second = first + discounted_mean * np.exp(-0.5 * d1**2) / (np.sqrt(2 * np.pi) * sd)
    return first, second


def geometric_greeks(c):
    """Spot Greeks and parameter sensitivities, keyed like the arithmetic estimators."""
    m, v = geometric_parameters(c)
    if c.tau <= 0 or v <= 0:
        raise ValueError("Geometric Greeks require positive maturity and basket variance")
    d = c.dimension
    w, s, spots = (np.asarray(x) for x in (c.weights, c.volatilities, c.spots))
    corr = c.correlation_matrix()
    first, second = _log_mean_derivatives(c, m, v)
    price = geometric_price(c)
    # The price depends on v only through E[f(m + sqrt(v) Z)], so dP/dv = P''(m) / 2.
    values = dict(price=price)
    values.update({f"delta_{i + 1}": float(first * w[i] / spots[i]) for i in range(d)})
    for i in range(d):
        for j in range(i, d):
            values[f"gamma_{i + 1}{j + 1}"] = float(
                (second * w[i] * w[j] - (first * w[i] if i == j else 0.0)) / (spots[i] * spots[j]))
    variance_gradient = 2 * c.tau * w * (corr @ (w * s))
    for i in range(d):
        values[f"vega_{i + 1}"] = float(-first * w[i] * s[i] * c.tau + 0.5 * second * variance_gradient[i])
    drift_rate = float(w @ (c.rate - np.asarray(c.dividends) - 0.5 * s**2))
    values["theta"] = float(c.rate * price - first * drift_rate - 0.5 * second * v / c.tau)
    values["rho"] = float(c.tau * (first - price))
    for i in range(d):
        for j in range(i + 1, d):
            values[f"correlation_{i + 1}{j + 1}"] = float(second * c.tau * w[i] * w[j] * s[i] * s[j])
    return values


def geometric_known_values(c, scale=1.0):
    """Exact means for the geometric control: analytic spot Greeks and the same
    central bumps that the sampled parameter sensitivities use."""
    exact = geometric_greeks(c)
    values = [exact[name] for name in spot_names(c.dimension)]
    for _, field, index, step, sign in parameter_bumps(c, scale):
        values.append(sign * (geometric_price(c.bumped(field, step, index))
                              - geometric_price(c.bumped(field, -step, index))) / (2 * step))
    return np.array(values)
