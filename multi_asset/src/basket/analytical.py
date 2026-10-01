import numpy as np
from scipy.special import ndtr


def geometric_parameters(contract):
    w = np.asarray(contract.weights)
    s = np.asarray(contract.volatilities)
    cov = np.outer(s, s) * np.array([[1, contract.correlation], [contract.correlation, 1]])
    variance_rate = float(w @ cov @ w)
    active = w > 0
    with np.errstate(divide="ignore"):
        log_spot = float(w[active] @ np.log(np.asarray(contract.spots)[active]))
    drift = float(w @ (contract.rate - np.asarray(contract.dividends) - 0.5 * s**2))
    return log_spot + drift * contract.tau, max(variance_rate, 0) * contract.tau, drift


def lognormal_payoff(log_mean, variance, strike, option="call"):
    log_mean, strike = np.broadcast_arrays(np.asarray(log_mean, float), np.asarray(strike, float))
    mean = np.exp(log_mean + 0.5 * variance)
    sign = 1 if option == "call" else -1
    result = np.maximum(sign * (mean - strike), 0)
    if variance > 0:
        valid = (strike > 0) & np.isfinite(log_mean)
        sd = np.sqrt(variance)
        d2 = (log_mean[valid] - np.log(strike[valid])) / sd
        d1 = d2 + sd
        result = np.array(result, copy=True)
        result[valid] = sign * (mean[valid] * ndtr(sign * d1)
                                - strike[valid] * ndtr(sign * d2))
    return result


def geometric_price(contract):
    mean, variance, _ = geometric_parameters(contract)
    return float(np.exp(-contract.rate * contract.tau)
                 * lognormal_payoff(mean, variance, contract.strike, contract.option))


def geometric_greeks(contract):
    m, v, drift = geometric_parameters(contract)
    if contract.tau <= 0 or v <= 0 or min(contract.spots) <= 0:
        raise ValueError("Geometric Greeks require positive maturity, spots and basket variance")
    w, s, spots = map(np.asarray, (contract.weights, contract.volatilities, contract.spots))
    sd = np.sqrt(v)
    d1 = (m - np.log(contract.strike) + v) / sd
    sign = 1 if contract.option == "call" else -1
    discounted_mean = np.exp(-contract.rate * contract.tau + m + 0.5 * v)
    pm = sign * discounted_mean * ndtr(sign * d1)
    pmm = pm + discounted_mean * np.exp(-0.5 * d1**2) / (np.sqrt(2 * np.pi) * sd)
    delta = pm * w / spots
    gamma = (pmm * np.outer(w, w) - pm * np.diag(w)) / np.outer(spots, spots)
    covariance_gradient = 2 * w * (w * s + w[::-1] * s[::-1] * contract.correlation)
    vega = -pm * w * s * contract.tau + 0.5 * pmm * covariance_gradient * contract.tau
    price = geometric_price(contract)
    return dict(delta_1=float(delta[0]), delta_2=float(delta[1]),
                gamma_11=float(gamma[0, 0]), gamma_22=float(gamma[1, 1]),
                gamma_12=float(gamma[0, 1]), vega_1=float(vega[0]), vega_2=float(vega[1]),
                theta=float(contract.rate * price - pm * drift - 0.5 * pmm * v / contract.tau),
                rho=float(contract.tau * (pm - price)),
                correlation=float(pmm * contract.tau * w[0] * w[1] * s[0] * s[1]))
