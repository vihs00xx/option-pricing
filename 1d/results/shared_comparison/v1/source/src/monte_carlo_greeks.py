from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.monte_carlo import validate_inputs
from src.sampling import estimate, validate_sampling


NAMES = ("price", "delta", "gamma", "vega", "theta", "rho")


@dataclass(frozen=True)
class MonteCarloGreeksResult:
    values: dict
    standard_errors: dict
    confidence_intervals: dict
    method: str
    estimator: str
    sample_evaluations: int
    payoff_evaluations: int
    independent_observations: int
    pilot_samples: int
    beta: dict | None
    elapsed_seconds: float


def monte_carlo_greeks(S, K, tau, r, sigma, q=0.0, option="call", *,
                       n_samples=16384, method="sobol", seed=42,
                       batch_size=65536, pilot_samples=1024, n_scrambles=8,
                       estimator="hybrid", relative_bump=0.005, rate_bump=0.0001,
                       conditional_fraction=0.5):
    """Hybrid uses pathwise first derivatives and mixed pathwise/score Gamma.
    Finite differences use common draws; their intervals exclude bump bias.
    Vega/rho are per unit decimal input; theta is calendar-time change per year.
    """
    start = perf_counter()
    validate_inputs(S, K, tau, r, sigma, q, option)
    validate_sampling(n_samples, method, batch_size, pilot_samples, n_scrambles)
    if min(S, tau, sigma) <= 0:
        raise ValueError("Greeks require positive S, tau, and sigma")
    if estimator not in ("hybrid", "likelihood_ratio", "finite_difference", "conditional"):
        raise ValueError("estimator must be hybrid, likelihood_ratio, finite_difference, or conditional")
    if not np.isfinite(relative_bump) or not 0 < relative_bump < 1 or not np.isfinite(rate_bump) or rate_bump <= 0:
        raise ValueError("Require 0 < relative_bump < 1 and a positive finite rate_bump")
    if not np.isfinite(conditional_fraction) or not 0 < conditional_fraction < 1:
        raise ValueError("conditional_fraction must be finite and between zero and one")
    sign = 1 if option == "call" else -1
    control_spot = S * np.exp(-q * tau)

    def terminal_payoffs(z, spot=S, vol=sigma, rate=r, maturity=tau):
        terminal = spot * np.exp((rate - q - 0.5 * vol**2) * maturity + vol * np.sqrt(maturity) * z)
        discounted = np.exp(-rate * maturity) * terminal
        y = np.exp(-rate * maturity) * np.maximum(sign * (terminal - K), 0)
        return discounted, y, terminal

    def evaluate(z):
        if estimator == "conditional":
            first_tau = conditional_fraction * tau
            remaining_tau = tau - first_tau
            first_asset = S * np.exp((r - q - 0.5 * sigma**2) * first_tau
                                     + sigma * np.sqrt(first_tau) * z)
            first_discount = np.exp(-r * first_tau)
            remaining_price = black_scholes_price(first_asset, K, remaining_tau, r, sigma, q, option)
            remaining_greeks = black_scholes_greeks(first_asset, K, remaining_tau, r, sigma, q, option)
            ratio = first_asset / S
            d_asset_d_sigma = first_asset * (np.sqrt(first_tau) * z - sigma * first_tau)
            d_asset_d_rate = first_tau * first_asset
            d_asset_d_tau = first_asset * (conditional_fraction * (r - q - 0.5 * sigma**2)
                                            + sigma * z * np.sqrt(conditional_fraction) / (2 * np.sqrt(tau)))
            conditional_price = first_discount * remaining_price
            delta = first_discount * remaining_greeks["delta"] * ratio
            gamma = first_discount * remaining_greeks["gamma"] * ratio**2
            vega = first_discount * (remaining_greeks["delta"] * d_asset_d_sigma
                                     + remaining_greeks["vega"])
            theta = first_discount * (r * conditional_fraction * remaining_price
                                      - remaining_greeks["delta"] * d_asset_d_tau
                                      + (1 - conditional_fraction) * remaining_greeks["theta"])
            rho = first_discount * (-first_tau * remaining_price
                                    + remaining_greeks["delta"] * d_asset_d_rate
                                    + remaining_greeks["rho"])
            conditional_stock = first_discount * first_asset
            delta_control = conditional_stock / S
            gamma_control = conditional_stock / S**2 * (z / (sigma * np.sqrt(first_tau)) - 1)
            vega_control = conditional_stock * (np.sqrt(first_tau) * z - sigma * first_tau)
            time_motion = conditional_fraction * (r - q - 0.5 * sigma**2) \
                + sigma * z * np.sqrt(conditional_fraction) / (2 * np.sqrt(tau))
            theta_control = r * conditional_fraction * conditional_stock - conditional_stock * time_motion
            x = np.column_stack((conditional_stock, delta_control, gamma_control,
                                 vega_control, theta_control, conditional_stock))
            y = np.column_stack((conditional_price, delta, gamma, vega, theta, rho))
            return x, y
        stock, price, terminal = terminal_payoffs(z)
        if estimator == "likelihood_ratio":
            spread = sigma * np.sqrt(tau)
            scores = np.column_stack((np.ones_like(z), z / (S * spread),
                                      (z**2 - 1 - spread * z) / (S**2 * spread**2),
                                      (z**2 - 1) / sigma - np.sqrt(tau) * z,
                                      r - (r - q - 0.5 * sigma**2) * z / spread - (z**2 - 1) / (2 * tau),
                                      -tau + np.sqrt(tau) * z / sigma))
            return stock[:, None] * scores, price[:, None] * scores
        if estimator == "hybrid":
            exercise = (sign * (terminal - K) > 0)
            delta_control = stock / S
            gamma_control = stock / S**2 * (z / (sigma * np.sqrt(tau)) - 1)
            vega_control = stock * (np.sqrt(tau) * z - sigma * tau)
            time_motion = (r - q - 0.5 * sigma**2) + sigma * z / (2 * np.sqrt(tau))
            theta_control = r * stock - stock * time_motion
            x = np.column_stack((stock, delta_control, gamma_control, vega_control, theta_control, stock))
            y = np.column_stack((price, sign * exercise * delta_control,
                                 sign * exercise * gamma_control, sign * exercise * vega_control,
                                 r * price - sign * exercise * stock * time_motion,
                                 sign * K * tau * np.exp(-r * tau) * exercise))
            return x, y

        hS, hv, ht = relative_bump * S, relative_bump * sigma, relative_bump * tau
        base = np.column_stack((stock, price))
        def pair(**kwargs):
            x, y, _ = terminal_payoffs(z, **kwargs)
            return np.column_stack((x, y))
        plus, minus = pair(spot=S + hS), pair(spot=S - hS)
        delta = (plus - minus) / (2 * hS)
        gamma = (plus - 2 * base + minus) / hS**2
        vega = (pair(vol=sigma + hv) - pair(vol=sigma - hv)) / (2 * hv)
        theta = -(pair(maturity=tau + ht) - pair(maturity=tau - ht)) / (2 * ht)
        rho = (pair(rate=r + rate_bump) - pair(rate=r - rate_bump)) / (2 * rate_bump)
        return np.column_stack((stock, delta[:, 0], stock, vega[:, 0], theta[:, 0], stock)), np.column_stack(
            (price, delta[:, 1], gamma[:, 1], vega[:, 1], theta[:, 1], rho[:, 1]))

    known = np.array([control_spot, control_spot / S, 0, 0, q * control_spot, control_spot])
    if estimator == "conditional":
        first_tau = conditional_fraction * tau
        conditional_spot = S * np.exp(-q * first_tau)
        known = np.array([conditional_spot, conditional_spot / S, 0, 0,
                          conditional_fraction * q * conditional_spot, conditional_spot])
    if estimator == "likelihood_ratio":
        known[5] = 0
    if estimator == "finite_difference":
        ht = relative_bump * tau
        known[2] = control_spot
        known[4] = -(S * np.exp(-q * (tau + ht)) - S * np.exp(-q * (tau - ht))) / (2 * ht)
    result = estimate(evaluate, known, n_samples, method, seed, batch_size, pilot_samples, n_scrambles)
    return MonteCarloGreeksResult(
        dict(zip(NAMES, map(float, result.mean))), dict(zip(NAMES, map(float, result.se))),
        {name: (float(mean - result.critical * se), float(mean + result.critical * se))
         for name, mean, se in zip(NAMES, result.mean, result.se)},
        method, estimator, int(n_samples), int(n_samples) * (9 if estimator == "finite_difference" else 1),
        result.observations, result.pilot_samples,
        dict(zip(NAMES, map(float, result.beta))) if result.beta is not None else None,
        perf_counter() - start,
    )
