"""Shape-violation and accuracy diagnostics for Black--Scholes PINNs.

The fixed-PINN validation reported only violation counts. This module
reports, for each region of the evaluation grid, the proportion, maximum
size, and mean size of every financial shape violation, together with
price and Greek errors evaluated only where smooth reference Greeks
exist (tau > 0). Violations follow the no-arbitrage bounds of a
European call:

    max(S e^{-q tau} - K e^{-r tau}, 0) <= C <= S e^{-q tau},
    0 <= C_S <= e^{-q tau},
    C_SS >= 0.
"""

import numpy as np
import torch

from src.black_scholes import black_scholes_greeks, black_scholes_price


VIOLATION_NAMES = (
    "price_below_intrinsic",
    "price_above_discounted_spot",
    "delta_below_zero",
    "delta_above_discount_factor",
    "gamma_negative",
)

REGION_NAMES = (
    "full",
    "core",
    "near_expiry",
    "near_strike",
    "low_tail",
    "high_tail",
)


def region_masks(spots, tau, k, core_x_range, core_minimum_tau,
                 near_strike_half_width, near_expiry_tau):
    """Return boolean masks for each evaluation region.

    ``core`` matches the fixed-PINN acceptance region. ``near_expiry``
    isolates the payoff kink neighbourhood in calendar time,
    ``near_strike`` the payoff kink neighbourhood in moneyness, and the
    tails the low-price and high-price regions away from the strike.
    """
    x = np.asarray(spots, dtype=float) / k
    tau = np.asarray(tau, dtype=float)
    full = np.ones(x.shape, dtype=bool)
    core = (
        (x >= core_x_range[0]) & (x <= core_x_range[1])
        & (tau >= core_minimum_tau)
    )
    near_expiry = tau <= near_expiry_tau
    near_strike = np.abs(x - 1.0) <= near_strike_half_width
    low_tail = x < 1.0 - near_strike_half_width
    high_tail = x > 1.0 + near_strike_half_width
    return {
        "full": full,
        "core": core,
        "near_expiry": near_expiry,
        "near_strike": near_strike,
        "low_tail": low_tail,
        "high_tail": high_tail,
    }


def shape_violations(price, delta, gamma, spots, tau, k, r, q):
    """Return positive-part violation magnitudes for every constraint."""
    spots = np.asarray(spots, dtype=float)
    tau = np.asarray(tau, dtype=float)
    discount_r = np.exp(-r * tau)
    discount_q = np.exp(-q * tau)
    intrinsic = np.maximum(spots * discount_q - k * discount_r, 0.0)
    upper = spots * discount_q
    return {
        "price_below_intrinsic": np.maximum(intrinsic - price, 0.0),
        "price_above_discounted_spot": np.maximum(price - upper, 0.0),
        "delta_below_zero": np.maximum(-delta, 0.0),
        "delta_above_discount_factor": np.maximum(delta - discount_q, 0.0),
        "gamma_negative": np.maximum(-gamma, 0.0),
    }


def violation_summary(magnitudes, mask=None):
    """Proportion, maximum and mean size of one violation over a mask."""
    values = np.asarray(magnitudes, dtype=float)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    count = int(np.sum(values > 0.0))
    points = int(values.size)
    return {
        "points": points,
        "count": count,
        "proportion": count / points if points else 0.0,
        "maximum": float(np.max(values)) if points else 0.0,
        "mean_size": float(np.mean(values)) if points else 0.0,
    }


def violation_table(violations, masks):
    """Per-region summary of every violation magnitude array."""
    table = {}
    for region, mask in masks.items():
        table[region] = {
            name: violation_summary(magnitudes, mask)
            for name, magnitudes in violations.items()
        }
    return table


def prediction_errors(estimate, reference, mask=None):
    difference = np.asarray(estimate, dtype=float) - np.asarray(reference, dtype=float)
    if mask is not None:
        difference = difference[np.asarray(mask, dtype=bool)]
    if difference.size == 0:
        return {"points": 0, "rmse": None, "mae": None,
                "maximum_absolute_error": None, "bias": None}
    return {
        "points": int(difference.size),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "bias": float(np.mean(difference)),
    }


def predict_grid(model, y, tau, batch_size):
    """Batched price/delta/gamma/theta predictions for flat input arrays."""
    names = ("price", "delta", "gamma", "theta")
    output = {name: [] for name in names}
    y = np.asarray(y, dtype=float)
    tau = np.asarray(tau, dtype=float)
    for start in range(0, len(y), batch_size):
        y_tensor = torch.tensor(y[start:start + batch_size, None], requires_grad=True)
        tau_tensor = torch.tensor(tau[start:start + batch_size, None], requires_grad=True)
        values = model.price_and_greeks(y_tensor, tau_tensor)
        for name in names:
            output[name].append(values[name].detach().numpy()[:, 0])
    return {name: np.concatenate(parts) for name, parts in output.items()}


def reference_values(spots, tau, k, r, sigma, q, option="call"):
    """Analytical prices everywhere and Greeks where tau > 0 only."""
    spots = np.asarray(spots, dtype=float)
    tau = np.asarray(tau, dtype=float)
    price = black_scholes_price(spots, k, tau, r, sigma, q, option)
    greeks = {}
    smooth = tau > 0
    exact = black_scholes_greeks(
        spots[smooth], k, tau[smooth], r, sigma, q, option
    )
    for name in ("delta", "gamma", "theta"):
        column = np.full(spots.shape, np.nan)
        column[smooth] = exact[name]
        greeks[name] = column
    return {"price": price, "greeks": greeks, "smooth": smooth}


def full_diagnostics(predicted, reference, spots, tau, k, r, q, evaluation):
    """Combine region-wise error metrics and violation statistics."""
    masks = region_masks(
        spots, tau, k,
        evaluation["core_x_range"], evaluation["core_minimum_tau"],
        evaluation["near_strike_half_width"], evaluation["near_expiry_tau"],
    )
    violations = shape_violations(
        predicted["price"], predicted["delta"], predicted["gamma"],
        spots, tau, k, r, q,
    )
    smooth = reference["smooth"]
    errors = {}
    for region, mask in masks.items():
        region_errors = {
            "price": prediction_errors(predicted["price"], reference["price"], mask),
        }
        for quantity in ("delta", "gamma", "theta"):
            quantity_mask = mask & smooth
            region_errors[quantity] = prediction_errors(
                predicted[quantity], reference["greeks"][quantity], quantity_mask
            )
        errors[region] = region_errors
    return {
        "regions": masks,
        "errors": errors,
        "violations": violation_table(violations, masks),
        "violation_magnitudes": violations,
    }


def acceptance_checks(errors, acceptance):
    """Fixed-PINN acceptance thresholds applied to the core region."""
    core = errors["core"]
    return {
        "core_price_rmse": core["price"]["rmse"] <= acceptance["core_price_rmse"],
        "core_delta_rmse": core["delta"]["rmse"] <= acceptance["core_delta_rmse"],
        "core_gamma_rmse": core["gamma"]["rmse"] <= acceptance["core_gamma_rmse"],
        "core_theta_rmse": core["theta"]["rmse"] <= acceptance["core_theta_rmse"],
    }
