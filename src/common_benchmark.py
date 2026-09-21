from collections import defaultdict

import numpy as np
from scipy.stats import qmc

from src.black_scholes import black_scholes_greeks, black_scholes_price


QUANTITIES = ("price", "delta", "gamma", "vega", "theta", "rho")
PARAMETERS = ("moneyness", "tau", "sigma", "r", "q")


def _case(case_id, split, reason, moneyness, tau, sigma, r, q, option, strike):
    return {
        "case_id": case_id,
        "split": split,
        "excluded_from_training": 1,
        "reason": reason,
        "S": float(strike * moneyness),
        "K": float(strike),
        "moneyness": float(moneyness),
        "tau": float(tau),
        "r": float(r),
        "sigma": float(sigma),
        "q": float(q),
        "option": option,
    }


def build_cases(settings):
    strike = settings["strike"]
    anchors = (
        ("atm_call", "standard ATM call", 1.0, 1.0, 0.2, 0.05, 0.0, "call"),
        ("atm_put", "standard ATM put", 1.0, 1.0, 0.2, 0.05, 0.0, "put"),
        ("itm_call", "in-the-money call", 1.2, 0.5, 0.25, 0.03, 0.01, "call"),
        ("otm_call", "out-of-the-money call", 0.8, 0.5, 0.25, 0.03, 0.01, "call"),
        ("itm_put", "in-the-money put", 0.8, 0.5, 0.25, 0.03, 0.01, "put"),
        ("otm_put", "out-of-the-money put", 1.2, 0.5, 0.25, 0.03, 0.01, "put"),
        ("short_atm_call", "short maturity payoff kink", 1.0, 0.02, 0.2, 0.03, 0.0, "call"),
        ("short_atm_put", "short maturity payoff kink", 1.0, 0.02, 0.2, 0.03, 0.0, "put"),
        ("high_vol_call", "high volatility and long maturity", 0.9, 2.0, 0.6, 0.05, 0.02, "call"),
        ("high_vol_put", "high volatility and long maturity", 1.1, 2.0, 0.6, 0.05, 0.02, "put"),
        ("low_vol_call", "low volatility", 1.03, 0.6, 0.1, 0.03, 0.01, "call"),
        ("negative_rate_put", "negative interest rate", 0.97, 1.4, 0.3, -0.02, 0.04, "put"),
        ("high_dividend_call", "high dividend yield", 1.05, 1.2, 0.3, 0.04, 0.06, "call"),
        ("high_rate_put", "high interest rate", 0.95, 1.0, 0.3, 0.1, 0.02, "put"),
        ("deep_itm_short_call", "deep ITM and short maturity", 1.3, 0.05, 0.2, 0.03, 0.0, "call"),
        ("deep_itm_short_put", "deep ITM and short maturity", 0.7, 0.05, 0.2, 0.03, 0.0, "put"),
    )
    cases = [
        _case(f"anchor_{index:02d}_{name}", "anchor", reason, m, tau, sigma, r, q, option, strike)
        for index, (name, reason, m, tau, sigma, r, q, option) in enumerate(anchors, 1)
    ]

    domain = settings["training_domain"]
    generation = settings["case_generation"]
    seed = generation["seed"]
    count = generation["interpolation_cases"]
    sample = qmc.LatinHypercube(d=len(PARAMETERS), seed=seed).random(count)
    lower = np.array([domain[name][0] for name in PARAMETERS])
    upper = np.array([domain[name][1] for name in PARAMETERS])
    sample = qmc.scale(sample, lower, upper)
    for index, values in enumerate(sample, 1):
        m, tau, sigma, r, q = values
        option = "call" if index % 2 else "put"
        cases.append(_case(
            f"interpolation_{index:02d}", "interpolation",
            "held out inside neural training domain", m, tau, sigma, r, q, option, strike,
        ))

    outside = generation["extrapolation_values"]
    count = sum(len(values) for values in outside.values())
    sample = qmc.LatinHypercube(d=len(PARAMETERS), seed=seed + 1).random(count)
    sample = qmc.scale(sample, lower, upper)
    index = 0
    for parameter in PARAMETERS:
        for value in outside[parameter]:
            values = dict(zip(PARAMETERS, sample[index]))
            values[parameter] = value
            option = "call" if index % 2 == 0 else "put"
            cases.append(_case(
                f"extrapolation_{parameter}_{index + 1:02d}", "extrapolation",
                f"{parameter} outside neural training domain", values["moneyness"],
                values["tau"], values["sigma"], values["r"], values["q"], option, strike,
            ))
            index += 1
    return cases


def inputs_from_case(case):
    return {
        "S": float(case["S"]), "K": float(case["K"]),
        "tau": float(case["tau"]), "r": float(case["r"]),
        "sigma": float(case["sigma"]), "q": float(case["q"]),
        "option": case["option"],
    }


def reference_values(case):
    inputs = inputs_from_case(case)
    return {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}


def aggregate_metrics(rows, floors):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["split"], row["quantity"])].append(row)
        grouped[(row["method"], "all", row["quantity"])].append(row)
    output = []
    for (method, split, quantity), selected in sorted(grouped.items()):
        errors = np.array([float(row["error"]) for row in selected])
        references = np.array([float(row["reference"]) for row in selected])
        eligible = np.abs(references) >= floors[quantity]
        relative = errors[eligible] / references[eligible]
        output.append({
            "method": method, "split": split, "quantity": quantity,
            "cases": len(selected), "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "maximum_absolute_error": float(np.max(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "relative_rmse": float(np.sqrt(np.mean(relative**2))) if np.any(eligible) else "",
            "relative_error_cases": int(np.sum(eligible)),
        })
    return output
