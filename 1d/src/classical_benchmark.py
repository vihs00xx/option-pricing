import numpy as np

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.common_benchmark import QUANTITIES, inputs_from_case
from src.crank_nicolson import crank_nicolson_values
from src.monte_carlo_greeks import monte_carlo_greeks


METHODS = ("black_scholes", "conditional_rqmc", "crank_nicolson")


def evaluate_case(method, case, settings, case_index=0):
    inputs = inputs_from_case(case)
    if method == "black_scholes":
        values = {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}
        return values, {name: 0.0 for name in QUANTITIES}
    if method == "conditional_rqmc":
        config = settings["monte_carlo"]
        production = config["n_scrambles"] * config["production_points_per_scramble"]
        pilot = config["n_scrambles"] * config["pilot_points_per_scramble"]
        result = monte_carlo_greeks(
            **inputs, method=config["method"], estimator=config["estimator"],
            n_samples=production + pilot, pilot_samples=pilot,
            n_scrambles=config["n_scrambles"],
            conditional_fraction=config["conditional_fraction"],
            seed=config["seed_root"] + case_index,
        )
        return result.values, result.standard_errors
    if method == "crank_nicolson":
        config = settings["crank_nicolson"]
        values = crank_nicolson_values(
            **inputs, n_space=config["base_grid_size"], n_time=config["base_grid_size"],
            rannacher_steps=config["rannacher_steps"], mesh=config["mesh"],
            concentration=config["mesh_concentration"],
            volatility_bump=config["volatility_bump"], rate_bump=config["rate_bump"],
            richardson=config["richardson"],
        )
        return values, {name: 0.0 for name in QUANTITIES}
    raise ValueError(f"Unknown method: {method}")


def evaluate_batch(method, cases, settings):
    if method != "black_scholes":
        return [evaluate_case(method, case, settings, index)[0]
                for index, case in enumerate(cases)]
    output = [None] * len(cases)
    for option in ("call", "put"):
        selected = [(index, case) for index, case in enumerate(cases) if case["option"] == option]
        if not selected:
            continue
        arrays = {
            name: np.array([float(case[name]) for _, case in selected])
            for name in ("S", "K", "tau", "r", "sigma", "q")
        }
        prices = black_scholes_price(**arrays, option=option)
        greeks = black_scholes_greeks(**arrays, option=option)
        for local_index, (original_index, _) in enumerate(selected):
            output[original_index] = {
                "price": float(prices[local_index]),
                **{name: float(greeks[name][local_index]) for name in greeks},
            }
    return output
