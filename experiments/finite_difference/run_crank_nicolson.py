import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.crank_nicolson import crank_nicolson_grid, crank_nicolson_price, crank_nicolson_values


ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    settings = json.loads((ROOT / "experiments" / "config" / "finite_difference_settings.json").read_text())
    table_dir = ROOT / "results" / "finite_difference" / "tables"
    figure_dir = ROOT / "results" / "finite_difference" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    cases = [{"name": case["name"], "inputs": {**settings["base"], **case["inputs"]}}
             for case in settings["cases"]]

    convergence = []
    for case in cases:
        exact = black_scholes_price(**case["inputs"])
        for size in settings["grid_sizes"]:
            start = perf_counter()
            estimate = crank_nicolson_price(
                **case["inputs"], n_space=size, n_time=size,
                rannacher_steps=settings["rannacher_steps"],
            )
            convergence.append({
                "case": case["name"], "n_space": size, "n_time": size,
                "price": estimate, "reference": exact,
                "absolute_error": abs(estimate - exact),
                "elapsed_seconds": perf_counter() - start,
            })
    write_csv(table_dir / "fd_convergence.csv", convergence)

    richardson_prices = []
    for case in cases:
        exact = black_scholes_price(**case["inputs"])
        for size in settings["grid_sizes"][:-1]:
            start = perf_counter()
            estimate = crank_nicolson_price(
                **case["inputs"], n_space=size, n_time=size,
                rannacher_steps=settings["rannacher_steps"], richardson=True,
                mesh=settings["enhanced_mesh"],
                concentration=settings["mesh_concentration"],
            )
            richardson_prices.append({
                "case": case["name"], "base_grid_size": size,
                "fine_grid_size": 2 * size, "price": estimate,
                "reference": exact, "absolute_error": abs(estimate - exact),
                "elapsed_seconds": perf_counter() - start,
            })
    write_csv(table_dir / "fd_richardson_convergence.csv", richardson_prices)

    richardson_rates = []
    for case in cases:
        rows = [row for row in richardson_prices if row["case"] == case["name"]][:3]
        x = np.log([row["fine_grid_size"] for row in rows])
        y = np.log([max(row["absolute_error"], 1e-15) for row in rows])
        richardson_rates.append({"case": case["name"], "alpha": float(-np.polyfit(x, y, 1)[0])})
    richardson_rates.append({
        "case": "median",
        "alpha": float(np.median([row["alpha"] for row in richardson_rates])),
    })
    write_csv(table_dir / "fd_richardson_convergence_rates.csv", richardson_rates)

    rates = []
    for case in cases:
        rows = [row for row in convergence if row["case"] == case["name"]]
        x = np.log([row["n_space"] for row in rows])
        y = np.log([max(row["absolute_error"], 1e-15) for row in rows])
        rates.append({"case": case["name"], "alpha": float(-np.polyfit(x, y, 1)[0])})
    rates.append({"case": "median", "alpha": float(np.median([row["alpha"] for row in rates]))})
    write_csv(table_dir / "fd_convergence_rates.csv", rates)

    greek_rows = []
    shape_rows = []
    size = settings["greek_grid_size"]
    for case in cases:
        exact = {"price": black_scholes_price(**case["inputs"]),
                 **black_scholes_greeks(**case["inputs"])}
        start = perf_counter()
        values = crank_nicolson_values(
            **case["inputs"], n_space=size, n_time=size,
            rannacher_steps=settings["rannacher_steps"],
            volatility_bump=settings["volatility_bump"], rate_bump=settings["rate_bump"],
        )
        elapsed = perf_counter() - start
        for quantity, value in values.items():
            greek_rows.append({
                "case": case["name"], "quantity": quantity,
                "estimate": value, "reference": exact[quantity],
                "error": value - exact[quantity], "absolute_error": abs(value - exact[quantity]),
                "elapsed_seconds_all_outputs": elapsed,
            })
        inputs = case["inputs"]
        grid = crank_nicolson_grid(
            inputs["K"], inputs["tau"], inputs["r"], inputs["sigma"], inputs["q"], inputs["option"],
            n_space=size, n_time=size,
            rannacher_steps=settings["rannacher_steps"],
        )
        first = np.diff(grid.values)
        second = np.diff(grid.values, n=2)
        tolerance = settings["shape_tolerance"]
        monotonic_violations = np.sum(first < -tolerance) if inputs["option"] == "call" else np.sum(first > tolerance)
        shape_rows.append({
            "case": case["name"], "minimum_value": float(grid.values.min()),
            "minimum_second_difference": float(second.min()),
            "negative_value_count": int(np.sum(grid.values < -tolerance)),
            "monotonicity_violation_count": int(monotonic_violations),
            "convexity_violation_count": int(np.sum(second < -tolerance)),
        })
    write_csv(table_dir / "fd_greeks.csv", greek_rows)
    write_csv(table_dir / "fd_shape_checks.csv", shape_rows)

    richardson_rows = []
    richardson_size = settings["richardson_base_grid_size"]
    for case in cases:
        exact = {"price": black_scholes_price(**case["inputs"]),
                 **black_scholes_greeks(**case["inputs"])}
        start = perf_counter()
        values = crank_nicolson_values(
            **case["inputs"], n_space=richardson_size, n_time=richardson_size,
            rannacher_steps=settings["rannacher_steps"],
            volatility_bump=settings["volatility_bump"], rate_bump=settings["rate_bump"],
            richardson=True, mesh=settings["enhanced_mesh"],
            concentration=settings["mesh_concentration"],
        )
        elapsed = perf_counter() - start
        for quantity, value in values.items():
            richardson_rows.append({
                "case": case["name"], "quantity": quantity,
                "estimate": value, "reference": exact[quantity],
                "error": value - exact[quantity], "absolute_error": abs(value - exact[quantity]),
                "elapsed_seconds_all_outputs": elapsed,
                "base_grid_size": richardson_size, "fine_grid_size": 2 * richardson_size,
            })
    write_csv(table_dir / "fd_richardson_greeks.csv", richardson_rows)

    enhanced_shape_rows = []
    for case in cases:
        inputs = case["inputs"]
        grid = crank_nicolson_grid(
            inputs["K"], inputs["tau"], inputs["r"], inputs["sigma"],
            inputs["q"], inputs["option"], n_space=2 * richardson_size,
            n_time=2 * richardson_size, rannacher_steps=settings["rannacher_steps"],
            mesh=settings["enhanced_mesh"], concentration=settings["mesh_concentration"],
        )
        first = np.diff(grid.values)
        second_slopes = np.diff(grid.values) / np.diff(grid.spots)
        tolerance = settings["shape_tolerance"]
        monotonic_violations = (np.sum(first < -tolerance) if inputs["option"] == "call"
                                   else np.sum(first > tolerance))
        enhanced_shape_rows.append({
            "case": case["name"], "minimum_value": float(grid.values.min()),
            "minimum_slope_change": float(np.diff(second_slopes).min()),
            "negative_value_count": int(np.sum(grid.values < -tolerance)),
            "monotonicity_violation_count": int(monotonic_violations),
            "convexity_violation_count": int(np.sum(np.diff(second_slopes) < -tolerance)),
        })
    write_csv(table_dir / "fd_enhanced_shape_checks.csv", enhanced_shape_rows)

    boundary_case = next(case for case in cases if case["name"] == "high_vol_put")
    inputs = boundary_case["inputs"]
    boundary_size = settings["boundary_base_grid_size"]
    automatic_grid = crank_nicolson_grid(
        inputs["K"], inputs["tau"], inputs["r"], inputs["sigma"],
        inputs["q"], inputs["option"], n_space=boundary_size, n_time=boundary_size,
        mesh=settings["enhanced_mesh"], concentration=settings["mesh_concentration"],
    )
    automatic_upper = float(automatic_grid.spots[-1])
    boundary_rows = []
    exact = black_scholes_price(**inputs)
    for multiplier in settings["boundary_multipliers"]:
        upper = automatic_upper * multiplier
        start = perf_counter()
        estimate = crank_nicolson_price(
            **inputs, s_max=upper, n_space=boundary_size, n_time=boundary_size,
            rannacher_steps=settings["rannacher_steps"], richardson=True,
            mesh=settings["enhanced_mesh"], concentration=settings["mesh_concentration"],
        )
        boundary_rows.append({
            "case": boundary_case["name"], "s_max_multiplier": multiplier,
            "s_max": upper, "price": estimate, "reference": exact,
            "absolute_error": abs(estimate - exact),
            "elapsed_seconds": perf_counter() - start,
        })
    write_csv(table_dir / "fd_boundary_sensitivity.csv", boundary_rows)

    limits = settings["error_limits"]
    failures = [
        {"case": row["case"], "quantity": row["quantity"],
         "absolute_error": row["absolute_error"], "limit": limits[row["quantity"]]}
        for row in greek_rows if row["absolute_error"] > limits[row["quantity"]]
    ]
    shape_failures = [row for row in shape_rows if row["negative_value_count"]
                      or row["monotonicity_violation_count"] or row["convexity_violation_count"]]
    enhanced_shape_failures = [
        row for row in enhanced_shape_rows if row["negative_value_count"]
        or row["monotonicity_violation_count"] or row["convexity_violation_count"]
    ]
    summary = {
        "passed_error_limits": not failures,
        "passed_shape_checks": not shape_failures,
        "error_failures": failures,
        "shape_failures": shape_failures,
        "median_convergence_alpha": rates[-1]["alpha"],
        "maximum_errors": {name: max(row["absolute_error"] for row in greek_rows if row["quantity"] == name)
                           for name in ("price", "delta", "gamma", "vega", "theta", "rho")},
        "mean_runtime_all_outputs": float(np.mean([row["elapsed_seconds_all_outputs"]
                                                    for row in greek_rows if row["quantity"] == "price"])),
        "richardson": {
            "base_grid_size": richardson_size,
            "fine_grid_size": 2 * richardson_size,
            "mesh": settings["enhanced_mesh"],
            "mesh_concentration": settings["mesh_concentration"],
            "median_convergence_alpha": richardson_rates[-1]["alpha"],
            "maximum_errors": {
                name: max(row["absolute_error"] for row in richardson_rows if row["quantity"] == name)
                for name in ("price", "delta", "gamma", "vega", "theta", "rho")
            },
            "mean_runtime_all_outputs": float(np.mean([
                row["elapsed_seconds_all_outputs"] for row in richardson_rows
                if row["quantity"] == "price"
            ])),
            "passed_shape_checks": not enhanced_shape_failures,
            "shape_failures": enhanced_shape_failures,
        },
        "boundary_sensitivity": {
            "case": boundary_case["name"],
            "automatic_s_max": automatic_upper,
            "multipliers": settings["boundary_multipliers"],
            "maximum_absolute_error": max(row["absolute_error"] for row in boundary_rows),
            "price_range": max(row["price"] for row in boundary_rows)
                           - min(row["price"] for row in boundary_rows),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    (table_dir / "fd_validation.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for case in cases:
        rows = [row for row in convergence if row["case"] == case["name"]]
        axes[0].loglog([row["n_space"] for row in rows],
                       [row["absolute_error"] for row in rows], "o-", label=case["name"])
    axes[0].set_xlabel("Space and time steps")
    axes[0].set_ylabel("Absolute price error")
    axes[0].set_title("Crank--Nicolson convergence")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    quantities = ("price", "delta", "gamma", "vega", "theta", "rho")
    maximums = [summary["maximum_errors"][name] for name in quantities]
    axes[1].bar(quantities, maximums, color="#2878B5")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Maximum absolute error")
    axes[1].set_title(f"Price and Greeks on {size} x {size} grid")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "crank_nicolson_validation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for case in cases:
        standard = [row for row in convergence if row["case"] == case["name"]][1:]
        enhanced = [row for row in richardson_prices if row["case"] == case["name"]]
        ax.loglog(
            [row["n_space"] for row in standard],
            [row["absolute_error"] for row in standard], "o--", alpha=0.45,
        )
        ax.loglog(
            [row["fine_grid_size"] for row in enhanced],
            [row["absolute_error"] for row in enhanced], "o-", label=case["name"],
        )
    ax.set_xlabel("Finest space and time grid")
    ax.set_ylabel("Absolute price error")
    ax.set_title("Richardson-extrapolated Crank--Nicolson")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "crank_nicolson_richardson.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(
        [row["s_max"] for row in boundary_rows],
        [row["absolute_error"] for row in boundary_rows], "o-", color="#C44E52",
    )
    ax.set_xlabel("Upper stock-price boundary")
    ax.set_ylabel("Absolute price error")
    ax.set_title("High-volatility boundary sensitivity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "crank_nicolson_boundary_sensitivity.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
