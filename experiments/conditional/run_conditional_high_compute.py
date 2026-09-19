import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.monte_carlo_greeks import NAMES, monte_carlo_greeks


def wilson(successes, count, z=1.959963984540054):
    proportion = successes / count
    denominator = 1 + z**2 / count
    centre = (proportion + z**2 / (2 * count)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / count + z**2 / (4 * count**2)) / denominator
    return float(centre - radius), float(centre + radius)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    settings_path = ROOT / "experiments" / "config" / "conditional_high_compute_settings.json"
    settings_bytes = settings_path.read_bytes()
    settings = json.loads(settings_bytes)
    table_dir = ROOT / "results" / "conditional" / "tables"
    figure_dir = ROOT / "results" / "conditional" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    runs_path = table_dir / "mc_conditional_high_compute_runs.csv"
    if runs_path.exists():
        raise FileExistsError(f"Refusing to replace sealed evaluation: {runs_path}")

    rows = []
    repetitions = settings["evaluation"]["repetitions"]
    for case_index, case in enumerate(settings["final_cases"]):
        inputs = case["inputs"]
        exact = {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}
        for repetition in range(repetitions):
            seed = settings["evaluation"]["seed_root"] + 10000 * case_index + repetition
            for config in settings["configurations"]:
                scrambles = config["n_scrambles"]
                production = scrambles * settings["points_per_scramble"]
                pilot = scrambles * settings["pilot_points_per_scramble"]
                result = monte_carlo_greeks(
                    **inputs, estimator="conditional", method="sobol_control",
                    n_samples=production + pilot, pilot_samples=pilot,
                    n_scrambles=scrambles,
                    conditional_fraction=settings["conditional_fraction"], seed=seed,
                )
                for quantity in NAMES:
                    low, high = result.confidence_intervals[quantity]
                    rows.append({
                        "case": case["name"], "configuration": config["name"],
                        "repetition": repetition, "seed": seed, "quantity": quantity,
                        "n_scrambles": scrambles, "production_points": production,
                        "pilot_points": pilot, "total_points": production + pilot,
                        "estimate": result.values[quantity], "reference": exact[quantity],
                        "error": result.values[quantity] - exact[quantity],
                        "standard_error": result.standard_errors[quantity],
                        "covered": int(low <= exact[quantity] <= high),
                        "elapsed_seconds": result.elapsed_seconds,
                    })
    write_csv(runs_path, rows)

    summaries = []
    aggregates = []
    for config in settings["configurations"]:
        for quantity in NAMES:
            quantity_rows = [row for row in rows if row["configuration"] == config["name"]
                             and row["quantity"] == quantity]
            case_rmses = []
            for case in settings["final_cases"]:
                selected = [row for row in quantity_rows if row["case"] == case["name"]]
                errors = np.array([row["error"] for row in selected])
                standard_errors = np.array([row["standard_error"] for row in selected])
                successes = sum(row["covered"] for row in selected)
                low, high = wilson(successes, len(selected))
                rmse = float(np.sqrt(np.mean(errors**2)))
                case_rmses.append(rmse)
                summaries.append({
                    "configuration": config["name"], "case": case["name"], "quantity": quantity,
                    "rmse": rmse, "bias": float(np.mean(errors)),
                    "coverage": successes / len(selected),
                    "coverage_wilson_low": low, "coverage_wilson_high": high,
                    "zero_standard_error_rate": float(np.mean(standard_errors == 0)),
                    "empirical_sd_to_mean_se": float(np.std(errors, ddof=1) / np.mean(standard_errors)),
                    "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in selected])),
                })
            errors = np.array([row["error"] for row in quantity_rows])
            aggregates.append({
                "configuration": config["name"], "quantity": quantity,
                "pooled_rmse": float(np.sqrt(np.mean(errors**2))),
                "geometric_case_rmse": float(np.exp(np.mean(np.log(np.maximum(case_rmses, 1e-300))))),
                "coverage": float(np.mean([row["covered"] for row in quantity_rows])),
                "zero_standard_error_rate": float(np.mean([row["standard_error"] == 0 for row in quantity_rows])),
                "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in quantity_rows])),
            })
    write_csv(table_dir / "mc_conditional_high_compute_summary.csv", summaries)
    write_csv(table_dir / "mc_conditional_high_compute_aggregate.csv", aggregates)

    efficient = {row["quantity"]: row for row in aggregates if row["configuration"] == "efficient_r8"}
    high = {row["quantity"]: row for row in aggregates if row["configuration"] == "high_compute_r64"}
    comparison = []
    for quantity in NAMES:
        comparison.append({
            "quantity": quantity,
            "rmse_improvement_factor": efficient[quantity]["geometric_case_rmse"] / high[quantity]["geometric_case_rmse"],
            "runtime_factor": high[quantity]["mean_elapsed_seconds"] / efficient[quantity]["mean_elapsed_seconds"],
            "equal_time_efficiency_factor":
                (efficient[quantity]["geometric_case_rmse"] / high[quantity]["geometric_case_rmse"])
                / np.sqrt(high[quantity]["mean_elapsed_seconds"] / efficient[quantity]["mean_elapsed_seconds"]),
            "efficient_coverage": efficient[quantity]["coverage"],
            "high_compute_coverage": high[quantity]["coverage"],
        })
    write_csv(table_dir / "mc_conditional_high_compute_comparison.csv", comparison)

    threshold = settings["evaluation"]["minimum_case_coverage_screen"]
    validation = {}
    for config in settings["configurations"]:
        selected = [row for row in summaries if row["configuration"] == config["name"]]
        failures = [
            {"case": row["case"], "quantity": row["quantity"], "coverage": row["coverage"],
             "zero_standard_error_rate": row["zero_standard_error_rate"]}
            for row in selected
            if row["coverage"] < threshold or row["zero_standard_error_rate"] > 0
        ]
        validation[config["name"]] = {
            "passed_pre_registered_screen": not failures,
            "minimum_observed_coverage": min(row["coverage"] for row in selected),
            "maximum_zero_standard_error_rate": max(row["zero_standard_error_rate"] for row in selected),
            "failures": failures,
        }
    (table_dir / "mc_conditional_high_compute_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    metadata = {
        "settings_sha256": hashlib.sha256(settings_bytes).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "cases": len(settings["final_cases"]), "repetitions_per_case": repetitions,
    }
    (table_dir / "mc_conditional_high_compute_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    improvement = [row["rmse_improvement_factor"] for row in comparison]
    bars = axes[0].bar(NAMES, improvement, color="#2878B5")
    axes[0].axhline(1, color="black", linewidth=0.8)
    axes[0].set_ylabel("RMSE improvement factor")
    axes[0].set_title("64 versus 8 scrambles")
    axes[0].bar_label(bars, labels=[f"{value:.2f}x" for value in improvement], padding=3)
    x = np.arange(len(NAMES))
    width = 0.36
    axes[1].bar(x - width / 2, [row["efficient_coverage"] for row in comparison], width, label="8 scrambles")
    axes[1].bar(x + width / 2, [row["high_compute_coverage"] for row in comparison], width, label="64 scrambles")
    axes[1].axhline(settings["evaluation"]["nominal_coverage"], color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x, NAMES)
    axes[1].set_ylim(0.84, 1.0)
    axes[1].set_ylabel("Empirical coverage")
    axes[1].set_title("95% interval coverage")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "mc_conditional_high_compute.png", dpi=180)
    plt.close(fig)
    print(json.dumps(validation, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
