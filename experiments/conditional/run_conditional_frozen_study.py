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


def reference(inputs):
    return {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}


def wilson(successes, count, z=1.959963984540054):
    proportion = successes / count
    denominator = 1 + z**2 / count
    centre = (proportion + z**2 / (2 * count)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / count + z**2 / (4 * count**2)) / denominator
    return float(centre - radius), float(centre + radius)


def configuration(label, selected, production):
    ratio = selected["pilot_ratio_to_production"] if label == "frozen" else 0.0
    pilot = int(production * ratio)
    return {
        "label": label,
        "method": "sobol_control" if pilot else "sobol",
        "pilot": pilot,
        "total": production + pilot,
    }


def calculate(inputs, selected, production, seed, label):
    config = configuration(label, selected, production)
    result = monte_carlo_greeks(
        **inputs, estimator="conditional", method=config["method"],
        n_samples=config["total"], pilot_samples=config["pilot"] or 1024,
        n_scrambles=selected["n_scrambles"],
        conditional_fraction=selected["conditional_fraction"], seed=seed,
    )
    return result, config


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    settings_path = ROOT / "experiments" / "config" / "conditional_study_settings.json"
    frozen_path = ROOT / "results" / "conditional" / "tables" / "mc_conditional_frozen_configuration.json"
    settings_bytes = settings_path.read_bytes()
    settings = json.loads(settings_bytes)
    frozen = json.loads(frozen_path.read_text())
    digest = hashlib.sha256(settings_bytes).hexdigest()
    if digest != frozen["settings_sha256"]:
        raise RuntimeError("Study settings changed after tuning was frozen")
    selected = frozen["selected"]
    benchmark = json.loads((ROOT / "experiments" / "config" / "benchmark_settings.json").read_text())
    base = benchmark["base"]
    tuning_cases = {case["name"]: {"name": case["name"], "inputs": {**base, **case["inputs"]}}
                    for case in benchmark["tuning_cases"]}
    table_dir = ROOT / "results" / "conditional" / "tables"
    figure_dir = ROOT / "results" / "conditional" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    final_runs_path = table_dir / "mc_conditional_final_runs.csv"
    if final_runs_path.exists():
        raise FileExistsError(f"Refusing to replace final validation: {final_runs_path}")

    convergence_rows = []
    convergence = settings["convergence"]
    labels = ["no_control"] if selected["pilot_ratio_to_production"] == 0 else ["no_control", "frozen"]
    for case_index, name in enumerate(convergence["case_names"]):
        case = tuning_cases[name]
        exact = reference(case["inputs"])
        for budget in convergence["production_points"]:
            for repetition in range(convergence["repetitions"]):
                seed = convergence["seed_root"] + 100000 * case_index + 100 * int(np.log2(budget)) + repetition
                for label in labels:
                    result, config = calculate(case["inputs"], selected, budget, seed, label)
                    for quantity in NAMES:
                        low, high = result.confidence_intervals[quantity]
                        convergence_rows.append({
                            "case": name, "configuration": label, "production_points": budget,
                            "pilot_points": config["pilot"], "total_points": config["total"],
                            "repetition": repetition, "seed": seed, "quantity": quantity,
                            "estimate": result.values[quantity], "reference": exact[quantity],
                            "error": result.values[quantity] - exact[quantity],
                            "standard_error": result.standard_errors[quantity],
                            "covered": int(low <= exact[quantity] <= high),
                            "elapsed_seconds": result.elapsed_seconds,
                        })
    write_csv(table_dir / "mc_conditional_convergence_runs.csv", convergence_rows)

    convergence_summary = []
    case_rmse = {}
    for label in labels:
        for quantity in NAMES:
            for budget in convergence["production_points"]:
                selected_rows = [row for row in convergence_rows if row["configuration"] == label
                                 and row["quantity"] == quantity and row["production_points"] == budget]
                errors = np.array([row["error"] for row in selected_rows])
                per_case = []
                for name in convergence["case_names"]:
                    case_errors = np.array([row["error"] for row in selected_rows if row["case"] == name])
                    value = float(np.sqrt(np.mean(case_errors**2)))
                    case_rmse[label, quantity, name, budget] = value
                    per_case.append(value)
                convergence_summary.append({
                    "configuration": label, "quantity": quantity, "production_points": budget,
                    "total_points": selected_rows[0]["total_points"],
                    "pooled_rmse": float(np.sqrt(np.mean(errors**2))),
                    "geometric_case_rmse": float(np.exp(np.mean(np.log(np.maximum(per_case, 1e-300))))),
                    "coverage": float(np.mean([row["covered"] for row in selected_rows])),
                    "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in selected_rows])),
                })
    write_csv(table_dir / "mc_conditional_convergence_summary.csv", convergence_summary)

    rates = []
    for label in labels:
        for quantity in NAMES:
            case_rates = []
            for name in convergence["case_names"]:
                x = np.log(np.array(convergence["production_points"], dtype=float))
                y = np.log([max(case_rmse[label, quantity, name, budget], 1e-300)
                            for budget in convergence["production_points"]])
                alpha = float(-np.polyfit(x, y, 1)[0])
                case_rates.append(alpha)
                rates.append({"configuration": label, "quantity": quantity, "case": name, "alpha": alpha})
            rates.append({"configuration": label, "quantity": quantity, "case": "median",
                          "alpha": float(np.median(case_rates))})
    write_csv(table_dir / "mc_conditional_convergence_rates.csv", rates)

    final_rows = []
    final_settings = settings["final_validation"]
    for case_index, case in enumerate(settings["final_cases"]):
        exact = reference(case["inputs"])
        for repetition in range(final_settings["repetitions"]):
            seed = final_settings["seed_root"] + 10000 * case_index + repetition
            result, config = calculate(case["inputs"], selected,
                                       final_settings["production_points"], seed, "frozen")
            for quantity in NAMES:
                low, high = result.confidence_intervals[quantity]
                final_rows.append({
                    "case": case["name"], "repetition": repetition, "seed": seed,
                    "quantity": quantity, "production_points": final_settings["production_points"],
                    "pilot_points": config["pilot"], "total_points": config["total"],
                    "estimate": result.values[quantity], "reference": exact[quantity],
                    "error": result.values[quantity] - exact[quantity],
                    "standard_error": result.standard_errors[quantity],
                    "covered": int(low <= exact[quantity] <= high),
                    "elapsed_seconds": result.elapsed_seconds,
                })
    write_csv(final_runs_path, final_rows)

    final_summary = []
    failures = []
    threshold = final_settings["minimum_case_coverage"]
    for case in settings["final_cases"]:
        for quantity in NAMES:
            selected_rows = [row for row in final_rows if row["case"] == case["name"]
                             and row["quantity"] == quantity]
            errors = np.array([row["error"] for row in selected_rows])
            standard_errors = np.array([row["standard_error"] for row in selected_rows])
            successes = sum(row["covered"] for row in selected_rows)
            low, high = wilson(successes, len(selected_rows))
            coverage = successes / len(selected_rows)
            row = {
                "case": case["name"], "quantity": quantity,
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mean_absolute_error": float(np.mean(np.abs(errors))),
                "bias": float(np.mean(errors)), "coverage": coverage,
                "coverage_wilson_low": low, "coverage_wilson_high": high,
                "zero_standard_error_rate": float(np.mean(standard_errors == 0)),
                "empirical_sd_to_mean_se": float(np.std(errors, ddof=1) / np.mean(standard_errors)),
                "mean_elapsed_seconds": float(np.mean([entry["elapsed_seconds"] for entry in selected_rows])),
            }
            final_summary.append(row)
            if coverage < threshold or row["zero_standard_error_rate"] > 0:
                failures.append({"case": case["name"], "quantity": quantity,
                                 "coverage": coverage,
                                 "zero_standard_error_rate": row["zero_standard_error_rate"]})
    write_csv(table_dir / "mc_conditional_final_summary.csv", final_summary)

    validation = {
        "passed_pre_registered_screen": not failures,
        "minimum_case_coverage_required": threshold,
        "minimum_observed_coverage": min(row["coverage"] for row in final_summary),
        "maximum_zero_standard_error_rate": max(row["zero_standard_error_rate"] for row in final_summary),
        "failures": failures,
    }
    (table_dir / "mc_conditional_final_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    metadata = {
        "settings_sha256": digest,
        "frozen_configuration_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "selected": selected, "final_cases": len(settings["final_cases"]),
        "final_repetitions_per_case": final_settings["repetitions"],
    }
    (table_dir / "mc_conditional_frozen_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, quantity in zip(axes.ravel(), NAMES):
        for label in labels:
            entries = [row for row in convergence_summary
                       if row["configuration"] == label and row["quantity"] == quantity]
            axis.loglog([row["total_points"] for row in entries],
                        [row["geometric_case_rmse"] for row in entries], "o-", label=label)
        axis.set_title(quantity.capitalize())
        axis.set_xlabel("Total samples")
        axis.set_ylabel("Geometric mean RMSE")
        axis.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "mc_conditional_convergence.png", dpi=180)
    plt.close(fig)

    print(json.dumps(validation, indent=2))
    print(json.dumps({row["quantity"]: row["alpha"] for row in rates
                      if row["configuration"] == "frozen" and row["case"] == "median"}, indent=2))


if __name__ == "__main__":
    main()
