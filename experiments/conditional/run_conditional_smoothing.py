import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.monte_carlo_greeks import NAMES, monte_carlo_greeks


def main():
    settings = json.loads((ROOT / "experiments" / "config" / "classical" / "benchmark_settings.json").read_text())
    budget = settings["greek_budget"]
    repetitions = settings["greek_evaluation_repetitions"]
    scrambles = 32
    pilot = budget // 2
    configs = (
        ("sobol_hybrid", "sobol", "hybrid", 1024),
        ("sobol_conditional", "sobol", "conditional", 1024),
        ("sobol_control_hybrid", "sobol_control", "hybrid", pilot),
        ("sobol_control_conditional", "sobol_control", "conditional", pilot),
    )
    rows = []
    for case_index, case in enumerate(settings["evaluation_cases"]):
        inputs = {**settings["base"], **case["inputs"]}
        reference = {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}
        for repetition in range(repetitions):
            seed = settings["seed"] + 10000 * case_index + repetition
            for label, method, estimator, pilot_samples in configs:
                result = monte_carlo_greeks(
                    **inputs, method=method, estimator=estimator, n_samples=budget,
                    n_scrambles=scrambles, pilot_samples=pilot_samples, seed=seed,
                )
                for name in NAMES:
                    low, high = result.confidence_intervals[name]
                    rows.append({
                        "case": case["name"], "repetition": repetition, "seed": seed,
                        "configuration": label, "quantity": name,
                        "estimate": result.values[name], "reference": reference[name],
                        "error": result.values[name] - reference[name],
                        "standard_error": result.standard_errors[name],
                        "covered": int(low <= reference[name] <= high),
                        "elapsed_seconds": result.elapsed_seconds,
                    })

    table_dir = ROOT / "results" / "conditional" / "tables"
    figure_dir = ROOT / "results" / "conditional" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    run_path = table_dir / "mc_conditional_greek_runs.csv"
    with run_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for label, _, _, _ in configs:
        for name in NAMES:
            selected = [row for row in rows if row["configuration"] == label and row["quantity"] == name]
            errors = np.array([row["error"] for row in selected])
            standard_errors = np.array([row["standard_error"] for row in selected])
            summary.append({
                "configuration": label,
                "quantity": name,
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mean_absolute_error": float(np.mean(np.abs(errors))),
                "bias": float(np.mean(errors)),
                "coverage": float(np.mean([row["covered"] for row in selected])),
                "zero_standard_error_rate": float(np.mean(standard_errors == 0)),
                "empirical_sd_to_mean_se": float(np.std(errors, ddof=1) / np.mean(standard_errors)),
                "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in selected])),
            })
    summary_path = table_dir / "mc_conditional_greek_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)

    metadata = {
        "budget_per_run": budget,
        "repetitions_per_case": repetitions,
        "evaluation_cases": len(settings["evaluation_cases"]),
        "sobol_scrambles": scrambles,
        "conditional_fraction": 0.5,
        "control_pilot_samples": pilot,
        "seed_design": "The same seed is used across configurations for each case and repetition.",
        "interpretation": "Conditional smoothing analytically values the second half of the option life.",
    }
    (table_dir / "mc_conditional_greek_metadata.json").write_text(json.dumps(metadata, indent=2))

    hybrid = {row["quantity"]: row["rmse"] for row in summary if row["configuration"] == "sobol_control_hybrid"}
    conditional = {row["quantity"]: row["rmse"] for row in summary
                   if row["configuration"] == "sobol_control_conditional"}
    ratios = [hybrid[name] / conditional[name] for name in NAMES]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(NAMES, ratios, color="#2878B5")
    ax.axhline(1, color="black", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE improvement factor")
    ax.set_title("Conditional smoothing versus the previous Sobol-control estimator")
    ax.bar_label(bars, labels=[f"{value:.1f}x" for value in ratios], padding=3)
    fig.tight_layout()
    fig.savefig(figure_dir / "mc_conditional_greek_improvement.png", dpi=180)
    plt.close(fig)

    print(json.dumps({name: round(hybrid[name] / conditional[name], 3) for name in NAMES}, indent=2))


if __name__ == "__main__":
    main()
