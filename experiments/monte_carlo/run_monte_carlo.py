import csv
import json
import platform
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from src.black_scholes import black_scholes_price
from src.monte_carlo import monte_carlo_price


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    project = Path(__file__).resolve().parents[2]
    settings = json.loads((project / "experiments" / "config" / "classical" / "monte_carlo_settings.json").read_text())
    setup = settings["monte_carlo"]
    methods = setup["methods"]
    tables = project / "results" / "monte_carlo" / "tables"
    figures = project / "results" / "monte_carlo" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    count = len(setup["cases"]) * len(setup["budgets"]) * setup["repetitions"] * len(methods)
    seeds = iter(np.random.SeedSequence(settings["seed"]).spawn(count))

    for method in methods:
        monte_carlo_price(**settings["base_case"], method=method, n_samples=2048,
                          pilot_samples=256, seed=0)

    rows = []
    for case in setup["cases"]:
        inputs = {**settings["base_case"], **case["inputs"]}
        reference = black_scholes_price(**inputs)
        for budget in setup["budgets"]:
            for repetition in range(setup["repetitions"]):
                # Rotate method order to reduce timing bias from a fixed sequence.
                order = methods[repetition % len(methods):] + methods[:repetition % len(methods)]
                for method in order:
                    seed = int(next(seeds).generate_state(1)[0])
                    result = monte_carlo_price(
                        **inputs, method=method, n_samples=budget, seed=seed,
                        batch_size=setup["batch_size"],
                        pilot_samples=max(2, budget // setup["pilot_budget_divisor"]),
                        n_scrambles=setup["n_scrambles"],
                    )
                    rows.append({
                        "case": case["name"], **inputs, "budget": budget,
                        "repetition": repetition, "seed": seed, "reference": reference,
                        **asdict(result), "error": result.price - reference,
                        "covered": int(result.ci_low <= reference <= result.ci_high),
                    })
        print(f"Completed {case['name']}: analytical price {reference:.8f}", flush=True)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["method"], row["budget"])].append(row)
    summary = []
    for (case, method, budget), group in groups.items():
        errors = np.array([row["error"] for row in group])
        summary.append({
            "case": case, "method": method, "budget": budget,
            "repetitions": len(group), "reference": group[0]["reference"],
            "mean_price": float(np.mean([row["price"] for row in group])),
            "bias_estimate": float(errors.mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "max_absolute_error": float(np.max(np.abs(errors))),
            "mean_standard_error": float(np.mean([row["standard_error"] for row in group])),
            "empirical_price_sd": float(np.std([row["price"] for row in group], ddof=1)),
            "coverage_95": float(np.mean([row["covered"] for row in group])),
            "mean_seconds": float(np.mean([row["elapsed_seconds"] for row in group])),
            "median_seconds": float(np.median([row["elapsed_seconds"] for row in group])),
        })

    save_csv(tables / "monte_carlo_runs.csv", rows)
    save_csv(tables / "monte_carlo_summary.csv", summary)
    metadata = {
        "settings": settings,
        "total_runs": len(rows),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "system": platform.platform(),
        "processor": platform.processor(),
        "timing": "Wall time per pricing call, including validation, RNG setup, pilot, payoff evaluation, and CI; excludes imports, warmup, analytical reference, and file output.",
        "budget": "Total payoff evaluations, including control pilots, both members of each antithetic pair, and all Sobol scrambles.",
        "uncertainty": "Approximate normal 95% intervals for pseudorandom estimators; Student-t intervals across independent Sobol scramble means. Coverage estimates from 20 repetitions are noisy.",
        "memory": "Not measured. Pseudorandom methods batch observations; Sobol stores one scramble at a time.",
        "scope": "Initial single-asset price experiment. Numerical Greeks, calibration, and basket comparisons are not included.",
    }
    (tables / "monte_carlo_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    colors = {"plain": "#4063a6", "antithetic": "#e08431", "control": "#258473", "sobol": "#9250a1"}
    fig, axes = plt.subplots(len(setup["cases"]), 2, figsize=(11, 10), squeeze=False)
    for i, case in enumerate(setup["cases"]):
        for method in methods:
            values = sorted([row for row in summary if row["case"] == case["name"] and row["method"] == method],
                            key=lambda row: row["budget"])
            errors = [row["rmse"] for row in values]
            axes[i, 0].loglog([row["budget"] for row in values], errors, "o-", label=method, color=colors[method])
            axes[i, 1].loglog([1000 * row["median_seconds"] for row in values], errors, "o-", label=method, color=colors[method])
        axes[i, 0].set_xlabel("Total payoff evaluations (including pilots)")
        axes[i, 1].set_xlabel("Median time per pricing call (ms)")
        for ax in axes[i]:
            ax.set_title(case["name"].replace("_", " "))
            ax.set_ylabel("Price RMSE")
            ax.grid(alpha=0.25, which="both")
    axes[0, 0].legend()
    fig.suptitle(f"Monte Carlo comparison: {setup['repetitions']} independent runs per setting", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures / "monte_carlo_comparison.png", dpi=180)
    plt.close(fig)

    largest = max(setup["budgets"])
    print(f"\nLargest budget: {largest} payoff evaluations")
    for row in summary:
        if row["budget"] == largest:
            print(f"{row['case']:20s} {row['method']:11s} RMSE={row['rmse']:.6f}  "
                  f"median={1000 * row['median_seconds']:.3f} ms  coverage={row['coverage_95']:.0%}")
    print(f"\nSaved {len(rows)} runs, {len(summary)} summaries, metadata, and a comparison plot.")


if __name__ == "__main__":
    main()
