import csv
import hashlib
import json
import platform
from collections import defaultdict
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


def fit_rate(rows):
    x = np.log(np.array([row["n_samples"] for row in rows], dtype=float))
    y = np.log(np.array([row["rmse"] for row in rows], dtype=float))
    design = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([intercept, slope])
    variance = np.sum(residual**2) / (len(x) - 2)
    covariance = variance * np.linalg.inv(design.T @ design)
    slope_se = np.sqrt(covariance[1, 1])
    fitted = design @ np.array([intercept, slope])
    total = np.sum((y - y.mean())**2)
    r_squared = 1 - np.sum((y - fitted)**2) / total if total else 1.0
    return -slope, slope_se, r_squared


def main():
    project = Path(__file__).resolve().parents[2]
    settings_bytes = (project / "experiments" / "config" / "classical" / "convergence_settings.json").read_bytes()
    settings = json.loads(settings_bytes)
    tables = project / "results" / "monte_carlo" / "tables"
    figures = project / "results" / "monte_carlo" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    total = len(settings["cases"]) * len(settings["methods"]) * len(settings["sample_sizes"]) * settings["repetitions"]
    seeds = iter(np.random.SeedSequence(settings["seed"]).spawn(total))
    runs = []

    for case in settings["cases"]:
        inputs = {**settings["base"], **case["inputs"]}
        reference = black_scholes_price(**inputs)
        for n_samples in settings["sample_sizes"]:
            for repetition in range(settings["repetitions"]):
                offset = repetition % len(settings["methods"])
                order = settings["methods"][offset:] + settings["methods"][:offset]
                for method in order:
                    seed = int(next(seeds).generate_state(1, dtype=np.uint64)[0])
                    kwargs = dict(method=method["method"], n_samples=n_samples, seed=seed,
                                  n_scrambles=method.get("n_scrambles", 8))
                    if "control" in method["method"]:
                        kwargs["pilot_samples"] = int(n_samples * method["pilot_fraction"])
                    result = monte_carlo_price(**inputs, **kwargs)
                    runs.append(dict(case=case["name"], method=method["name"], n_samples=n_samples,
                                     repetition=repetition, seed=seed, reference=reference,
                                     estimate=result.price, error=result.price-reference,
                                     absolute_error=abs(result.price-reference),
                                     standard_error=result.standard_error,
                                     ci_low=result.ci_low, ci_high=result.ci_high,
                                     covered=int(result.ci_low <= reference <= result.ci_high),
                                     elapsed_seconds=result.elapsed_seconds,
                                     payoff_evaluations=result.payoff_evaluations,
                                     pilot_samples=result.pilot_samples,
                                     independent_observations=result.independent_observations))
        print(f"Completed {case['name']}", flush=True)

    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["case"], row["method"], row["n_samples"])].append(row)
    summary = []
    for (case, method, n_samples), values in grouped.items():
        errors = np.array([row["error"] for row in values])
        summary.append(dict(case=case, method=method, n_samples=n_samples,
                            repetitions=len(values), reference=values[0]["reference"],
                            mean_estimate=np.mean([row["estimate"] for row in values]),
                            bias=errors.mean(), rmse=np.sqrt(np.mean(errors**2)),
                            mean_absolute_error=np.mean(np.abs(errors)),
                            max_absolute_error=np.max(np.abs(errors)),
                            empirical_sd=np.std([row["estimate"] for row in values], ddof=1),
                            mean_reported_se=np.mean([row["standard_error"] for row in values]),
                            coverage=np.mean([row["covered"] for row in values]),
                            median_seconds=np.median([row["elapsed_seconds"] for row in values])))

    rates = []
    for case in [case["name"] for case in settings["cases"]]:
        for method in [method["name"] for method in settings["methods"]]:
            values = sorted([row for row in summary if row["case"] == case and row["method"] == method],
                            key=lambda row: row["n_samples"])
            alpha, slope_se, r_squared = fit_rate(values)
            tail_alpha, tail_se, tail_r_squared = fit_rate(values[-4:])
            rates.append(dict(case=case, method=method, alpha_all=alpha,
                              alpha_all_standard_error=slope_se, alpha_all_r_squared=r_squared,
                              alpha_tail=tail_alpha, alpha_tail_standard_error=tail_se,
                              alpha_tail_r_squared=tail_r_squared,
                              n_min=values[0]["n_samples"], n_max=values[-1]["n_samples"]))

    pooled = []
    for method in [method["name"] for method in settings["methods"]]:
        for n_samples in settings["sample_sizes"]:
            values = [row for row in summary if row["method"] == method and row["n_samples"] == n_samples]
            pooled.append(dict(method=method, n_samples=n_samples,
                               pooled_rmse=np.sqrt(np.mean([row["rmse"]**2 for row in values])),
                               median_seconds=np.median([row["median_seconds"] for row in values])))
    pooled_rates = []
    for method in [method["name"] for method in settings["methods"]]:
        values = sorted([row for row in pooled if row["method"] == method], key=lambda row: row["n_samples"])
        converted = [dict(n_samples=row["n_samples"], rmse=row["pooled_rmse"]) for row in values]
        alpha, slope_se, r_squared = fit_rate(converted)
        tail_alpha, tail_se, tail_r_squared = fit_rate(converted[-4:])
        pooled_rates.append(dict(method=method, alpha_all=alpha,
                                 alpha_all_standard_error=slope_se, alpha_all_r_squared=r_squared,
                                 alpha_tail=tail_alpha, alpha_tail_standard_error=tail_se,
                                 alpha_tail_r_squared=tail_r_squared))

    save_csv(tables / "mc_convergence_runs.csv", runs)
    save_csv(tables / "mc_convergence_summary.csv", summary)
    save_csv(tables / "mc_convergence_rates.csv", rates)
    save_csv(tables / "mc_convergence_pooled.csv", pooled)
    save_csv(tables / "mc_convergence_pooled_rates.csv", pooled_rates)
    metadata = dict(settings=settings, total_runs=len(runs), python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                    platform=platform.platform(),
                    protocol_id=settings["protocol_id"],
                    settings_sha256=hashlib.sha256(settings_bytes).hexdigest(),
                    rate_fit="OLS of log(RMSE) on log(N). alpha is minus the slope. Tail fit uses the largest four N values. Standard error is the ordinary regression slope SE over six or four aggregate points; it does not include case-selection uncertainty.",
                    work="N is total payoff evaluations for prices. Sobol-control includes its 50% independent pilot in N. Each Sobol production or pilot scramble has a power-of-two point count.",
                    caution="Twenty-four independent repetitions give noisy RMSE and coverage estimates, especially when QMC errors are very small. Observed alpha is empirical for these four one-dimensional European cases, not a universal QMC convergence theorem.")
    (tables / "mc_convergence_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, case in zip(axes.flat, [case["name"] for case in settings["cases"]]):
        for method in [method["name"] for method in settings["methods"]]:
            values = sorted([row for row in summary if row["case"] == case and row["method"] == method],
                            key=lambda row: row["n_samples"])
            ax.loglog([row["n_samples"] for row in values], [row["rmse"] for row in values], "o-", label=method)
        ax.set_title(case.replace("_", " "))
        ax.set_xlabel("Total payoff evaluations N")
        ax.set_ylabel("Price RMSE")
        ax.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Monte Carlo convergence through $2^{20}$")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures / "mc_convergence_to_2pow20.png", dpi=180)
    plt.close(fig)

    print("Pooled convergence rates")
    for row in pooled_rates:
        print(f"{row['method']:18s} alpha={row['alpha_all']:.3f}  tail={row['alpha_tail']:.3f}")


if __name__ == "__main__":
    main()
