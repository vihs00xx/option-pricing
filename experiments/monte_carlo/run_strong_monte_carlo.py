import csv
import hashlib
import json
import platform
import tracemalloc
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.monte_carlo import monte_carlo_price
from src.monte_carlo_greeks import NAMES, monte_carlo_greeks
from src.sampling import METHODS


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def arguments(config, budget):
    return dict(method=config["method"], n_samples=budget,
                pilot_samples=budget // config.get("pilot_divisor", 8),
                n_scrambles=config.get("n_scrambles", 8))


def candidates(settings):
    result = [{"id": method, "method": method} for method in ("plain", "antithetic")]
    for method in ("control", "antithetic_control"):
        result.extend({"id": f"{method}_p{divisor}", "method": method, "pilot_divisor": divisor}
                      for divisor in settings["pilot_divisors"])
    for method in ("sobol", "sobol_control"):
        result.extend({"id": f"{method}_r{count}", "method": method, "n_scrambles": count,
                       "pilot_divisor": 2 if method == "sobol_control" else 8}
                      for count in settings["scramble_counts"])
    return result


def summarise(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["config"], row["budget"], row["quantity"])].append(row)
    summary = []
    for (case, config, budget, quantity), values in groups.items():
        errors = np.array([row["error"] for row in values])
        hits = sum(row["covered"] for row in values)
        n, z = len(values), 1.959963984540054
        center = (hits / n + z*z / (2*n)) / (1 + z*z/n)
        radius = z * np.sqrt((hits/n)*(1-hits/n)/n + z*z/(4*n*n)) / (1 + z*z/n)
        summary.append(dict(
            case=case, config=config, method=values[0]["method"], budget=budget, quantity=quantity,
            repetitions=n, reference=values[0]["reference"], scale=values[0]["scale"],
            mean_estimate=float(np.mean([row["estimate"] for row in values])),
            bias_estimate=float(errors.mean()), rmse=float(np.sqrt(np.mean(errors**2))),
            max_absolute_error=float(np.max(np.abs(errors))),
            mean_standard_error=float(np.mean([row["standard_error"] for row in values])),
            coverage=hits/n, coverage_wilson_low=center-radius, coverage_wilson_high=center+radius,
            zero_se_rate=float(np.mean([row["standard_error"] == 0 for row in values])),
            median_seconds=float(np.median([row["seconds"] for row in values])),
            payoff_evaluations=values[0]["payoff_evaluations"],
        ))
    return summary


def score(summary, config):
    rows = [row for row in summary if row["config"] == config]
    costs = [max((row["rmse"] / row["scale"])**2, 1e-28) * row["median_seconds"]
             * (10 if row["coverage"] < 0.8 or row["zero_se_rate"] > 0 else 1) for row in rows]
    return float(np.exp(np.mean(np.log(costs))))


def run_cases(settings, cases, configs, budgets, repetitions, seed_stream, greeks=False):
    rows = []
    for case in cases:
        inputs = {**settings["base"], **case["inputs"]}
        refs = {"price": black_scholes_price(**inputs)}
        if greeks:
            refs.update(black_scholes_greeks(**inputs))
        scales = dict(price=inputs["K"], delta=1.0, gamma=1 / inputs["K"],
                      vega=inputs["K"], theta=inputs["K"], rho=inputs["K"])
        for budget in budgets:
            for repeat in range(repetitions):
                offset = repeat % len(configs)
                for config in configs[offset:] + configs[:offset]:
                    child = seed_stream.spawn(1)[0]
                    seed = int(child.generate_state(1, dtype=np.uint64)[0])
                    kwargs = arguments(config, budget)
                    if greeks:
                        result = monte_carlo_greeks(**inputs, **kwargs, seed=seed,
                                                   estimator=config.get("estimator", "hybrid"),
                                                   relative_bump=config.get("relative_bump", 0.005))
                        estimates, errors, intervals = result.values, result.standard_errors, result.confidence_intervals
                    else:
                        result = monte_carlo_price(**inputs, **kwargs, seed=seed)
                        estimates, errors = {"price": result.price}, {"price": result.standard_error}
                        intervals = {"price": (result.ci_low, result.ci_high)}
                    for name, value in estimates.items():
                        lo, hi = intervals[name]
                        rows.append(dict(case=case["name"], **inputs, config=config["id"], method=config["method"],
                                         budget=budget, repetition=repeat, seed=seed, quantity=name,
                                         estimate=value, standard_error=errors[name], ci_low=lo, ci_high=hi,
                                         reference=refs[name], error=value-refs[name], scale=scales[name],
                                         covered=int(lo <= refs[name] <= hi), seconds=result.elapsed_seconds,
                                         payoff_evaluations=result.payoff_evaluations,
                                         independent_observations=result.independent_observations,
                                         pilot_samples=result.pilot_samples))
        print(f"Completed {'Greeks' if greeks else 'prices'}: {case['name']}", flush=True)
    return rows


def targets(settings, summary):
    groups = defaultdict(list)
    for row in summary:
        groups[(row["case"], row["config"])].append(row)
    rows = []
    for (case, config), values in groups.items():
        for tolerance in settings["absolute_price_targets"]:
            eligible = [v for v in values if v["rmse"] <= tolerance and
                        v["coverage"] >= settings["minimum_coverage"] and v["zero_se_rate"] == 0]
            selected = min(eligible, key=lambda v: v["median_seconds"]) if eligible else None
            rows.append(dict(case=case, config=config, target=tolerance,
                             attained=selected is not None,
                             budget=selected["budget"] if selected else "",
                             median_seconds=selected["median_seconds"] if selected else "",
                             measured_rmse=selected["rmse"] if selected else "",
                             coverage=selected["coverage"] if selected else ""))
    return rows


def plot_results(summary, greek_summary, figures):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method in METHODS:
        data = [row for row in summary if row["method"] == method]
        budgets = sorted({row["budget"] for row in data})
        rmses, times = [], []
        for budget in budgets:
            group = [row for row in data if row["budget"] == budget]
            rmses.append(np.sqrt(np.mean([row["rmse"]**2 for row in group])))
            times.append(1000 * np.median([row["median_seconds"] for row in group]))
        axes[0].loglog(budgets, rmses, "o-", label=method)
        axes[1].loglog(times, rmses, "o-", label=method)
    axes[0].set_xlabel("Total payoff evaluations, including pilot")
    axes[1].set_xlabel("Median pricing time across cases (ms)")
    for ax in axes:
        ax.set_ylabel("Pooled price RMSE across evaluation cases")
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen configurations on separate evaluation cases")
    fig.tight_layout()
    fig.savefig(figures / "strong_monte_carlo_prices.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    configs = list(dict.fromkeys(row["config"] for row in greek_summary))
    for ax, name in zip(axes.flat, NAMES):
        values = [np.sqrt(np.mean([(row["rmse"] / row["scale"])**2 for row in greek_summary
                                  if row["config"] == config and row["quantity"] == name])) for config in configs]
        ax.bar(range(len(configs)), values)
        ax.set_yscale("log")
        ax.set_xticks(range(len(configs)), [str(i+1) for i in range(len(configs))])
        ax.set_title(name)
        ax.set_ylabel("Scaled RMSE")
        ax.grid(axis="y", alpha=0.2)
    labels = [f"{i+1}: {name}" for i, name in enumerate(configs)]
    key = "\n".join("   ".join(labels[i:i+3]) for i in range(0, len(labels), 3))
    fig.suptitle("Price and Greeks estimated together; FD costs nine payoffs per draw")
    fig.text(0.5, 0.01, key, ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(figures / "strong_monte_carlo_greeks.png", dpi=180)
    plt.close(fig)


def main():
    start = perf_counter()
    project = Path(__file__).resolve().parents[2]
    settings = json.loads((project / "experiments" / "config" / "classical" / "benchmark_settings.json").read_text())
    tables = project / "results" / "monte_carlo" / "tables"
    figures = project / "results" / "monte_carlo" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(settings["seed"]).spawn(5)
    configs = candidates(settings)
    for config in configs:
        monte_carlo_price(**settings["base"], **arguments(config, 2048), seed=0)

    tuning_start = perf_counter()
    tuning = run_cases(settings, settings["tuning_cases"], configs, settings["tuning_budgets"],
                       settings["tuning_repetitions"], seeds[0])
    tuning_summary = summarise(tuning)
    save_csv(tables / "mc_tuning_runs.csv", tuning)
    save_csv(tables / "mc_tuning_summary.csv", tuning_summary)
    selected = [min([c for c in configs if c["method"] == method], key=lambda c: score(tuning_summary, c["id"]))
                for method in METHODS]
    preferred = min(selected, key=lambda c: score(tuning_summary, c["id"]))
    frozen = dict(selected=selected, preferred_by_tuning=preferred,
                  selection_rule="Geometric mean of scaled MSE times median runtime on tuning cases; tenfold penalty for coverage below 80% or any zero standard errors.",
                  tuning_seconds=perf_counter()-tuning_start,
                  scores={c["id"]: score(tuning_summary, c["id"]) for c in configs})
    frozen_path = tables / "mc_frozen_configurations.json"
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"Frozen price configurations; preferred on tuning: {preferred['id']}", flush=True)

    evaluation = run_cases(settings, settings["evaluation_cases"], selected, settings["budgets"],
                           settings["evaluation_repetitions"], seeds[1])
    summary = summarise(evaluation)
    save_csv(tables / "mc_evaluation_runs.csv", evaluation)
    save_csv(tables / "mc_evaluation_summary.csv", summary)
    save_csv(tables / "mc_target_accuracy.csv", targets(settings, summary))

    greek_configs = [{**c, "id": c["id"] + "_" + estimator, "estimator": estimator}
                     for estimator in ("hybrid", "likelihood_ratio") for c in configs]
    fd_configs = [{**preferred, "id": f"fd_{bump}", "estimator": "finite_difference", "relative_bump": bump}
                  for bump in settings["greek_bumps"]]
    greek_tuning_start = perf_counter()
    greek_tuning = run_cases(settings, settings["tuning_cases"][:4], greek_configs + fd_configs,
                             [settings["greek_budget"]], settings["greek_tuning_repetitions"], seeds[2], greeks=True)
    greek_tuning_summary = summarise(greek_tuning)
    save_csv(tables / "mc_greek_tuning_runs.csv", greek_tuning)
    save_csv(tables / "mc_greek_tuning_summary.csv", greek_tuning_summary)
    selected_fd = min(fd_configs, key=lambda c: score(greek_tuning_summary, c["id"]))
    reliable = [c for c in greek_configs if all(row["coverage"] >= 0.75 and row["zero_se_rate"] == 0
                for row in greek_tuning_summary if row["config"] == c["id"])]
    greek_selected = []
    for method in METHODS:
        eligible = [c for c in reliable if c["method"] == method]
        pool = eligible or [c for c in greek_configs if c["method"] == method]
        greek_selected.append(min(pool, key=lambda c: score(greek_tuning_summary, c["id"])))
    greek_selected.append(selected_fd)
    frozen["greek_selected"] = greek_selected
    frozen["greek_reliable_on_tuning"] = [c["id"] for c in reliable]
    frozen["greek_preferred_by_tuning"] = min(reliable, key=lambda c: score(greek_tuning_summary, c["id"])) if reliable else None
    frozen["greek_reliability_gate"] = "At least 75% coverage and no zero standard errors for every tuning case and quantity, before ranking by score. No recommendation if none qualify."
    frozen["greek_tuning_seconds"] = perf_counter()-greek_tuning_start
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    greek_runs = run_cases(settings, settings["evaluation_cases"], greek_selected, [settings["greek_budget"]],
                           settings["greek_evaluation_repetitions"], seeds[3], greeks=True)
    greek_summary = summarise(greek_runs)
    save_csv(tables / "mc_greek_evaluation_runs.csv", greek_runs)
    save_csv(tables / "mc_greek_evaluation_summary.csv", greek_summary)

    memory_rows = []
    for config in selected:
        tracemalloc.start()
        monte_carlo_price(**settings["base"], **arguments(config, max(settings["budgets"])), seed=99)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_rows.append(dict(config=config["id"], budget=max(settings["budgets"]), peak_traced_bytes=peak))
    save_csv(tables / "mc_memory.csv", memory_rows)
    plot_results(summary, greek_summary, figures)
    metadata = dict(settings=settings, python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(),
                    elapsed_seconds=perf_counter()-start, price_tuning_runs=len(tuning),
                    price_evaluation_runs=len(evaluation), greek_tuning_calls=len(greek_tuning)//6,
                    greek_evaluation_calls=len(greek_runs)//6,
                    source_hashes={str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in list((project / "src").glob("*.py")) + [Path(__file__)]},
                    timing="Warmup excluded. Each call includes sampling, pilot, payoff/Greek calculations and confidence intervals. Tuning cost recorded separately. File output and reference evaluation excluded from call timing.",
                    targets="Fastest measured budget satisfying observed RMSE, minimum empirical coverage and no zero SEs. Fixed independent repetitions; not adaptive stopping or an accuracy guarantee. Missing targets remain marked unattained.",
                    memory="Separate tracemalloc measurement: Python and tracked NumPy allocation peak, not total process RSS or all native allocations.",
                    greeks="Hybrid first derivatives are pathwise; Gamma is mixed pathwise/likelihood-ratio. Finite-difference CIs include sampling error only; comparisons against analytical Greeks include bump bias. All six outputs share the reported call time.",
                    development_archive="results/monte_carlo_development.zip retains earlier code, tuning and evaluation results. Poor Greek interval coverage motivated adding likelihood-ratio Greeks; the current evaluation uses new parameter cases and a new seed.",
                    scope="Single-asset constant-parameter European options only. No universal optimality claim; baskets, importance sampling and learned surrogates are not benchmarked.")
    (tables / "mc_benchmark_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Finished in {metadata['elapsed_seconds']:.1f}s; preferred on tuning: {preferred['id']}", flush=True)


if __name__ == "__main__":
    main()
