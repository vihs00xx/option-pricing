import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.monte_carlo_greeks import NAMES, monte_carlo_greeks


def references(inputs):
    return {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}


def run(cases, repetitions, seed_root, production, fraction, scrambles, pilot_ratio, stage):
    rows = []
    pilot = int(production * pilot_ratio)
    method = "sobol_control" if pilot else "sobol"
    for case_index, case in enumerate(cases):
        inputs = case["inputs"]
        reference = references(inputs)
        for repetition in range(repetitions):
            seed = seed_root + 10000 * case_index + repetition
            result = monte_carlo_greeks(
                **inputs, estimator="conditional", method=method,
                n_samples=production + pilot, pilot_samples=pilot or 1024,
                n_scrambles=scrambles, conditional_fraction=fraction, seed=seed,
            )
            for name in NAMES:
                low, high = result.confidence_intervals[name]
                rows.append({
                    "stage": stage, "case": case["name"], "repetition": repetition,
                    "seed": seed, "conditional_fraction": fraction,
                    "n_scrambles": scrambles, "pilot_ratio": pilot_ratio,
                    "production_points": production, "total_points": production + pilot,
                    "quantity": name, "estimate": result.values[name],
                    "reference": reference[name], "error": result.values[name] - reference[name],
                    "standard_error": result.standard_errors[name],
                    "covered": int(low <= reference[name] <= high),
                    "elapsed_seconds": result.elapsed_seconds,
                })
    return rows


def summarize(rows):
    summaries = []
    keys = sorted({(row["stage"], row["conditional_fraction"], row["n_scrambles"], row["pilot_ratio"])
                   for row in rows})
    for stage, fraction, scrambles, pilot_ratio in keys:
        configuration = [row for row in rows if row["stage"] == stage
                         and row["conditional_fraction"] == fraction
                         and row["n_scrambles"] == scrambles
                         and row["pilot_ratio"] == pilot_ratio]
        for name in NAMES:
            selected = [row for row in configuration if row["quantity"] == name]
            errors = np.array([row["error"] for row in selected])
            summaries.append({
                "stage": stage, "conditional_fraction": fraction, "n_scrambles": scrambles,
                "pilot_ratio": pilot_ratio, "quantity": name,
                "total_points": configuration[0]["total_points"],
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "bias": float(np.mean(errors)),
                "coverage": float(np.mean([row["covered"] for row in selected])),
                "zero_standard_error_rate": float(np.mean([row["standard_error"] == 0 for row in selected])),
                "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in selected])),
            })
    return summaries


def choose(summary, stage, coverage_target, include_cost):
    candidates = sorted({(row["conditional_fraction"], row["n_scrambles"], row["pilot_ratio"])
                         for row in summary if row["stage"] == stage})
    by_quantity_minimum = {}
    for name in NAMES:
        values = [row["rmse"] * np.sqrt(row["total_points"]) if include_cost else row["rmse"]
                  for row in summary if row["stage"] == stage and row["quantity"] == name]
        by_quantity_minimum[name] = min(values)
    scored = []
    for candidate in candidates:
        selected = [row for row in summary if row["stage"] == stage
                    and (row["conditional_fraction"], row["n_scrambles"], row["pilot_ratio"]) == candidate]
        normalized = []
        for row in selected:
            value = row["rmse"] * np.sqrt(row["total_points"]) if include_cost else row["rmse"]
            normalized.append(value / by_quantity_minimum[row["quantity"]])
        eligible = min(row["coverage"] for row in selected) >= coverage_target \
            and max(row["zero_standard_error_rate"] for row in selected) == 0
        score = float(np.mean(np.log(normalized))) + (0 if eligible else 100)
        scored.append((score, candidate, eligible))
    return min(scored, key=lambda item: item[0]), scored


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    settings_path = ROOT / "experiments" / "config" / "conditional_study_settings.json"
    settings_bytes = settings_path.read_bytes()
    settings = json.loads(settings_bytes)
    benchmark = json.loads((ROOT / "experiments" / "config" / "benchmark_settings.json").read_text())
    base = benchmark["base"]
    cases = [{"name": case["name"], "inputs": {**base, **case["inputs"]}}
             for case in benchmark["tuning_cases"]]
    tuning = settings["tuning"]
    rows = []
    for fraction in tuning["conditional_fractions"]:
        for scrambles in tuning["scramble_counts"]:
            rows.extend(run(cases, tuning["repetitions"], tuning["seed_root"],
                            tuning["production_points"], fraction, scrambles, 0.0, "smoothing"))
    first_summary = summarize(rows)
    selected_smoothing, smoothing_scores = choose(
        first_summary, "smoothing", tuning["minimum_aggregate_coverage"], False)
    _, (fraction, scrambles, _), _ = selected_smoothing
    for pilot_ratio in tuning["pilot_ratios"]:
        rows.extend(run(cases, tuning["repetitions"], tuning["seed_root"] + 1000000,
                        tuning["production_points"], fraction, scrambles, pilot_ratio, "control"))
    summary = summarize(rows)
    selected_control, control_scores = choose(
        summary, "control", tuning["minimum_aggregate_coverage"], True)
    _, (fraction, scrambles, pilot_ratio), eligible = selected_control
    output = ROOT / "results" / "conditional" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "mc_conditional_tuning_runs.csv", rows)
    write_csv(output / "mc_conditional_tuning_summary.csv", summary)
    frozen = {
        "settings_sha256": hashlib.sha256(settings_bytes).hexdigest(),
        "selection_rule": "Minimum geometric mean of quantity-normalized RMSE; control stage uses RMSE times sqrt(total sample count).",
        "coverage_requirement": tuning["minimum_aggregate_coverage"],
        "selected": {"conditional_fraction": fraction, "n_scrambles": scrambles,
                     "pilot_ratio_to_production": pilot_ratio,
                     "method": "sobol_control" if pilot_ratio else "sobol"},
        "eligible": eligible,
        "smoothing_scores": [{"score": score, "configuration": candidate, "eligible": valid}
                             for score, candidate, valid in sorted(smoothing_scores)],
        "control_scores": [{"score": score, "configuration": candidate, "eligible": valid}
                           for score, candidate, valid in sorted(control_scores)],
    }
    frozen_path = output / "mc_conditional_frozen_configuration.json"
    if frozen_path.exists():
        raise FileExistsError(f"Refusing to replace frozen configuration: {frozen_path}")
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps(frozen["selected"], indent=2))


if __name__ == "__main__":
    main()
