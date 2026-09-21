import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import t

from src.classical_benchmark import METHODS, evaluate_batch, evaluate_case
from src.common_benchmark import QUANTITIES, aggregate_metrics, reference_values


ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def measure_timing(method, cases, settings):
    timing = settings["timing"]
    for _ in range(timing["warmup_batches"]):
        evaluate_batch(method, cases, settings)
    elapsed = []
    for _ in range(timing["repetitions"]):
        start = perf_counter()
        evaluate_batch(method, cases, settings)
        elapsed.append(perf_counter() - start)
    values = np.array(elapsed)
    median = float(np.median(values))
    return {
        "method": method, "batch_size": len(cases),
        "warmup_batches": timing["warmup_batches"],
        "repetitions": timing["repetitions"],
        "median_batch_seconds": median,
        "q25_batch_seconds": float(np.quantile(values, 0.25)),
        "q75_batch_seconds": float(np.quantile(values, 0.75)),
        "median_seconds_per_option": median / len(cases),
        "median_options_per_second": len(cases) / median,
    }


def measure_memory(method):
    command = [sys.executable, "-m", "experiments.common_benchmark.memory_worker", method]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main():
    settings_path = ROOT / "experiments" / "config" / "common_benchmark_settings.json"
    settings_bytes = settings_path.read_bytes()
    settings = json.loads(settings_bytes)
    table_dir = ROOT / "results" / "common_benchmark" / "tables"
    figure_dir = ROOT / "results" / "common_benchmark" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    cases = load_csv(table_dir / "benchmark_cases.csv")
    references = {row["case_id"]: row for row in load_csv(table_dir / "benchmark_references.csv")}

    output_rows = []
    for method in METHODS:
        for case_index, case in enumerate(cases):
            start = perf_counter()
            values, standard_errors = evaluate_case(method, case, settings, case_index)
            elapsed = perf_counter() - start
            reference = references[case["case_id"]]
            for quantity in QUANTITIES:
                estimate = float(values[quantity])
                exact = float(reference[quantity])
                error = estimate - exact
                floor = settings["relative_error_floors"][quantity]
                output_rows.append({
                    "method": method, "case_id": case["case_id"], "split": case["split"],
                    "quantity": quantity, "estimate": estimate, "reference": exact,
                    "error": error, "absolute_error": abs(error),
                    "relative_error": error / exact if abs(exact) >= floor else "",
                    "standard_error": standard_errors[quantity],
                    "elapsed_seconds_all_outputs": elapsed,
                })
    write_csv(table_dir / "classical_outputs.csv", output_rows)
    metrics = aggregate_metrics(output_rows, settings["relative_error_floors"])
    write_csv(table_dir / "classical_metrics.csv", metrics)

    timing_cases = cases[:settings["timing"]["case_count"]]
    timing_rows = [measure_timing(method, timing_cases, settings) for method in METHODS]
    write_csv(table_dir / "classical_timing.csv", timing_rows)
    memory_rows = [measure_memory(method) for method in METHODS]
    write_csv(table_dir / "classical_memory.csv", memory_rows)

    critical = float(t.ppf(0.975, settings["monte_carlo"]["n_scrambles"] - 1))
    mc_rows = [row for row in output_rows if row["method"] == "conditional_rqmc"]
    coverage = {
        quantity: float(np.mean([
            abs(float(row["error"])) <= critical * float(row["standard_error"])
            for row in mc_rows if row["quantity"] == quantity
        ]))
        for quantity in QUANTITIES
    }
    validation = {
        "passed": all(np.isfinite(float(row["estimate"])) for row in output_rows),
        "case_count": len(cases),
        "method_count": len(METHODS),
        "output_rows": len(output_rows),
        "monte_carlo_95_percent_coverage": coverage,
        "timing_protocol": settings["timing"],
        "relative_error_floors": settings["relative_error_floors"],
    }
    (table_dir / "classical_validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    metadata = {
        "settings_sha256": hashlib.sha256(settings_bytes).hexdigest(),
        "cases_sha256": hashlib.sha256((table_dir / "benchmark_cases.csv").read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "numpy": np.__version__, "scipy": scipy.__version__,
        "methods": list(METHODS),
    }
    (table_dir / "classical_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(QUANTITIES))
    width = 0.35
    for offset, method in zip((-width / 2, width / 2), ("conditional_rqmc", "crank_nicolson")):
        selected = {(row["quantity"]): float(row["rmse"]) for row in metrics
                    if row["method"] == method and row["split"] == "all"}
        axes[0].bar(x + offset, [max(selected[name], 1e-16) for name in QUANTITIES],
                    width, label=method)
    axes[0].set_xticks(x, QUANTITIES)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Common-grid accuracy")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(
        [row["method"] for row in timing_rows],
        [row["median_seconds_per_option"] for row in timing_rows],
        color=("#55A868", "#C44E52", "#4C72B0"),
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median seconds per option")
    axes[1].set_title(f"Warm timing, batch size {len(timing_cases)}")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "classical_common_benchmark.png", dpi=180)
    plt.close(fig)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
