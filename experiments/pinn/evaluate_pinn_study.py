"""Evaluate every PINN study checkpoint under one shared protocol.

Usage:
    python -m experiments.pinn.evaluate_pinn_study

Evaluates the matched-budget variants and the refined checkpoints with
the same grid and acceptance thresholds as the fixed-PINN evaluation,
adds region-wise violation diagnostics for each model, and writes a
combined comparison table. Evaluation only measures; it never raises
on failed acceptance thresholds.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.neural.fixed_pinn import load_fixed_pinn
from src.neural.pinn_diagnostics import (
    acceptance_checks,
    full_diagnostics,
    predict_grid,
    reference_values,
)


ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "experiments" / "config" / "pinn_study_settings.json"
STUDY_NAMES = (
    "baseline_matched",
    "constrained",
    "causal",
    "refined_lbfgs",
    "refined_nys_newton_cg",
)


def evaluate_model(model, settings, name):
    evaluation = settings["evaluation"]
    problem = settings["problem"]
    x_axis = np.linspace(problem["x_min"], problem["x_max"], evaluation["x_points"])
    tau_axis = np.linspace(0, problem["T"], evaluation["tau_points"])
    x_grid, tau_grid = np.meshgrid(x_axis, tau_axis, indexing="xy")
    spots = (problem["K"] * x_grid).ravel()
    tau = tau_grid.ravel()
    y = np.log(spots / problem["K"])
    predicted = predict_grid(model, y, tau, evaluation["batch_size"])
    reference = reference_values(
        spots, tau, problem["K"], problem["r"], problem["sigma"], problem["q"]
    )
    diagnostics = full_diagnostics(
        predicted, reference, spots, tau,
        problem["K"], problem["r"], problem["q"], evaluation,
    )
    result = {
        "model": name,
        "errors": diagnostics["errors"],
        "violations": diagnostics["violations"],
        "acceptance": acceptance_checks(diagnostics["errors"], settings["acceptance"]),
        "violation_magnitudes": diagnostics["violation_magnitudes"],
        "regions": diagnostics["regions"],
    }
    grid = {
        "spots": spots, "tau": tau, "predicted": predicted,
        "reference": reference, "x_axis": x_axis, "tau_axis": tau_axis,
    }
    return result, grid


def serializable(result):
    """Return a JSON-safe copy (drops raw mask and magnitude arrays)."""
    return {
        "model": result["model"],
        "errors": result["errors"],
        "violations": result["violations"],
        "acceptance": result["acceptance"],
        "region_points": {
            name: int(mask.sum()) for name, mask in result["regions"].items()
        },
    }


def comparison_rows(results):
    rows = []
    for name, result in results.items():
        for region in ("full", "core", "near_expiry", "near_strike",
                       "low_tail", "high_tail"):
            row = {"model": name, "region": region}
            for quantity in ("price", "delta", "gamma", "theta"):
                metrics = result["errors"][region][quantity]
                row[f"{quantity}_rmse"] = metrics["rmse"]
                row[f"{quantity}_max_error"] = metrics["maximum_absolute_error"]
            for violation in (
                "price_below_intrinsic", "price_above_discounted_spot",
                "delta_below_zero", "delta_above_discount_factor",
                "gamma_negative",
            ):
                summary = result["violations"][region][violation]
                row[f"{violation}_proportion"] = summary["proportion"]
                row[f"{violation}_maximum"] = summary["maximum"]
                row[f"{violation}_mean_size"] = summary["mean_size"]
            row["acceptance_passed"] = all(result["acceptance"].values())
            rows.append(row)
    return rows


def run_metadata(name):
    """Wall-time and memory metadata for a model, resolved by name."""
    variants = ("baseline_matched", "constrained", "causal")
    if name in variants:
        path = ROOT / "results" / "pinn" / f"study_{name}" / "tables" / "training_metadata.json"
        if not path.exists():
            return {}
        meta = json.loads(path.read_text())
        lbfgs = meta.get("lbfgs", {}) or {}
        return {
            "adam_wall_seconds": meta.get("wall_seconds"),
            "lbfgs_wall_seconds": lbfgs.get("wall_seconds"),
            "lbfgs_closure_calls": lbfgs.get("closure_calls"),
            "total_wall_seconds": (meta.get("wall_seconds") or 0.0)
                                  + (lbfgs.get("wall_seconds") or 0.0),
            "adam_peak_rss_mb": meta.get("peak_rss_mb"),
            "lbfgs_peak_rss_mb": lbfgs.get("peak_rss_mb"),
        }
    if name in ("refined_lbfgs", "refined_nys_newton_cg"):
        optimiser = name.replace("refined_", "")
        path = ROOT / "results" / "pinn" / "study_refinement" / "tables" / f"refinement_{optimiser}.json"
        if not path.exists():
            return {}
        meta = json.loads(path.read_text())
        return {
            "initial_loss": meta.get("initial_loss"),
            "final_loss": meta.get("final_loss"),
            "wall_seconds": meta.get("wall_seconds"),
            "peak_rss_mb": meta.get("peak_rss_mb"),
            "incremental_peak_mb": meta.get("incremental_peak_mb"),
        }
    return {}


def run_summary_rows(results):
    """One row per model: core accuracy, worst violation magnitude, and
    run cost (wall time and peak memory)."""
    rows = []
    for name, result in results.items():
        core = result["errors"]["core"]
        worst_violation_magnitude = max(
            summary["maximum"] for summary in result["violations"]["full"].values()
        )
        worst_violation_proportion = max(
            summary["proportion"] for summary in result["violations"]["full"].values()
        )
        row = {
            "model": name,
            "core_price_rmse": core["price"]["rmse"],
            "core_price_max_error": core["price"]["maximum_absolute_error"],
            "core_delta_rmse": core["delta"]["rmse"],
            "core_gamma_rmse": core["gamma"]["rmse"],
            "core_theta_rmse": core["theta"]["rmse"],
            "core_theta_max_error": core["theta"]["maximum_absolute_error"],
            "full_worst_violation_proportion": worst_violation_proportion,
            "full_worst_violation_magnitude": worst_violation_magnitude,
            "acceptance_passed": all(result["acceptance"].values()),
        }
        row.update(run_metadata(name))
        rows.append(row)
    return rows


def main():
    settings = json.loads(SETTINGS_PATH.read_text())
    table_dir = ROOT / "results" / "pinn" / "study" / "tables"
    figure_dir = ROOT / "results" / "pinn" / "study" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    grids = {}
    for name in STUDY_NAMES:
        checkpoint = ROOT / "checkpoints" / "pinn" / f"study_{name}.pt"
        if not checkpoint.exists():
            print(f"Skipping '{name}': no checkpoint at {checkpoint}")
            continue
        model = load_fixed_pinn(checkpoint)
        model.eval()
        result, grid = evaluate_model(model, settings, name)
        results[name] = result
        grids[name] = grid
        (table_dir / f"{name}_evaluation.json").write_text(
            json.dumps(serializable(result), indent=2) + "\n"
        )
        acceptance = ", ".join(
            f"{key}={'pass' if value else 'FAIL'}"
            for key, value in result["acceptance"].items()
        )
        core = result["errors"]["core"]
        print(f"[{name}] core price RMSE={core['price']['rmse']:.3e} "
              f"delta={core['delta']['rmse']:.3e} "
              f"gamma={core['gamma']['rmse']:.3e} "
              f"theta={core['theta']['rmse']:.3e} | {acceptance}")

    rows = comparison_rows(results)
    with (table_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = run_summary_rows(results)
    if summary_rows:
        with (table_dir / "run_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
    make_figures(results, grids, figure_dir)
    print(f"Wrote comparison table for {len(results)} models to {table_dir}")


def make_figures(results, grids, figure_dir):
    if not grids:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    first = next(iter(grids.values()))
    spots, tau, reference = first["spots"], first["tau"], first["reference"]
    final = tau == np.max(tau)
    axes[0, 0].plot(spots[final], reference["price"][final], "k-",
                    label="Black--Scholes", linewidth=1.5)
    for name, grid in grids.items():
        axes[0, 0].plot(spots[final], grid["predicted"]["price"][final],
                        "--", label=name)
    axes[0, 0].set_title("Price at one year to maturity")
    axes[0, 0].set_xlabel("Spot")
    axes[0, 1].plot(spots[final], reference["greeks"]["delta"][final], "k-",
                    linewidth=1.5)
    for name, grid in grids.items():
        axes[0, 1].plot(spots[final], grid["predicted"]["delta"][final],
                        "--", label=name)
    axes[0, 1].set_title("Delta at one year")
    axes[1, 0].plot(spots[final], reference["greeks"]["gamma"][final], "k-",
                    linewidth=1.5)
    for name, grid in grids.items():
        axes[1, 0].plot(spots[final], grid["predicted"]["gamma"][final],
                        "--", label=name)
    axes[1, 0].set_title("Gamma at one year")
    axes[1, 0].set_xlabel("Spot")
    regions = ("near_expiry", "near_strike", "low_tail", "high_tail")
    width = 0.8 / max(len(grids), 1)
    positions = np.arange(len(regions))
    for index, (name, result) in enumerate(results.items()):
        proportions = [
            max(
                summary["proportion"]
                for summary in result["violations"][region].values()
            )
            for region in regions
        ]
        axes[1, 1].bar(positions + index * width, proportions, width,
                       label=name)
    axes[1, 1].set_xticks(positions + width * (len(grids) - 1) / 2)
    axes[1, 1].set_xticklabels(regions)
    axes[1, 1].set_title("Worst violation proportion per region")
    axes[1, 1].set_ylabel("Proportion of points")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "study_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
