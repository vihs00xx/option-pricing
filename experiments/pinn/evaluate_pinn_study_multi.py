"""Evaluate all PINN study checkpoints (multi-seed) and aggregate over seeds."""

import csv
import json
import statistics
from pathlib import Path

from src.neural.fixed_pinn import load_fixed_pinn

from experiments.pinn.evaluate_pinn_study import evaluate_model


ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "experiments" / "config" / "pinn_study_settings.json"
VARIANTS = ("baseline_matched", "constrained", "causal")
# Tag families: the first physics-only study (original) and the revised
# study with corrected causal scaling / tuned concentration and rebalanced
# constraint categories (revised).
TAG_FAMILIES = {
    "original": ("s0", "s1", "s2"),
    "revised": ("v0", "v1", "v2"),
}


def revision_of(name):
    for family, tags in TAG_FAMILIES.items():
        if any(name.endswith(f"_{tag}") for tag in tags):
            return family
    return "reference"


def discover_checkpoints():
    """Return {name: checkpoint_path} for variant seeds and refinements."""
    checkpoints = ROOT / "checkpoints" / "pinn"
    found = {}
    for variant in VARIANTS:
        for family, tags in TAG_FAMILIES.items():
            for tag in tags:
                path = checkpoints / f"study_{variant}_{tag}.pt"
                if path.exists():
                    found[f"{variant}_{tag}"] = path
    for optimiser in ("lbfgs", "nys_newton_cg"):
        path = checkpoints / f"study_refined_{optimiser}.pt"
        if path.exists():
            found[f"refined_{optimiser}"] = path
    anchored = checkpoints / "fixed_black_scholes_pinn.pt"
    if anchored.exists():
        found["anchored"] = anchored
    return found


def model_metadata(name):
    """Wall-time/memory for a variant-seed or refined model."""
    all_tags = tuple(tag for tags in TAG_FAMILIES.values() for tag in tags)
    for variant in VARIANTS:
        for tag in all_tags:
            if name != f"{variant}_{tag}":
                continue
            path = (ROOT / "results" / "pinn" / f"study_{variant}_{tag}"
                    / "tables" / "training_metadata.json")
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
                "peak_rss_mb": lbfgs.get("peak_rss_mb") or meta.get("peak_rss_mb"),
            }
    if name == "anchored":
        path = ROOT / "results" / "pinn" / "fixed" / "tables" / "training_metadata.json"
        if not path.exists():
            return {}
        meta = json.loads(path.read_text())
        return {
            "total_wall_seconds": meta.get("wall_seconds"),
            "peak_rss_mb": meta.get("peak_rss_mb") or meta.get("baseline_rss_mb"),
            "uses_labelled_anchors": True,
        }
    if name in ("refined_lbfgs", "refined_nys_newton_cg"):
        optimiser = name.replace("refined_", "")
        path = (ROOT / "results" / "pinn" / "study_refinement" / "tables"
                / f"refinement_{optimiser}.json")
        if not path.exists():
            return {}
        meta = json.loads(path.read_text())
        return {
            "wall_seconds": meta.get("wall_seconds"),
            "peak_rss_mb": meta.get("peak_rss_mb"),
            "incremental_peak_mb": meta.get("incremental_peak_mb"),
            "initial_loss": meta.get("initial_loss"),
            "final_loss": meta.get("final_loss"),
        }
    return {}


def summary_row(name, result):
    core = result["errors"]["core"]
    worst_magnitude = max(
        s["maximum"] for s in result["violations"]["full"].values()
    )
    worst_proportion = max(
        s["proportion"] for s in result["violations"]["full"].values()
    )
    row = {
        "model": name,
        "revision": revision_of(name),
        "core_price_rmse": core["price"]["rmse"],
        "core_price_max_error": core["price"]["maximum_absolute_error"],
        "core_delta_rmse": core["delta"]["rmse"],
        "core_gamma_rmse": core["gamma"]["rmse"],
        "core_theta_rmse": core["theta"]["rmse"],
        "core_theta_max_error": core["theta"]["maximum_absolute_error"],
        "full_worst_violation_proportion": worst_proportion,
        "full_worst_violation_magnitude": worst_magnitude,
        "acceptance_passed": all(result["acceptance"].values()),
    }
    row.update(model_metadata(name))
    return row
HEADLINE = (
    "core_price_rmse", "core_theta_rmse", "core_delta_rmse", "core_gamma_rmse",
    "core_price_max_error", "core_theta_max_error",
    "full_worst_violation_proportion", "full_worst_violation_magnitude",
    "total_wall_seconds", "peak_rss_mb",
)


def aggregate(rows):
    """Mean +/- std per (variant, revision) across the three seed tags."""
    out = []
    for variant in VARIANTS:
        for family, tags in TAG_FAMILIES.items():
            seed_rows = [
                r for r in rows
                if r["model"] in (f"{variant}_{tag}" for tag in tags)
            ]
            if not seed_rows:
                continue
            agg = {"variant": variant, "revision": family,
                   "seeds": len(seed_rows)}
            for metric in HEADLINE:
                values = [r.get(metric) for r in seed_rows
                          if isinstance(r.get(metric), (int, float))]
                if not values:
                    continue
                agg[f"{metric}_mean"] = round(statistics.fmean(values), 6)
                agg[f"{metric}_std"] = round(statistics.pstdev(values), 6) \
                    if len(values) > 1 else 0.0
            out.append(agg)
    # Reference models (anchored, refined) carry no seed spread.
    for reference in ("anchored", "refined_lbfgs", "refined_nys_newton_cg"):
        reference_rows = [r for r in rows if r["model"] == reference]
        if not reference_rows:
            continue
        row = reference_rows[0]
        agg = {"variant": reference, "revision": "reference", "seeds": 1}
        for metric in HEADLINE:
            if isinstance(row.get(metric), (int, float)):
                agg[f"{metric}_mean"] = round(row[metric], 6)
                agg[f"{metric}_std"] = 0.0
        out.append(agg)
    return out


def main():
    settings = json.loads(SETTINGS_PATH.read_text())
    table_dir = ROOT / "results" / "pinn" / "study" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints()

    summary_rows = []
    for name, path in sorted(checkpoints.items()):
        try:
            model = load_fixed_pinn(path)
            model.eval()
            result, _ = evaluate_model(model, settings, name)
        except Exception as exc:  # noqa: BLE001 - keep evaluating the rest
            print(f"Skipping '{name}': {exc}")
            continue
        summary_rows.append(summary_row(name, result))
        core = result["errors"]["core"]
        print(f"[{name}] price={core['price']['rmse']:.3e} "
              f"theta={core['theta']['rmse']:.3e} "
              f"gamma={core['gamma']['rmse']:.3e}")

    if summary_rows:
        write_rows(table_dir / "run_summary.csv", summary_rows)
    aggregates = aggregate(summary_rows)
    if aggregates:
        write_rows(table_dir / "seed_aggregate.csv", aggregates)
    print(f"Wrote run_summary.csv ({len(summary_rows)} models) and "
          f"seed_aggregate.csv ({len(aggregates)} groups) to {table_dir}")


def write_rows(path, rows):
    """Write dict rows, using the ordered union of all keys as columns."""
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()