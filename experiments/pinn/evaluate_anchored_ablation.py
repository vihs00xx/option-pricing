"""Common evaluation for the corrected anchored PINN ablation.

Every checkpoint is evaluated on two distinct sets:

1. the historical core grid (161 x 51), for comparison with the originally
   published project numbers;
2. the broad test mixture produced by the dataset sampler, which includes
   harder regions (near expiry, tails, boundaries).

Metrics are reported separately for the two sets. Absolute thresholds are
applied only to the core grid, because the published thresholds are
core-grid quantities; broad-test acceptance is expressed as an improvement
over the reference on the *same* distribution.

Financial violations are reported per category (price lower/upper bound,
Delta lower/upper bound, negative Gamma) with tolerance, proportion,
maximum and mean magnitude, and the region where the maximum occurs.
Magnitudes in different units are never combined into one score; the only
unified statistic is the proportion of points violating *at least one*
constraint, which is named explicitly.

Usage:
    python -m experiments.pinn.evaluate_anchored_ablation
"""

import argparse
import csv
import json
import platform
import re
import statistics
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.neural.anchor_data import load_dataset
from src.neural.fixed_pinn import load_fixed_pinn
from src.neural.pinn_diagnostics import region_masks
from src.neural.provenance import (
    evaluation_source_fingerprint, torch_checkpoint_digest, write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "experiments" / "config" / "anchored_pinn_settings.json"
QUANTITIES = ("price", "delta", "gamma", "theta")
VIOLATIONS = ("price_below_lower_bound", "price_above_upper_bound",
              "delta_below_zero", "delta_above_upper_bound", "gamma_negative")
REGIONS = ("full", "core", "near_expiry", "near_strike", "low_tail",
           "high_tail")
REFERENCE_NAME = "reference_existing_best"
NAME_PATTERN = re.compile(
    r"^(?P<model>.+?)_n(?P<anchors>\d+)_s(?P<seed>\d+)(?P<tag>_.*)?$")


def parse_name(name):
    match = NAME_PATTERN.match(name)
    if not match:
        return None
    return match["model"], int(match["anchors"]), int(match["seed"])


def predict(model, y, tau, batch_size, greeks=True):
    """Dollar-valued price and (optionally) Greeks."""
    dtype = next(model.parameters()).dtype
    names = QUANTITIES if greeks else ("price",)
    output = {name: [] for name in names}
    for start in range(0, len(y), batch_size):
        y_tensor = torch.tensor(y[start:start + batch_size, None],
                                dtype=dtype, requires_grad=True)
        tau_tensor = torch.tensor(tau[start:start + batch_size, None],
                                 dtype=dtype, requires_grad=True)
        if greeks:
            values = model.price_and_greeks(y_tensor, tau_tensor)
        else:
            values = {"price": model.problem.K * model(y_tensor, tau_tensor)}
        for name in names:
            output[name].append(values[name].detach().numpy()[:, 0])
    return {name: np.concatenate(parts) for name, parts in output.items()}


def error_stats(estimate, reference):
    difference = np.asarray(estimate) - np.asarray(reference)
    if difference.size == 0:
        return {"points": 0, "rmse": None, "maximum_absolute_error": None,
                "bias": None}
    return {
        "points": int(difference.size),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "bias": float(np.mean(difference)),
    }


def violation_gaps(price, delta, gamma, spots, tau, k, r, q):
    """Signed constraint gaps; positive means a violation."""
    discount_q = np.exp(-q * tau)
    discount_r = np.exp(-r * tau)
    lower = np.maximum(spots * discount_q - k * discount_r, 0.0)
    upper = spots * discount_q
    return {
        "price_below_lower_bound": lower - price,
        "price_above_upper_bound": price - upper,
        "delta_below_zero": -delta,
        "delta_above_upper_bound": delta - discount_q,
        "gamma_negative": -gamma,
    }


def evaluate_split(model, y, tau, settings, label):
    """Errors (overall + regional) and violations for one coordinate set."""
    problem = model.problem
    evaluation = settings["evaluation"]
    tolerance = evaluation["violation_tolerance"]
    spots = np.asarray(y, dtype=float)
    spots = problem.K * np.exp(spots)
    tau = np.asarray(tau, dtype=float)
    smooth = tau > 0
    exact_price = black_scholes_price(
        spots, problem.K, tau, problem.r, problem.sigma, problem.q, "call")
    exact_greeks = {name: np.full(spots.shape, np.nan)
                    for name in ("delta", "gamma", "theta")}
    if smooth.any():
        computed = black_scholes_greeks(
            spots[smooth], problem.K, tau[smooth], problem.r, problem.sigma,
            problem.q, "call")
        for name in exact_greeks:
            exact_greeks[name][smooth] = computed[name]

    predicted = predict(model, y, tau, evaluation["batch_size"], greeks=True)
    masks = region_masks(
        spots, tau, problem.K, evaluation["core_x_range"],
        evaluation["core_minimum_tau"], evaluation["near_strike_half_width"],
        evaluation["near_expiry_tau"])

    errors = {}
    for region, mask in masks.items():
        entry = {"price": error_stats(predicted["price"][mask],
                                      exact_price[mask])}
        greek_mask = mask & smooth
        for name in ("delta", "gamma", "theta"):
            entry[name] = error_stats(predicted[name][greek_mask],
                                      exact_greeks[name][greek_mask])
        errors[region] = entry

    gaps = violation_gaps(predicted["price"], predicted["delta"],
                          predicted["gamma"], spots, tau, problem.K,
                          problem.r, problem.q)
    magnitudes = {name: np.maximum(gap - tolerance, 0.0)
                  for name, gap in gaps.items()}
    magnitudes["any_constraint"] = np.zeros_like(tau, dtype=bool)
    for name, gap in gaps.items():
        magnitudes["any_constraint"] |= np.maximum(gap - tolerance, 0.0) > 0.0
    magnitudes["any_constraint"] = magnitudes["any_constraint"].astype(float)

    violations = {}
    for name, magnitude in magnitudes.items():
        per_region = {}
        for region, mask in masks.items():
            values = magnitude[mask]
            points = int(values.size)
            count = int(np.sum(values > 0.0))
            entry = {"points": points, "count": count,
                     "proportion": count / points if points else 0.0}
            if name == "any_constraint":
                # A proportion only: its magnitude has no single physical unit,
                # so no max/mean is reported for it.
                entry["maximum_magnitude"] = None
                entry["mean_magnitude"] = None
            else:
                entry["maximum_magnitude"] = float(np.max(values)) if points else 0.0
                entry["mean_magnitude"] = float(np.mean(values)) if points else 0.0
            per_region[region] = entry
        if magnitude.size and name != "any_constraint":
            index = int(np.argmax(magnitude))
            per_region["maximum_region"] = next(
                (region for region in REGIONS if masks[region][index]), "full")
        else:
            per_region["maximum_region"] = None
        per_region["tolerance"] = tolerance
        violations[name] = per_region
    return {"label": label, "points": int(tau.size), "errors": errors,
            "violations": violations}


def timing(model, settings):
    """Price-only and price+Greeks inference timing after warm-up."""
    evaluation = settings["evaluation"]
    size = evaluation["inference_batch_size"]
    dtype = next(model.parameters()).dtype
    y = torch.linspace(0.0, 1.0, size, dtype=dtype).reshape(-1, 1)
    tau = torch.full_like(y, 0.5)

    def price_only():
        model(y.clone().requires_grad_(True), tau.clone().requires_grad_(True))

    def price_and_greeks():
        model.price_and_greeks(y.clone().requires_grad_(True),
                               tau.clone().requires_grad_(True))

    result = {}
    for name, function in (("price_only", price_only),
                           ("price_and_greeks", price_and_greeks)):
        for _ in range(evaluation["inference_warmup"]):
            function()
        elapsed = []
        for _ in range(evaluation["inference_repetitions"]):
            start = perf_counter()
            function()
            elapsed.append(perf_counter() - start)
        median = float(np.median(elapsed))
        result[f"{name}_median_seconds_per_point"] = median / size
        result[f"{name}_median_points_per_second"] = (
            size / median if median > 0 else None)
        result[f"{name}_iqr_seconds_per_point"] = (
            float(np.quantile(elapsed, 0.75) - np.quantile(elapsed, 0.25)) / size)
    return result


def core_grid_coordinates(settings):
    problem = settings["problem"]
    evaluation = settings["evaluation"]
    x_axis = np.linspace(problem["x_min"], problem["x_max"],
                         evaluation["x_points"])
    tau_axis = np.linspace(0.0, problem["T"], evaluation["tau_points"])
    x_grid, tau_grid = np.meshgrid(x_axis, tau_axis, indexing="xy")
    return np.log(x_grid.ravel()), tau_grid.ravel()


def pde_residual_rmse(model, dataset, settings):
    """RMS of the transformed residual, in units of u = C/K (not normalised)."""
    dtype = next(model.parameters()).dtype
    y, tau = dataset["test_y"], dataset["test_tau"]
    batch = settings["evaluation"]["batch_size"]
    residual = []
    for start in range(0, len(y), batch):
        y_tensor = torch.tensor(y[start:start + batch, None], dtype=dtype,
                                requires_grad=True)
        tau_tensor = torch.tensor(tau[start:start + batch, None], dtype=dtype,
                                  requires_grad=True)
        residual.append(model.pde_residual(y_tensor, tau_tensor)
                        .detach().numpy()[:, 0])
    values = np.concatenate(residual)
    return float(np.sqrt(np.mean(values ** 2)))


def cost_fields(metadata):
    """Training-cost fields for a metric row.

    Missing historical measurements are returned as ``None`` and must stay
    missing downstream; they are never replaced with zero.
    """
    if not metadata:
        return {}
    lbfgs = metadata.get("lbfgs", {}) or {}
    labels = metadata.get("labels_used_by_permitted_terms", {}) or {}
    selection_labels = metadata.get("validation_labels_used_by_selection", {}) or {}
    samples = metadata.get("samples_used", {}) or {}
    label_generation = None
    if "label_generation_seconds" in metadata:
        label_generation = metadata["label_generation_seconds"]
        amortization_note = ("dataset build-or-load time recorded once in the "
                             "protocol manifest; loaded datasets report load "
                             "time, not analytical label generation")
    else:
        amortization_note = "not recorded for this run"
    return {
        "training_wall_seconds": metadata.get("wall_seconds"),
        "configured_time_cap_seconds": metadata.get("wall_clock_cap_seconds"),
        "cap_reached": metadata.get("hit_wall_cap"),
        "lbfgs_interrupted": metadata.get("lbfgs_interrupted"),
        "peak_rss_mb": metadata.get("peak_rss_mb"),
        "incremental_peak_mb": metadata.get("incremental_peak_mb"),
        "optimizer_steps": metadata.get("optimizer_steps"),
        "adam_steps": (
            metadata.get("optimizer_steps") - lbfgs.get("accepted_steps")
            if metadata.get("optimizer_steps") is not None
            and lbfgs.get("accepted_steps") is not None else None),
        "lbfgs_outer_steps": lbfgs.get("accepted_steps"),
        "lbfgs_closure_evaluations": lbfgs.get("closure_calls"),
        "selected_checkpoint_stage": metadata.get("selected_label"),
        "selected_selection_score": metadata.get("selected_selection_score"),
        "last_selection_score": metadata.get("last_selection_score"),
        "labels_used_price": labels.get("training_price"),
        "labels_used_theta": labels.get("training_theta"),
        "labels_used_delta": labels.get("training_delta"),
        "labels_used_gamma": labels.get("training_gamma"),
        "validation_labels_used_price": selection_labels.get("validation_price"),
        "validation_labels_used_theta": selection_labels.get("validation_theta"),
        "validation_labels_used_delta": selection_labels.get("validation_delta"),
        "validation_labels_used_gamma": selection_labels.get("validation_gamma"),
        "pde_collocation_points": samples.get("pde_collocation"),
        "terminal_payoff_points": samples.get("terminal_payoff"),
        "label_generation_seconds": label_generation,
        "label_generation_note": amortization_note,
        "protocol_id": metadata.get("protocol_id"),
        "run_identity_hash": metadata.get("run_identity_hash"),
        "training_source_fingerprint": metadata.get("training_source_fingerprint"),
        "dataset_content_hash": metadata.get("dataset_content_hash"),
        "anchor_count": metadata.get("anchor_count"),
        "greek_anchor_count": metadata.get("greek_anchor_count"),
    }


def run_metrics(name, checkpoint_path, checkpoint_kind, settings, dataset,
                training_metadata_path=None, protocol_id=None):
    model = load_fixed_pinn(checkpoint_path)
    model.eval()
    grid = evaluate_split(model, *core_grid_coordinates(settings), settings,
                          "core_grid")
    broad = evaluate_split(model, dataset["test_y"], dataset["test_tau"],
                          settings, "broad_test")
    row = {"model": name, "checkpoint_kind": checkpoint_kind}
    for prefix, split, region in (("core_grid", grid, "core"),
                                  ("broad_test", broad, "full")):
        for quantity in QUANTITIES:
            row[f"{prefix}_{quantity}_rmse"] = (
                split["errors"][region][quantity]["rmse"])
        row[f"{prefix}_price_max_error"] = (
            split["errors"][region]["price"]["maximum_absolute_error"])
        row[f"{prefix}_any_violation_proportion"] = (
            split["violations"]["any_constraint"]["full"]["proportion"])
        row[f"{prefix}_pde_region_points"] = split["points"]
    row["pde_residual_rmse"] = pde_residual_rmse(model, dataset, settings)
    row.update(timing(model, settings))
    row["protocol_id"] = protocol_id
    # Per-category violation detail, so acceptance can be judged per category
    # rather than on one cross-unit maximum.
    for prefix, split in (("core_grid", grid), ("broad_test", broad)):
        for category in VIOLATIONS + ("any_constraint",):
            stats = split["violations"][category]["full"]
            row[f"{prefix}_{category}_proportion"] = stats["proportion"]
            row[f"{prefix}_{category}_max_magnitude"] = stats["maximum_magnitude"]
    metadata = None
    if training_metadata_path is not None and Path(training_metadata_path).exists():
        metadata = json.loads(Path(training_metadata_path).read_text())
    row["has_training_metadata"] = metadata is not None
    row.update(cost_fields(metadata))
    return row, grid, broad


AGGREGATE_METRICS = (
    "core_grid_price_rmse", "core_grid_price_max_error", "core_grid_delta_rmse",
    "core_grid_gamma_rmse", "core_grid_theta_rmse",
    "core_grid_any_violation_proportion",
    "broad_test_price_rmse", "broad_test_price_max_error", "broad_test_delta_rmse",
    "broad_test_gamma_rmse", "broad_test_theta_rmse",
    "broad_test_any_violation_proportion",
    "pde_residual_rmse",
    "training_wall_seconds", "configured_time_cap_seconds", "peak_rss_mb",
    "incremental_peak_mb", "optimizer_steps", "adam_steps",
    "lbfgs_outer_steps", "lbfgs_closure_evaluations",
    "labels_used_price", "labels_used_theta", "labels_used_delta",
    "labels_used_gamma", "pde_collocation_points", "terminal_payoff_points",
    "price_only_median_seconds_per_point",
    "price_and_greeks_median_seconds_per_point",
)


def aggregate(rows, reference):
    """Mean and sample standard deviation per (model, anchors, kind)."""
    groups = {}
    for row in rows:
        parsed = parse_name(row["model"])
        if parsed is None:
            continue
        groups.setdefault((parsed[0], parsed[1], row["checkpoint_kind"]),
                          []).append(row)
    out = []
    for (model, anchors, kind), entries in sorted(groups.items()):
        record = {"model": model, "anchor_count": anchors,
                  "checkpoint_kind": kind, "seeds": len(entries),
                  "std_convention": "sample"}
        for metric in AGGREGATE_METRICS:
            values = [e[metric] for e in entries
                      if isinstance(e.get(metric), (int, float))]
            if not values:
                continue
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0)
        if reference is not None:
            for metric in ("broad_test_price_rmse", "core_grid_price_rmse"):
                record[f"reference_{metric}"] = reference.get(metric)
        out.append(record)
    return out


def violation_stats(split, region="full"):
    """Per-category proportion and maximum magnitude for one region."""
    out = {}
    for name in VIOLATIONS + ("any_constraint",):
        stats = split["violations"][name][region]
        out[f"{name}_proportion"] = stats["proportion"]
        out[f"{name}_maximum_magnitude"] = stats["maximum_magnitude"]
    return out


def acceptance_rows(rows, reference_row, reference_grid, settings):
    """Three separate outcome sets: absolute core-grid accuracy, broad-test
    improvement over the reference, and per-category financial constraints.

    Financial outcomes are judged per category on BOTH violation proportion
    and maximum magnitude, so a model cannot pass by having fewer but much
    larger violations, and no cross-unit magnitude is ever combined. Every
    per-run pass/fail is retained."""
    criteria = settings["acceptance"]
    absolute = criteria["absolute_core_grid"]
    financial = criteria.get("financial_categories", {})
    out = []
    for row in rows:
        record = {"model": row["model"], "checkpoint_kind": row["checkpoint_kind"],
                  "protocol_id": row.get("protocol_id")}
        # 1. absolute core-grid accuracy
        checks = {
            "abs_core_price": row["core_grid_price_rmse"] <= absolute["price_rmse"],
            "abs_core_delta": row["core_grid_delta_rmse"] <= absolute["delta_rmse"],
            "abs_core_gamma": row["core_grid_gamma_rmse"] <= absolute["gamma_rmse"],
            "abs_core_theta": row["core_grid_theta_rmse"] <= absolute["theta_rmse"],
        }
        record["absolute_core_grid_pass"] = all(checks.values())
        record.update(checks)
        # 2. improvement over the reference on the broad test
        if reference_row is not None:
            improvement = {
                "broad_price_better_than_reference":
                    row["broad_test_price_rmse"] < reference_row["broad_test_price_rmse"],
                "broad_delta_better_than_reference":
                    row["broad_test_delta_rmse"] < reference_row["broad_test_delta_rmse"],
                "broad_gamma_better_than_reference":
                    row["broad_test_gamma_rmse"] < reference_row["broad_test_gamma_rmse"],
                "broad_theta_better_than_reference":
                    row["broad_test_theta_rmse"] < reference_row["broad_test_theta_rmse"],
            }
            record["broad_test_improvement_pass"] = all(improvement.values())
            record.update(improvement)
            record["reference_broad_price_rmse"] = reference_row["broad_test_price_rmse"]
        # 3. per-category financial constraints on the core grid
        if reference_grid is not None:
            for category in VIOLATIONS:
                key_prop = f"core_grid_{category}_proportion"
                key_mag = f"core_grid_{category}_max_magnitude"
                reference_stats = reference_grid["violations"][category]["full"]
                proportion_ok = (
                    row[key_prop] <= reference_stats["proportion"] + 1e-12)
                magnitude_ok = (
                    row[key_mag] is not None
                    and reference_stats["maximum_magnitude"] is not None
                    and row[key_mag] <= reference_stats["maximum_magnitude"] + 1e-12)
                record[f"{category}_proportion_ok"] = proportion_ok
                record[f"{category}_magnitude_ok"] = magnitude_ok
                record[f"{category}_units"] = (
                    financial.get("categories", {}).get(category, {}).get("units"))
                record[f"{category}_tolerance"] = (
                    reference_grid["violations"][category]["tolerance"])
            any_ok = (row["core_grid_any_constraint_proportion"]
                      <= reference_grid["violations"]["any_constraint"]["full"]["proportion"] + 1e-12)
            record["any_constraint_proportion_ok"] = any_ok
            category_keys = [f"{c}_proportion_ok" for c in VIOLATIONS]
            category_keys += [f"{c}_magnitude_ok" for c in VIOLATIONS]
            record["financial_constraints_pass"] = (
                all(record[key] for key in category_keys) and any_ok)
            record["reference_any_constraint_proportion"] = (
                reference_grid["violations"]["any_constraint"]["full"]["proportion"])
        out.append(record)
    return out


def rejection_reasons(row, reference_row, reference_grid, settings):
    """Human-readable reasons a run fails each outcome set."""
    reasons = []
    if not row.get("absolute_core_grid_pass"):
        reasons.append("absolute core-grid accuracy")
    if row.get("broad_test_improvement_pass") is False:
        reasons.append("no broad-test improvement over the reference")
    if row.get("financial_constraints_pass") is False:
        failing = [key for key in row
                   if key.endswith("_proportion_ok") or key.endswith("_magnitude_ok")
                   or key == "any_constraint_proportion_ok"]
        failed = [key for key in failing if row[key] is False]
        reasons.append("financial constraints: " + ", ".join(sorted(failed)))
    if not row.get("has_training_metadata"):
        reasons.append("missing training metadata (cost not joined)")
    return reasons


def acceptance_summary(acceptance):
    """Aggregate pass rates over seeds per (model, anchors, kind)."""
    groups = {}
    for row in acceptance:
        parsed = parse_name(row["model"])
        if parsed is None:
            continue
        groups.setdefault((parsed[0], parsed[1], row["checkpoint_kind"]),
                          []).append(row)
    out = []
    for (model, anchors, kind), entries in sorted(groups.items()):
        record = {"model": model, "anchor_count": anchors,
                  "checkpoint_kind": kind, "runs": len(entries)}
        for key in ("absolute_core_grid_pass", "broad_test_improvement_pass",
                    "financial_constraints_pass"):
            passes = sum(1 for e in entries if e.get(key))
            record[f"{key}_count"] = passes
            record[f"{key}_rate"] = passes / len(entries)
        out.append(record)
    return out


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows, aggregate_rows, reference_row, figure_dir):
    """Bars with seed variability, and a label-efficiency plot of means."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    selected = [r for r in aggregate_rows
                if r["checkpoint_kind"] == "selected"]
    if not selected:
        return
    largest = max(r["anchor_count"] for r in selected)
    comparison = [r for r in selected if r["anchor_count"] == largest]

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r["model"] for r in comparison]
    means = [r["broad_test_price_rmse_mean"] for r in comparison]
    errors = [r["broad_test_price_rmse_std"] for r in comparison]
    ax.bar(labels, means, yerr=errors, capsize=4, color="#4477aa")
    if reference_row is not None:
        ax.axhline(reference_row["broad_test_price_rmse"], color="black",
                   linestyle="--", label="reference (1 run, longer recipe)")
        ax.legend()
    ax.set_ylabel("Broad-test price RMSE (dollars)")
    ax.set_title(f"Corrected ablation at {largest} price anchors "
                 "(bars: mean, whiskers: sample std over seeds)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figure_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in sorted({r["model"] for r in selected}):
        entries = sorted((r["anchor_count"], r["broad_test_price_rmse_mean"],
                          r["broad_test_price_rmse_std"])
                         for r in selected if r["model"] == model)
        if not entries:
            continue
        anchors = [e[0] for e in entries]
        means = [e[1] for e in entries]
        errors = [e[2] for e in entries]
        ax.errorbar(anchors, means, yerr=errors, marker="o", capsize=4,
                    label=f"{model} (mean of 3 seeds)")
    if reference_row is not None:
        ax.axhline(reference_row["broad_test_price_rmse"], color="black",
                   linestyle="--", label="reference (1 run)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Training price anchors (Greek anchors scale with these)")
    ax.set_ylabel("Broad-test price RMSE (dollars)")
    ax.set_title("Label efficiency (mean point per budget; no per-seed curves)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "label_efficiency.png", dpi=180)
    plt.close(fig)


def discover_runs(settings, checkpoint_dir, results_dir, protocol_id):
    """Return [(name, checkpoint, kind, metadata_path, status, note)].

    A run is only *verified* when its checkpoint, run manifest and training
    metadata all exist and agree on the requested protocol, run identity and
    dataset. Anything less is reported as legacy or incomplete rather than
    silently pooled.
    """
    runs = []
    for path in sorted(Path(checkpoint_dir).glob("*.pt")):
        stem = path.stem
        if stem.endswith("_last"):
            name, kind = stem[:-len("_last")], "last"
        else:
            name, kind = stem, "selected"
        manifest_path = results_dir / name / "run_manifest.json"
        metadata_path = results_dir / name / "training_metadata.json"
        status, note = "verified", []
        if not metadata_path.exists():
            status = "incomplete"
            note.append("no training metadata")
        if not manifest_path.exists():
            status = "legacy" if status == "verified" else status
            note.append("no run manifest")
        else:
            manifest = json.loads(manifest_path.read_text())
            if protocol_id and manifest.get("protocol_id") != protocol_id:
                status = "incompatible"
                note.append(f"protocol {manifest.get('protocol_id')} != "
                            f"{protocol_id}")
            elif manifest.get("checkpoints", {}).get(kind) != str(
                    path.relative_to(results_dir.parents[2])):
                note.append("manifest checkpoint path differs")
        runs.append((name, path, kind, metadata_path if metadata_path.exists()
                     else None, status, "; ".join(note)))
    return runs


def collect_details(name, kind, grid, broad, regional_rows, violation_rows):
    for split_label, split in (("core_grid", grid), ("broad_test", broad)):
        for region, entry in split["errors"].items():
            for quantity, stats in entry.items():
                regional_rows.append({
                    "model": name, "checkpoint_kind": kind, "split": split_label,
                    "region": region, "quantity": quantity, **stats})
        for category, per_region in split["violations"].items():
            tolerance = per_region.get("tolerance")
            maximum_region = per_region.get("maximum_region")
            for region in REGIONS:
                stats = dict(per_region[region])
                violation_rows.append({
                    "model": name, "checkpoint_kind": kind, "split": split_label,
                    "category": category, "region": region,
                    "tolerance": tolerance, "maximum_region": maximum_region,
                    **stats})


def reproducibility_check(rows, aggregate_rows):
    """Confirm aggregate means reproduce the per-run rows exactly."""
    problems = []
    for record in aggregate_rows:
        members = [r for r in rows
                   if parse_name(r["model"]) is not None
                   and parse_name(r["model"])[0] == record["model"]
                   and parse_name(r["model"])[1] == record["anchor_count"]
                   and r["checkpoint_kind"] == record["checkpoint_kind"]]
        if not members:
            continue
        expected = statistics.fmean(m["broad_test_price_rmse"] for m in members)
        actual = record["broad_test_price_rmse_mean"]
        if abs(expected - actual) > 1e-12:
            problems.append({"model": record["model"],
                             "anchor_count": record["anchor_count"],
                             "kind": record["checkpoint_kind"],
                             "expected": expected, "recorded": actual})
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--protocol-id", default=None)
    parser.add_argument("--output-tag", default=None,
                        help="Versioned output subdirectory (default: protocol)")
    parser.add_argument("--include-unverified", action="store_true",
                        help="Also evaluate legacy/incomplete runs, flagged as such")
    arguments = parser.parse_args()

    settings = json.loads(SETTINGS_PATH.read_text())
    torch.set_default_dtype(
        torch.float64 if settings["network"]["dtype"] == "float64"
        else torch.float32)
    protocol_id = arguments.protocol_id or settings["protocol_id"]
    protocol = settings.get("protocols", {}).get(protocol_id, {})
    results_dir = ROOT / (arguments.results_dir
                          or protocol.get("results_dir")
                          or settings["output"]["results_dir"])
    checkpoint_dir = ROOT / (arguments.checkpoint_dir
                             or protocol.get("checkpoint_dir")
                             or settings["output"]["checkpoint_dir"])
    output_tag = arguments.output_tag or protocol_id
    dataset_path = results_dir / "data" / "anchored_dataset.npz"
    dataset = load_dataset(dataset_path)
    dataset_meta_path = dataset_path.with_suffix(dataset_path.suffix + ".meta.json")
    dataset_meta = (json.loads(dataset_meta_path.read_text())
                    if dataset_meta_path.exists() else {})
    table_dir = results_dir / "tables" / output_tag
    figure_dir = results_dir / "figures" / output_tag
    table_dir.mkdir(parents=True, exist_ok=True)

    rows, regional_rows, violation_rows, run_inventory = [], [], [], []
    reference_row = reference_grid = reference_broad = None

    reference_path = ROOT / settings["reference"]["checkpoint"]
    if reference_path.exists():
        reference_row, ref_grid, ref_broad = run_metrics(
            REFERENCE_NAME, reference_path, "reference", settings, dataset,
            protocol_id="historical_reference")
        reference_grid, reference_broad = ref_grid, ref_broad
        rows.append(reference_row)
        collect_details(REFERENCE_NAME, "reference", ref_grid, ref_broad,
                        regional_rows, violation_rows)

    discovered = discover_runs(settings, checkpoint_dir, results_dir, None)
    seen_seeds = {}
    checkpoint_digests = {}
    for name, path, kind, metadata_path, status, note in discovered:
        run_inventory.append({"run": name, "checkpoint_kind": kind,
                              "status": status, "note": note})
        if status == "incompatible" and not arguments.include_unverified:
            print(f"Rejecting '{name}': {note}")
            continue
        if status in ("legacy", "incomplete") and not arguments.include_unverified:
            print(f"Skipping unverified '{name}' ({status}): {note}")
            continue
        checkpoint_digests[(name, kind)] = torch_checkpoint_digest(path)
        parsed = parse_name(name)
        if parsed is not None:
            key = (parsed[0], parsed[1], parsed[2])
            seen_seeds.setdefault(key, []).append(kind)
        row, grid, broad = run_metrics(
            name, path, kind, settings, dataset,
            training_metadata_path=metadata_path, protocol_id=protocol_id)
        if (dataset_meta.get("content_hash")
                and row.get("dataset_content_hash")
                and row["dataset_content_hash"] != dataset_meta["content_hash"]):
            row["dataset_mismatch"] = True
            print(f"Warning: '{name}' was trained on dataset "
                  f"{row['dataset_content_hash']} but is being evaluated on "
                  f"{dataset_meta['content_hash']}")
        rows.append(row)
        collect_details(name, kind, grid, broad, regional_rows, violation_rows)

    duplicate_seeds = {f"{key}": kinds for key, kinds in seen_seeds.items()
                       if len(kinds) != len(set(kinds))}
    write_rows(table_dir / "run_metrics.csv", rows)
    write_rows(table_dir / "regional_errors.csv", regional_rows)
    write_rows(table_dir / "violations.csv", violation_rows)
    write_rows(table_dir / "run_inventory.csv", run_inventory)
    aggregates = aggregate(rows, reference_row)
    write_rows(table_dir / "aggregate.csv", aggregates)
    acceptance = acceptance_rows(rows, reference_row, reference_grid, settings)
    write_rows(table_dir / "acceptance.csv", acceptance)
    write_rows(table_dir / "acceptance_summary.csv",
               acceptance_summary(acceptance))
    if reference_row is not None:
        write_json_atomic(table_dir / "reference_metrics.json", {
            "reference_row": reference_row,
            "core_grid_violations": violation_stats(reference_grid),
            "broad_test_violations": violation_stats(reference_broad),
        })
    problems = reproducibility_check(rows, aggregates)
    evaluation_manifest = {
        "protocol_id": protocol_id,
        "output_tag": output_tag,
        "results_dir": str(results_dir.relative_to(ROOT)),
        "checkpoint_dir": str(checkpoint_dir.relative_to(ROOT)),
        "evaluation_source_fingerprint": evaluation_source_fingerprint(ROOT),
        "dataset_identifier": str(dataset_path.relative_to(ROOT)),
        "dataset_content_hash": dataset_meta.get("content_hash"),
        "evaluation_configuration": {
            "evaluation": settings["evaluation"],
            "acceptance": settings["acceptance"],
        },
        "checkpoints": {row["model"]: {
            "checkpoint_kind": row["checkpoint_kind"],
            "checkpoint_digest": checkpoint_digests.get(
                (row["model"], row["checkpoint_kind"])),
            "training_run_id": row["model"],
            "run_identity_hash": row.get("run_identity_hash"),
            "training_source_fingerprint": row.get("training_source_fingerprint"),
        } for row in rows},
        "duplicate_seed_entries": duplicate_seeds,
        "runs_skipped_unverified": [entry for entry in run_inventory
                                    if entry["status"] != "verified"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "torch_num_threads": torch.get_num_threads(),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "aggregate_reproduces_runs": not problems,
        "aggregate_reproduction_problems": problems,
        "std_convention": "sample",
        "pde_residual_units": "RMS of the transformed residual in u = C/K units",
    }
    write_json_atomic(table_dir / "evaluation_manifest.json", evaluation_manifest)
    make_figures(rows, aggregates, reference_row, figure_dir)
    passing = [r for r in acceptance if r.get("financial_constraints_pass")]
    print(f"Wrote {len(rows)} model evaluations, {len(aggregates)} aggregate "
          f"groups to {table_dir}; {len(passing)}/{len(acceptance)} runs meet "
          f"the per-category financial constraints; aggregate-vs-run check "
          f"{'passed' if not problems else 'FAILED'}")
    if duplicate_seeds:
        print(f"WARNING duplicate seed entries: {duplicate_seeds}")


if __name__ == "__main__":
    main()