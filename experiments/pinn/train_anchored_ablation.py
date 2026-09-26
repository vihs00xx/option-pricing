"""Train the anchored PINN variants on one fixed, versioned dataset.

Usage:
    python -m experiments.pinn.train_anchored_ablation --dataset-only
    python -m experiments.pinn.train_anchored_ablation --models hybrid v2
    python -m experiments.pinn.train_anchored_ablation --smoke --models v2

Outputs are versioned (see ``output`` in the settings) so corrected runs
never mix with the earlier defective study. The historical best
checkpoint is read-only and is never written here.

Each run records the *effective* configuration (settings plus any
command-line overrides), its hash, and the dataset identifier. A
checkpoint is reused only when its stored effective-configuration hash
matches, so a stale file with the same name is not silently trusted.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.neural.anchor_data import (
    build_dataset, config_hash, load_dataset, problem_from, reference_scales,
    save_dataset, subset, verify_disjoint, verify_domain,
)
from src.neural.anchored_pinn import MODES, AnchoredTrainer
from src.neural.fixed_pinn import FixedBlackScholesPINN
from src.neural.provenance import (
    run_identity, training_source_fingerprint, write_json_atomic,
    write_torch_atomic, verify_run_manifest, torch_checkpoint_digest,
)


ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "experiments" / "config" / "neural" / "anchored_pinn_settings.json"
NORMALIZATION_POLICY = (
    "supervised terms divided by (RMS of the full training labels)^2 + "
    "scale_epsilon; scales computed once from the largest training set so "
    "the anchor-budget sweep does not change the objective")
HISTORY_FIELDS = (
    "stage", "phase", "step", "selection_score", "stage_loss",
    "closure_calls", "accepted_steps",
    "train_price", "train_theta", "train_delta", "train_gamma",
    "train_pde", "train_terminal", "validation_price", "validation_theta",
    "validation_delta", "validation_gamma", "validation_pde",
    "validation_terminal", "elapsed_seconds",
)


def stable_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def git_revision():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, timeout=10).stdout.strip())
        return {"revision": revision or None, "uncommitted_changes": dirty}
    except Exception:  # noqa: BLE001 - provenance is best effort
        return {"revision": None, "uncommitted_changes": None}


def source_fingerprint():
    """Hash of the relevant source files plus the settings, so uncommitted
    edits are part of the recorded identity of a run."""
    relevant = [
        "src/neural/fixed_pinn.py", "src/neural/anchor_data.py",
        "src/neural/anchored_pinn.py", "src/neural/memory.py",
        "src/neural/pinn_diagnostics.py",
        "experiments/pinn/train_anchored_ablation.py",
        "experiments/pinn/evaluate_anchored_ablation.py",
        "experiments/config/neural/anchored_pinn_settings.json",
    ]
    digest = hashlib.sha256()
    present = []
    for name in relevant:
        path = ROOT / name
        if not path.exists():
            continue
        digest.update(name.encode())
        digest.update(path.read_bytes())
        present.append(name)
    return {"files": present, "digest": digest.hexdigest()[:16]}


def environment_provenance():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "git": git_revision(),
        "source_fingerprint": source_fingerprint(),
    }


def superseded_run_config(settings, mode, anchor_count, seed, stages, selection):
    """Superseded by ``run_identity``: reuse now compares the complete run
    identity, not just this partial configuration. Kept only so historical
    helper imports do not break."""
    return {
        "study_version": settings["study_version"],
        "model": mode,
        "anchor_count": anchor_count,
        "seed": seed,
        "stages": stages,
        "selection": selection,
    }


def dataset_paths(settings, smoke):
    base = ROOT / settings["output"]["results_dir"] / "data"
    name = "anchored_dataset_smoke.npz" if smoke else "anchored_dataset.npz"
    return base / name


def load_or_build_dataset(settings, smoke):
    """Return ``(dataset, path, seconds)``; timing is label-generation cost."""
    path = dataset_paths(settings, smoke)
    start = perf_counter()
    if path.exists():
        dataset = load_dataset(path, settings)
    else:
        dataset = build_dataset(settings)
        metadata = save_dataset(path, dataset, settings)
        print(json.dumps({"built_dataset": str(path.relative_to(ROOT)),
                          "config_hash": metadata["config_hash"],
                          "content_hash": metadata["content_hash"]}, indent=2))
    return dataset, path, perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None,
                        choices=list(MODES.keys()))
    parser.add_argument("--anchor-counts", nargs="*", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--tag", default="", help="Suffix for run names")
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny budget; separate dataset path")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if a matching checkpoint exists")
    parser.add_argument("--wall-clock-cap", type=float, default=None,
                        help="Override the per-run wall-clock cap (seconds)")
    parser.add_argument("--protocol-id", default=None,
                        help="Protocol identifier shared by comparable runs")
    arguments = parser.parse_args()

    settings = json.loads(SETTINGS_PATH.read_text())
    if arguments.smoke:
        settings["anchor_counts"] = [256]
        settings["greek_anchor_max"] = 64
        settings["validation_points"] = 256
        settings["test_points"] = 256
        settings["pde_points"] = 256
        settings["terminal_points"] = 256
        settings["validation_every"] = 2
        settings["wall_clock_cap_seconds"] = 120
        for stage in settings["stages"]:
            stage["adam_epochs"] = 4
            stage["lbfgs_iterations"] = 5
    if arguments.wall_clock_cap is not None:
        settings["wall_clock_cap_seconds"] = arguments.wall_clock_cap

    data_settings = settings  # dataset config == settings after overrides
    dataset, dataset_file, label_generation_seconds = load_or_build_dataset(
        data_settings, arguments.smoke)
    problems = verify_domain(dataset, settings)
    overlaps = verify_disjoint(dataset)
    report = {
        "dataset": str(dataset_file.relative_to(ROOT)),
        "data_config_hash": config_hash(settings),
        "counts": {name: int(dataset[name].size) for name in
                   ("anchor_y", "greek_y", "validation_y", "validation_pde_y",
                    "validation_terminal_y", "test_y", "pde_y", "terminal_y")},
        "domain_issues": problems,
        "coordinate_overlaps": overlaps,
    }
    print(json.dumps(report, indent=2))
    if problems or overlaps:
        raise ValueError("dataset failed verification; see report above")
    if arguments.dataset_only:
        return

    models = arguments.models or settings["models"]
    anchor_counts = arguments.anchor_counts or settings["anchor_counts"]
    seeds = arguments.seeds or settings["seeds"]
    network = settings["network"]
    dtype = torch.float64 if network["dtype"] == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    environment = environment_provenance()
    dataset_meta = json.loads(
        dataset_file.with_suffix(dataset_file.suffix + ".meta.json").read_text())
    results_dir = ROOT / settings["output"]["results_dir"]
    checkpoint_dir = ROOT / settings["output"]["checkpoint_dir"]

    training_fingerprint = training_source_fingerprint(ROOT)
    execution_settings = {
        "device": "cpu",
        "dtype": settings["network"]["dtype"],
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    protocol_id = arguments.protocol_id or settings.get(
        "protocol_id", f"study_v{settings['study_version']}")
    write_json_atomic(results_dir / "protocol_manifest.json", {
        "protocol_id": protocol_id,
        "dataset_identifier": str(dataset_file.relative_to(ROOT)),
        "dataset_content_hash": dataset_meta["content_hash"],
        "data_config_hash": dataset_meta["config_hash"],
        "dataset_schema_version": dataset_meta["schema_version"],
        "architecture": settings["network"],
        "stages": settings["stages"],
        "selection": settings["selection"],
        "wall_clock_cap_seconds": settings["wall_clock_cap_seconds"],
        "label_normalization_policy": NORMALIZATION_POLICY,
        "label_generation_seconds": label_generation_seconds,
        "dataset_generation_amortization": (
            "label-generation cost is recorded once here; the timing shown in "
            "that field is dataset build-or-load, so a reused dataset reports "
            "load time, not generation time"),
        "intended_differences": {
            "hybrid vs supervised_hybrid_control":
                "PDE residual only; identical labels, payoff and selection",
            "v2 vs supervised_v2_control":
                "PDE residual only; identical labels, payoff and selection",
            "v2 vs hybrid":
                "extra Delta/Gamma supervision; not a physics comparison",
        },
        "timing_policy": ("fixed declared optimizer schedule with a wall-clock "
                          "cap; not an equal-compute comparison"),
        "training_source_fingerprint": training_fingerprint,
        "execution_settings": execution_settings,
        "seeds": list(seeds),
    })

    for mode in models:
        for anchor_count in anchor_counts:
            for seed in seeds:
                name = f"{mode}_n{anchor_count}_s{seed}"
                if arguments.tag:
                    name = f"{name}_{arguments.tag}"
                subset_sample = subset(dataset, anchor_count, settings)
                scales = reference_scales(dataset)
                identity = run_identity(
                    settings={**settings, "permitted_terms": list(MODES[mode])},
                    model_variant=mode, seed=seed, subset=subset_sample,
                    dataset_identifier=str(dataset_file.relative_to(ROOT)),
                    dataset_content_hash=dataset_meta["content_hash"],
                    label_normalization_policy=NORMALIZATION_POLICY,
                    label_normalization_values=scales,
                    selection_policy=settings["selection"][mode],
                    stages=settings["stages"],
                    training_fingerprint=training_fingerprint,
                    execution_settings=execution_settings)
                identity_hash = identity["run_identity_hash"]
                run_dir = results_dir / name
                manifest_path = run_dir / "run_manifest.json"
                metadata_path = run_dir / "training_metadata.json"
                selected_path = checkpoint_dir / f"{name}.pt"
                last_path = checkpoint_dir / f"{name}_last.pt"

                if manifest_path.exists() and not arguments.force:
                    stored = json.loads(manifest_path.read_text())
                    reasons = verify_run_manifest(
                        stored, identity_hash, ["selected", "last"], ROOT)
                    if not reasons:
                        print(f"Reusing '{name}': identity "
                              f"{identity_hash} matches and both checkpoints "
                              "and the manifest agree")
                        continue
                    # Do not silently retrain into the same path: write the
                    # new revision under its own run identity.
                    print(f"Not reusing '{name}': {'; '.join(reasons)}. "
                          f"Writing revision {identity_hash[:6]} instead.")
                    name = f"{name}__rev{identity_hash[:6]}"
                    run_dir = results_dir / name
                    manifest_path = run_dir / "run_manifest.json"
                    metadata_path = run_dir / "training_metadata.json"
                    selected_path = checkpoint_dir / f"{name}.pt"
                    last_path = checkpoint_dir / f"{name}_last.pt"
                elif (selected_path.exists() or metadata_path.exists()) \
                        and not arguments.force:
                    print(f"Not reusing '{name}': checkpoint or metadata "
                          "exists without a run manifest; writing revision "
                          f"{identity_hash[:6]} instead.")
                    name = f"{name}__rev{identity_hash[:6]}"
                    run_dir = results_dir / name
                    manifest_path = run_dir / "run_manifest.json"
                    metadata_path = run_dir / "training_metadata.json"
                    selected_path = checkpoint_dir / f"{name}.pt"
                    last_path = checkpoint_dir / f"{name}_last.pt"

                torch.manual_seed(seed)
                problem = problem_from(settings)
                model = FixedBlackScholesPINN(
                    problem, hidden_layers=network["hidden_layers"],
                    hidden_width=network["hidden_width"])
                sample = subset_sample
                scales = reference_scales(dataset)
                trainer = AnchoredTrainer(model, sample, scales, settings,
                                          mode, seed, dtype=dtype)
                print(f"Training '{name}' "
                      f"(price anchors={len(sample['anchor_y'])}, "
                      f"greek anchors={len(sample['greek_y'])})")
                history, metadata, states = trainer.train(
                    settings["stages"],
                    wall_clock_cap=settings["wall_clock_cap_seconds"], log=print)
                permitted = set(MODES[mode])
                selection_terms = set(settings["selection"][mode])
                metadata.update({
                    "training_run_id": f"{name}_seed_{seed}",
                    "run_identity": identity["identity"],
                    "run_identity_hash": identity_hash,
                    "protocol_id": protocol_id,
                    "data_config_hash": dataset_meta["config_hash"],
                    "dataset_file": str(dataset_file.relative_to(ROOT)),
                    "dataset_content_hash": dataset_meta["content_hash"],
                    "dataset_schema_version": dataset_meta["schema_version"],
                    "effective_settings": settings,
                    "environment": environment,
                    "anchor_count": int(anchor_count),
                    "greek_anchor_count": int(len(sample["greek_y"])),
                    # Labels present in the saved dataset (available, not used).
                    "labels_available": dataset_meta["label_counts"],
                    # Labels actually consumed by this variant's permitted
                    # terms: physics_only consumes none of them, whatever its
                    # filename says.
                    "labels_used_by_permitted_terms": {
                        "training_price": (int(len(sample["anchor_y"]))
                                           if "price" in permitted else 0),
                        "training_theta": (int(len(sample["greek_y"]))
                                           if "theta" in permitted else 0),
                        "training_delta": (int(len(sample["greek_y"]))
                                           if "delta" in permitted else 0),
                        "training_gamma": (int(len(sample["greek_y"]))
                                           if "gamma" in permitted else 0),
                    },
                    "validation_labels_used_by_selection": {
                        "validation_price": (int(len(sample["validation_y"]))
                                             if "price" in selection_terms else 0),
                        "validation_theta": (int(len(sample["validation_y"]))
                                             if "theta" in selection_terms else 0),
                        "validation_delta": (int(len(sample["validation_y"]))
                                             if "delta" in selection_terms else 0),
                        "validation_gamma": (int(len(sample["validation_y"]))
                                             if "gamma" in selection_terms else 0),
                    },
                    "samples_used": {
                        "price_anchor_coordinates": int(len(sample["anchor_y"])),
                        "greek_anchor_coordinates": int(len(sample["greek_y"])),
                        "pde_collocation": (int(len(sample["pde_y"]))
                                            if "pde" in permitted else 0),
                        "terminal_payoff": (int(len(sample["terminal_y"]))
                                            if "terminal" in permitted else 0),
                    },
                    "label_generation_seconds": label_generation_seconds,
                    "label_scales": scales,
                })

                components = {}
                for kind, key in (("anchor", "anchor_component"),
                                  ("greek", "greek_component")):
                    counts = sample[key]
                    components[kind] = {
                        name_: int(np.sum(counts == index))
                        for index, name_ in enumerate(
                            ("uniform", "near_strike", "near_expiry",
                             "high_gamma", "boundaries"))
                    }
                metadata["component_counts_used"] = components

                run_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "problem": asdict(problem),
                    "network": {"hidden_layers": network["hidden_layers"],
                                "hidden_width": network["hidden_width"]},
                    "model": mode,
                    "seed": seed,
                    "anchor_count": anchor_count,
                    "run_identity_hash": identity_hash,
                    "protocol_id": protocol_id,
                    "dataset_content_hash": dataset_meta["content_hash"],
                    "training_source_fingerprint":
                        training_fingerprint["digest"],
                }
                # Atomic: an interrupted write leaves no checkpoint that
                # could be mistaken for complete.
                write_torch_atomic(selected_path,
                                   {**payload, "state_dict": states["selected"],
                                    "checkpoint_kind": "selected"})
                write_torch_atomic(last_path,
                                   {**payload, "state_dict": states["last"],
                                    "checkpoint_kind": "last"})
                metadata["checkpoints"] = {
                    "selected": str(selected_path.relative_to(ROOT)),
                    "last": str(last_path.relative_to(ROOT)),
                }
                metadata["checkpoint_digests"] = {
                    "selected": torch_checkpoint_digest(selected_path),
                    "last": torch_checkpoint_digest(last_path),
                }
                write_json_atomic(manifest_path, {
                    "run_id": name,
                    "protocol_id": protocol_id,
                    "model_variant": mode,
                    "seed": seed,
                    "anchor_count": anchor_count,
                    "run_identity_hash": identity_hash,
                    "run_identity": identity["identity"],
                    "dataset_identifier": str(dataset_file.relative_to(ROOT)),
                    "dataset_content_hash": dataset_meta["content_hash"],
                    "training_source_fingerprint": training_fingerprint,
                    "execution_settings": execution_settings,
                    "checkpoints": metadata["checkpoints"],
                    "checkpoint_digests": metadata["checkpoint_digests"],
                    "training_metadata": str(metadata_path.relative_to(ROOT)),
                })
                metadata["selected_checkpoint"] = str(selected_path.relative_to(ROOT))
                metadata["last_checkpoint"] = str(last_path.relative_to(ROOT))
                with (run_dir / "training_history.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
                    writer.writeheader()
                    for row in history:
                        writer.writerow({field: row.get(field, "")
                                         for field in HISTORY_FIELDS})
                write_json_atomic(metadata_path, metadata)
                print(json.dumps({
                    "run": name,
                    "selected_label": metadata["selected_label"],
                    "selected_score": metadata["selected_selection_score"],
                    "last_score": metadata["last_selection_score"],
                    "wall_seconds": metadata["wall_seconds"],
                    "peak_rss_mb": metadata["peak_rss_mb"],
                }, indent=2))


if __name__ == "__main__":
    main()
