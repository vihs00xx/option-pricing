"""Run identity, provenance and atomic output helpers.

Two fingerprints are kept deliberately separate:

* ``training_source_fingerprint`` covers the files that define how a
  checkpoint is produced (network, dataset, trainer, settings).
* ``evaluation_source_fingerprint`` covers the files that define how a
  checkpoint is scored.

Editing plotting code therefore changes the evaluation identity without
invalidating a trained checkpoint, which is the property the earlier
combined fingerprint lacked.

A *run identity* is the complete, hashable description of everything that
determines a trained checkpoint: effective configuration, variant,
initialisation seed, architecture and dtype, dataset content hash, the
exact training subset, the normalisation policy and values, the fixed
validation selection policy, optimizer settings including L-BFGS
tolerances, the training-source fingerprint and execution settings.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


TRAINING_SOURCE_FILES = (
    "src/neural/fixed_pinn.py",
    "src/neural/anchor_data.py",
    "src/neural/anchored_pinn.py",
    "src/neural/memory.py",
    "experiments/pinn/train_anchored_ablation.py",
    "experiments/config/neural/anchored_pinn_settings.json",
)

EVALUATION_SOURCE_FILES = (
    "src/neural/fixed_pinn.py",
    "src/neural/anchor_data.py",
    "src/neural/anchored_pinn.py",
    "src/neural/pinn_diagnostics.py",
    "experiments/pinn/evaluate_anchored_ablation.py",
    "experiments/config/neural/anchored_pinn_settings.json",
)


def _digest():
    return hashlib.sha256()


def _file_digest(path):
    digest = _digest()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def source_fingerprint(files, root):
    """Fingerprint of the named source files that exist."""
    digest = _digest()
    hashed = []
    for name in files:
        path = Path(root) / name
        if path.exists():
            digest.update(name.encode())
            digest.update(_file_digest(path).encode())
            hashed.append(name)
    return {"files": hashed, "digest": digest.hexdigest()[:16]}


def training_source_fingerprint(root):
    return source_fingerprint(TRAINING_SOURCE_FILES, root)


def evaluation_source_fingerprint(root):
    return source_fingerprint(EVALUATION_SOURCE_FILES, root)


def array_digest(array):
    array = np.ascontiguousarray(np.asarray(array))
    digest = _digest()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def training_subset_identity(subset):
    """Exact identity of the training subset handed to a run."""
    arrays = {}
    for name in ("anchor_y", "anchor_tau", "anchor_price",
                 "greek_y", "greek_tau", "greek_delta", "greek_gamma",
                 "greek_theta", "pde_y", "pde_tau",
                 "terminal_y", "terminal_tau", "terminal_payoff"):
        if name in subset:
            arrays[name] = array_digest(subset[name])
    component_counts = {}
    if "anchor_component" in subset:
        component_counts = {
            key: int(np.sum(subset["anchor_component"] == index))
            for index, key in enumerate(
                ("uniform", "near_strike", "near_expiry", "high_gamma",
                 "boundaries"))
        }
    return {
        "arrays": arrays,
        "component_counts": component_counts,
        "digest": hashlib.sha256(
            json.dumps(arrays, sort_keys=True).encode()).hexdigest()[:16],
    }


def run_identity(*, settings, model_variant, seed, subset,
                 dataset_identifier, dataset_content_hash,
                 label_normalization_policy, label_normalization_values,
                 selection_policy, stages, training_fingerprint,
                 execution_settings):
    """Complete run identity plus a short digest."""
    architecture = dict(settings["network"])
    identity = {
        "effective_configuration": {
            key: settings[key]
            for key in ("problem", "network", "minimum_tau", "weight_min",
                        "weight_max", "scale_epsilon", "adam_learning_rate",
                        "validation_every", "lbfgs_history_size",
                        "lbfgs_tolerance_grad", "lbfgs_tolerance_change",
                        "lbfgs_validation_every",
                        "wall_clock_cap_seconds", "greek_anchor_divisor",
                        "greek_anchor_max", "density", "greek_density")
            if key in settings
        },
        "model_variant": model_variant,
        "permitted_terms": list(settings.get("permitted_terms", ())),
        "initialisation_seed": int(seed),
        "architecture": architecture,
        "dtype": architecture.get("dtype"),
        "dataset_content_hash": dataset_content_hash,
        "dataset_identifier": dataset_identifier,
        "training_subset_identity": training_subset_identity,
        "label_normalization_policy": label_normalization_policy,
        "label_normalization_values": label_normalization_values,
        "validation_selection_policy": selection_policy,
        "stages": stages,
        "optimizer_settings": {
            "adam_learning_rate": settings.get("adam_learning_rate"),
            "lbfgs_history_size": settings.get("lbfgs_history_size"),
            "lbfgs_tolerance_grad": settings.get("lbfgs_tolerance_grad"),
            "lbfgs_tolerance_change": settings.get("lbfgs_tolerance_change"),
            "lbfgs_validation_every": settings.get("lbfgs_validation_every"),
            "wall_clock_cap_seconds": settings.get("wall_clock_cap_seconds"),
            "validation_every": settings.get("validation_every"),
        },
        "training_source_fingerprint": training_fingerprint,
        "execution_settings": execution_settings,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return {"identity": identity, "run_identity_hash": digest}


def write_json_atomic(path, payload):
    """Write JSON through a temp file and an atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)
    return path


def write_torch_atomic(path, payload):
    """Save a torch payload through a temp file and an atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def verify_run_manifest(manifest, expected_hash, required_checkpoints,
                        root):
    """Return a list of reasons a stored run cannot be reused.

    A reusable run needs a matching identity hash, a manifest that agrees
    with the checkpoint files it claims, and every checkpoint present.
    """
    reasons = []
    if not manifest:
        return ["no manifest"]
    if manifest.get("run_identity_hash") != expected_hash:
        reasons.append(
            f"run identity {manifest.get('run_identity_hash')} != "
            f"requested {expected_hash}")
    paths = manifest.get("checkpoints", {})
    for kind, relative in paths.items():
        path = Path(root) / relative
        if not path.exists():
            reasons.append(f"missing {kind} checkpoint {relative}")
    if "selected" not in paths:
        reasons.append("manifest records no selected checkpoint")
    return reasons


def verify_checkpoint_metadata(manifest_path, manifest):
    """Confirm the manifest on disk agrees with the run manifest."""
    if not Path(manifest_path).exists():
        return ["missing run manifest"]
    stored = json.loads(Path(manifest_path).read_text())
    return [
        f"{key} differs ({stored.get(key)!r} != {manifest.get(key)!r})"
        for key in ("run_identity_hash", "model_variant", "seed")
        if stored.get(key) != manifest.get(key)
    ]


def torch_checkpoint_digest(path):
    """Stable digest of a checkpoint file, for the evaluation manifest."""
    return _file_digest(path)
