"""Evaluate a DeepONet checkpoint against the dataset recorded for that run.

The dataset is resolved from the run's own metadata (schema v2 or the legacy
adapter), never from inherited configuration. File SHA-256 and array-content
SHA-256 are verified separately and never interchangeably. When a historical
record cannot verify its dataset or checkpoint, evaluation fails with an
actionable error instead of bypassing verification. The held-out benchmark is
gated behind --final-test and is never used for model selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.neural import run_metadata as runs
from src.neural.deeponet import load_deeponet


ROOT = Path(__file__).resolve().parents[2]
QUANTITIES = ("price", "delta", "gamma", "vega", "theta", "rho")
SETTINGS = json.loads((ROOT / "experiments/config/neural/deeponet_training.json").read_text())
FROZEN_DATASET = ROOT / SETTINGS["dataset"]
FROZEN_MANIFEST = ROOT / SETTINGS["dataset_manifest"]


def trusted_file_sha256(dataset_path):
    """Trusted full-file hash from the frozen pre-training manifest, when the run
    used the frozen dataset. None otherwise: unverifiable splits are rejected."""
    if Path(dataset_path).resolve() != FROZEN_DATASET.resolve():
        return None
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    return manifest.get("dataset_sha256")


def evaluate_checkpoint(checkpoint, final_test=False, output=None):
    checkpoint = Path(checkpoint).resolve()
    folder = checkpoint.parent
    view = runs.load_run_metadata(folder)
    runs.verify_checkpoint(checkpoint, view, checkpoint.stem)
    dataset_path = ROOT / view["dataset"]["path"]
    split = "test" if final_test else "validation"
    checked_hashes = runs.verify_dataset(dataset_path, view["dataset"], evaluated_split=split,
                                         trusted_file_sha256=trusted_file_sha256(dataset_path))
    model = load_deeponet(checkpoint)
    with np.load(dataset_path, allow_pickle=False) as data:
        branch = torch.tensor(data[f"{split}_branch"], dtype=torch.float64)
        trunk = torch.tensor(data[f"{split}_trunk"], dtype=torch.float64)
        reference = data[f"{split}_targets"]
        groups = data["test_benchmark_split"] if final_test else np.full(len(branch), "validation")
    prediction = model.price_and_greeks(branch, trunk)
    metrics = {}
    for group in ["all", *sorted(set(groups))]:
        mask = np.ones(len(branch), dtype=bool) if group == "all" else groups == group
        metrics[group] = {"points": int(mask.sum())}
        for index, name in enumerate(QUANTITIES):
            error = prediction[name].numpy()[mask] - reference[mask, index]
            metrics[group][name] = {"rmse": float(np.sqrt(np.mean(error ** 2))),
                                    "max_absolute_error": float(np.max(np.abs(error)))}
    result = {"split": split, "checkpoint": str(checkpoint),
              "checkpoint_sha256": runs.sha256(checkpoint),
              "metadata_schema": view["schema"],
              "dataset": {**view["dataset"], "verified_hash_kinds": checked_hashes},
              "final_test_accessed": bool(final_test),
              "metrics": metrics}
    if output is not None:
        runs.save_json_atomic(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--final-test", action="store_true",
                        help="Explicitly unlock the held-out benchmark after model selection.")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON output path (stdout always receives the result).")
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, final_test=args.final_test, output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
