"""Check the saved neural study: hashes, then re-evaluate the selected checkpoint without retraining."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.neural.evaluate import LEVELS, PANELS, RESULTS, SETTINGS, load_model, panels
from src.neural.reference import QUANTITIES, price_and_greeks
from src.neural.training import quantity_metrics

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_default_dtype(torch.float64)
    protocol = json.loads((RESULTS/"protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULTS/"summary.json").read_text(encoding="utf-8"))
    if digest(RESULTS/"test_sets.npz") != protocol["test_sets_sha256"] or digest(SETTINGS) != protocol["settings_sha256"]:
        raise RuntimeError("Frozen test sets or their settings changed")
    for group, base in (("outputs_sha256", RESULTS), ("checkpoints_sha256", RESULTS/"ladder"),
                        ("source_sha256", ROOT)):
        for relative, expected in summary[group].items():
            if digest(base/relative) != expected:
                raise RuntimeError(f"Integrity mismatch: {relative}")
    levels = {level["name"]: level for level in json.loads(LEVELS.read_text(encoding="utf-8"))["levels"]}
    model = load_model(levels[summary["selected"]])
    data, _ = panels()
    with np.load(RESULTS/"selected_predictions.npz", allow_pickle=False) as saved:
        with (RESULTS/"test_metrics.csv").open(encoding="utf-8") as handle:
            rows = {r["panel"]: r for r in csv.DictReader(handle) if r["level"] == summary["selected"]}
        for panel in PANELS:
            predicted = price_and_greeks(model, data[panel]["x"], data[panel]["put"])
            stored = saved[f"{panel}_predicted"]
            for i, q in enumerate(QUANTITIES):
                if not np.allclose(predicted[q], stored[:, i], rtol=1e-9, atol=1e-11):
                    raise RuntimeError(f"Re-evaluated {q} on {panel} differs from saved predictions")
            metrics = quantity_metrics(predicted, data[panel]["reference"])
            for q in QUANTITIES:
                if not np.isclose(metrics[q]["rmse"], float(rows[panel][f"{q}_rmse"]), rtol=1e-6):
                    raise RuntimeError(f"Recomputed {q} RMSE on {panel} differs from the saved table")
    print(f"Verified protocol, {len(summary['outputs_sha256'])} outputs, {len(summary['checkpoints_sha256'])} "
          f"checkpoints and {len(summary['source_sha256'])} sources; re-evaluated {summary['selected']} "
          f"on {len(PANELS)} panels. No model was trained.")


if __name__ == "__main__":
    main()
