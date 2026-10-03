"""Verify the saved PINN/DeepONet study without training: hashes, then a re-evaluation.

Checks the pinned sources and outputs (``metadata.json``), every ladder file
(``ladder_manifest.json``) and the frozen selection, then re-prices the frozen test set
with the selected checkpoints and compares with the saved predictions.

Usage: python -m experiments.neural.verify_results
"""

import csv
import json

import numpy as np
import torch

from experiments.neural.common import RESULTS
from experiments.neural.evaluate_test import load_test_set, verify_selection
from experiments.records import ROOT, digest, verify_metadata
from src.neural.networks import load_model
from src.neural.reference import QUANTITIES, price_and_greeks


def main():
    torch.set_num_threads(4)
    metadata = verify_metadata(RESULTS)
    manifest = json.loads((RESULTS/"ladder_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        if digest(ROOT/name) != expected:
            raise RuntimeError(f"Ladder file changed: {name}")
    selection = verify_selection()
    checked = 0
    if (RESULTS/"test_predictions.csv").exists():
        test = load_test_set()
        with (RESULTS/"test_predictions.csv").open(encoding="utf-8", newline="") as handle:
            saved = list(csv.DictReader(handle))
        for family, entry in selection["selected"].items():
            values = price_and_greeks(load_model(ROOT/entry["checkpoint"]).price, test["x"], test["put"])
            rows = {r["case_id"]: r for r in saved if r["method"] == f"{family}_selected"}
            for q in QUANTITIES:
                stored = np.array([float(rows[c][q]) for c in test["case_id"]])
                if not np.allclose(values[q], stored, rtol=1e-9, atol=1e-12):
                    raise RuntimeError(f"Re-evaluated {family} {q} differs from the saved predictions")
            checked += 1
    print(f"Verified {len(metadata['source_sha256'])} sources, {len(metadata['outputs_sha256'])} outputs, "
          f"{len(manifest)} ladder files and the frozen selection; re-evaluated {checked} selected pricers "
          "on the test set.")


if __name__ == "__main__":
    main()
