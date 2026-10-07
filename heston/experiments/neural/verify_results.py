"""Verify the saved Heston neural study without training: hashes, then a re-evaluation.

Checks metadata.json (outputs and sources), the frozen selection (checkpoints and evaluation
sources), the test-set seal and the classical results' metadata, then re-prices the test set
with the selected checkpoints and compares with the saved predictions.

Usage (from heston/): python -m experiments.neural.verify_results
"""

import csv
import json

import numpy as np
import torch

from experiments.neural.common import LADDER, RESULTS
from experiments.neural.evaluate_test import verify_selection
from experiments.records import ROOT, digest, verify_folder
from experiments.test_set.load import load_test_set
from src.heston.model import QUANTITIES
from src.neural.coordinates import rows_from_contracts
from src.neural.networks import load_model
from src.neural.reference import price_and_greeks


def main():
    torch.set_num_threads(4)
    for folder in (RESULTS, ROOT / "results/classical_v1", ROOT / "results/test_set_v1"):
        bad = verify_folder(folder)
        if bad:
            raise RuntimeError(f"{folder.name}: changed since metadata.json was written: {bad}")
    meta = json.loads((RESULTS / "metadata.json").read_text(encoding="utf-8"))
    changed = [n for n, h in meta["ladder_files"].items() if digest(LADDER / n) != h]
    if changed:
        raise RuntimeError(f"ladder files changed: {changed}")
    selection = verify_selection()
    test = load_test_set()
    x, put = rows_from_contracts(test["contracts"])
    with (RESULTS / "test_predictions.csv").open(encoding="utf-8", newline="") as fh:
        saved = list(csv.DictReader(fh))
    for family, entry in selection["selected"].items():
        values = price_and_greeks(load_model(ROOT / entry["checkpoint"]).price, x, put)
        rows = {r["case_id"]: r for r in saved if r["pricer"] == f"{family}_selected"}
        for q in QUANTITIES:
            stored = np.array([float(rows[c][q]) for c in test["case_id"]])
            if not np.allclose(values[q], stored, rtol=1e-9, atol=1e-12):
                raise RuntimeError(f"re-evaluated {family} {q} differs from the saved predictions")
    print("verified: neural, classical and test-set hashes; frozen selection; re-evaluated "
          f"{len(selection['selected'])} selected pricers on {len(x)} test cases")


if __name__ == "__main__":
    main()
