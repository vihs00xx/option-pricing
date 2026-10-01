"""Check saved basket-study hashes without running pricing models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"results/basket/two_asset_v1")
    output = parser.parse_args().output.resolve()
    metadata = json.loads((output/"metadata.json").read_text(encoding="utf-8"))
    for relative, expected in metadata["source_sha256"].items():
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source integrity mismatch: {relative}")
    for relative, expected in metadata["outputs_sha256"].items():
        path = (output/relative).resolve()
        if not path.is_relative_to(output) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Result integrity mismatch: {relative}")
    with np.load(output/"replicate_means.npz", allow_pickle=False) as arrays:
        with (output/"price_runs.csv").open(encoding="utf-8") as handle:
            prices = list(csv.DictReader(handle))
        means = arrays["price"]
        if len(prices) != len(means):
            raise RuntimeError("Price replicate row count does not match CSV")
        for row, replicates in zip(prices, means):
            if not np.isclose(replicates.mean(), float(row["price"]), atol=1e-13, rtol=1e-13):
                raise RuntimeError("Price replicate mean does not match CSV")
            if not np.isclose(replicates.std(ddof=1)/np.sqrt(len(replicates)), float(row["standard_error"]), atol=1e-13, rtol=1e-13):
                raise RuntimeError("Price replicate uncertainty does not match CSV")
        with (output/"greek_runs.csv").open(encoding="utf-8") as handle:
            greeks = list(csv.DictReader(handle))
        quantities = arrays["greek_quantities"].tolist()
        if len(greeks) != len(arrays["greek_groups"])*len(quantities):
            raise RuntimeError("Greek replicate row count does not match CSV")
        for i, row in enumerate(greeks):
            group, quantity = divmod(i, len(quantities))
            identity = f"{row['case']}|{row['method']}|{row['seed']}"
            if identity != arrays["greek_groups"][group] or row["quantity"] != quantities[quantity]:
                raise RuntimeError("Greek replicate identity does not match CSV")
            replicates = arrays["greeks"][group, :, quantity]
            expected = [replicates.mean(), replicates.std(ddof=1)/np.sqrt(len(replicates))]
            if not np.allclose(expected, [float(row["value"]), float(row["standard_error"])], atol=1e-13, rtol=1e-13):
                raise RuntimeError("Greek replicate estimates do not match CSV")
    print(f"Verified {len(metadata['source_sha256'])} source/configuration files and "
          f"{len(metadata['outputs_sha256'])} saved outputs. No models were evaluated.")


if __name__ == "__main__":
    main()
