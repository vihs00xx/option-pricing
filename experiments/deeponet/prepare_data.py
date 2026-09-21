import hashlib
import json
from pathlib import Path

import numpy as np

from src.common_benchmark import build_cases
from src.neural.deeponet_data import (
    BRANCH_FEATURES,
    TARGETS,
    TRUNK_FEATURES,
    build_dataset,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha256(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    settings_path = ROOT / "experiments/config/neural/deeponet_settings.json"
    benchmark_path = ROOT / "experiments/config/classical/common_benchmark_settings.json"
    settings = json.loads(settings_path.read_text())
    benchmark_settings = json.loads(benchmark_path.read_text())
    dataset = build_dataset(settings, build_cases(benchmark_settings))

    output_path = ROOT / settings["output"]["dataset"]
    manifest_path = ROOT / settings["output"]["manifest"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split, values in dataset.items():
        for name, value in values.items():
            arrays[f"{split}_{name}"] = value
    np.savez_compressed(output_path, **arrays)

    manifest = {
        "dataset_version": settings["dataset_version"],
        "protocol_id": settings["protocol_id"],
        "status": settings["status"],
        "settings_sha256": _sha256(settings_path),
        "common_benchmark_settings_sha256": _sha256(benchmark_path),
        "dataset_sha256": _sha256(output_path),
        "dataset_content_sha256": _content_sha256(arrays),
        "branch_features": list(BRANCH_FEATURES),
        "trunk_features": list(TRUNK_FEATURES),
        "targets": list(TARGETS),
        "split_points": {
            split: int(len(values["branch"])) for split, values in dataset.items()
        },
        "split_contracts": {
            "train": int(settings["sampling"]["train_contracts"]),
            "validation": int(settings["sampling"]["validation_contracts"]),
            "test": len(set(dataset["test"]["case_id"].tolist())),
        },
        "test_policy": "Frozen common benchmark; excluded from training and model selection.",
        "real_market_data_policy": settings["data_policy"]["real_market_data"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
