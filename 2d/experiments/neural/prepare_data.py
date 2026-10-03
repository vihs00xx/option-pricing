"""Generate contract sets and reference labels; freeze the test sets before any training.

Training and validation labels go to artifacts/neural (rebuildable, not committed).
The test sets and their references go to the results folder with a protocol record.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
import torch

from src.basket import BasketContract
from src.neural.domain import box_from_dict, sample_contracts
from src.neural.reference import QUANTITIES, contract_row, reference_values
from src.neural.training import reduced_tensor, reference_labels

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT/"experiments/config/neural/basket_neural_settings.json"
CASES = ROOT/"experiments/config/classical/basket_settings.json"
DATA = ROOT/"artifacts/neural"
RESULTS = ROOT/"results/neural/two_asset_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_settings():
    return json.loads(SETTINGS.read_text(encoding="utf-8"))


def contract_set(settings, name):
    spec = settings["sets"][name]
    box = box_from_dict(settings["train_box" if name == "train" else "test_box"])
    if name == "short_maturity":
        box = replace(box, tau=tuple(settings["short_maturity_tau"]))
    x = sample_contracts(box, spec["count"], spec["seed"], spec["tau_power"])
    return x, np.arange(len(x)) % 2 == 1


def study_cases():
    """The eight classical-study contracts, as an illustrative named panel."""
    rows = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    pairs = [contract_row(BasketContract(**{k: v for k, v in row.items() if k != "name"})) for row in rows]
    return [row["name"] for row in rows], np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    rule = settings["reference_rule"]
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frozen = RESULTS/"test_sets.npz"
    if frozen.exists() and not args.overwrite:
        raise FileExistsError("Test sets are frozen; use --overwrite only to rebuild an unused study")
    timing = {}

    start = perf_counter()
    x, _ = contract_set(settings, "train")
    r = reduced_tensor(x)
    torch.save(dict(x=torch.as_tensor(x), r=r, **reference_labels(r, 2, nodes=rule["nodes"], limit=rule["limit"])),
               DATA/"train.pt")
    timing["train_labels_seconds"] = perf_counter()-start

    start = perf_counter()
    x, put = contract_set(settings, "validation")
    r = reduced_tensor(x)
    reference = reference_values(x, put, nodes=rule["nodes"], limit=rule["limit"])
    torch.save(dict(x=torch.as_tensor(x), put=torch.as_tensor(put), r=r,
                    reference={k: torch.as_tensor(v) for k, v in reference.items()},
                    **reference_labels(r, 2, nodes=rule["nodes"], limit=rule["limit"])), DATA/"validation.pt")
    timing["validation_labels_seconds"] = perf_counter()-start

    start = perf_counter()
    arrays = {}
    names, cx, cput = study_cases()
    panels = {"test": contract_set(settings, "test"), "short_maturity": contract_set(settings, "short_maturity"),
              "study_cases": (cx, cput)}
    for panel, (x, put) in panels.items():
        reference = reference_values(x, put, nodes=rule["nodes"], limit=rule["limit"])
        arrays[f"{panel}_contracts"] = x
        arrays[f"{panel}_put"] = put
        arrays[f"{panel}_reference"] = np.column_stack([reference[q] for q in QUANTITIES])
    arrays["study_case_names"] = np.array(names)
    arrays["quantities"] = np.array(QUANTITIES)
    np.savez(frozen, **arrays)
    timing["test_labels_seconds"] = perf_counter()-start

    protocol = dict(
        study_id=settings["study_id"],
        frozen_at=datetime.now(timezone.utc).isoformat(),
        statement=("Test contracts and references were generated and hashed before any model was trained. "
                   "Model and level selection use the validation set only."),
        settings_sha256=digest(SETTINGS),
        test_sets_sha256=digest(frozen),
        source_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sorted(
            list((ROOT/"src/neural").glob("*.py"))+[Path(__file__)])},
        timing=timing,
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                         threads=torch.get_num_threads()),
    )
    (RESULTS/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
