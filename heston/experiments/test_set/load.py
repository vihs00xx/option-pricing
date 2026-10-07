"""Load the sealed heston_test_v1 after checking its seal."""

import csv
import json

import numpy as np

from experiments.records import ROOT, digest
from src.heston.model import QUANTITIES, Contracts

TEST = ROOT / "results" / "test_set_v1"


def load_test_set(folder=TEST):
    seal = (folder / "protocol.sha256").read_text(encoding="utf-8").strip()
    if digest(folder / "protocol.json") != seal:
        raise RuntimeError("protocol.json does not match protocol.sha256")
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    for name, key in (("cases.json", "cases_sha256"), ("references.csv", "references_sha256")):
        if digest(folder / name) != protocol[key]:
            raise RuntimeError(f"{name} does not match the sealed protocol")
    cases = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    refs = {r["case_id"]: r for r in csv.DictReader((folder / "references.csv").open(encoding="utf-8"))}
    contracts = Contracts.from_records([{**c["parameters"], "call": c["call"]} for c in cases])
    return dict(contracts=contracts, case_id=np.array([c["case_id"] for c in cases]),
                split=np.array([c["split"] for c in cases]), cases=cases, protocol=protocol,
                reference={q: np.array([float(refs[c["case_id"]][q]) for c in cases]) for q in QUANTITIES})
