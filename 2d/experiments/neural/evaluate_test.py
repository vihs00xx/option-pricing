"""Score the frozen PINN and DeepONet selections on the frozen two-stock test set.

Checks the test set's seal and the selection's checkpoint and source hashes first, so
the models scored are exactly the ones frozen before the test set was opened. Every
case of every split is reported; the Levy approximation and every ladder level are
scored alongside for context.

Usage: python -m experiments.neural.evaluate_test
"""

import csv
import json
from time import perf_counter

import numpy as np
import torch

from experiments.neural.common import RESULTS, evaluation_set, training_data
from experiments.records import ROOT, digest, verify_metadata, write_csv
from experiments.test_set.freeze_test_set import FLOORS
from src.neural.coordinates import FIELDS, levy_call
from src.neural.networks import load_model
from src.neural.reference import QUANTITIES, call_pricer, price_and_greeks, reference_values

TEST_SET = ROOT/"results/test_set_v1"
SPLITS = ("interpolation", "short_maturity", "extrapolation", "anchor")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_test_set():
    """Verified cases (contract rows, put flags, splits, ids) and references."""
    verify_metadata(TEST_SET)
    protocol = TEST_SET/"protocol.json"
    if digest(protocol) != (TEST_SET/"protocol.sha256").read_text(encoding="utf-8").strip():
        raise RuntimeError("Test-set protocol seal mismatch")
    sealed = json.loads(protocol.read_text(encoding="utf-8"))
    if digest(TEST_SET/"cases.json") != sealed["cases_sha256"] or \
            digest(TEST_SET/"references.csv") != sealed["references_sha256"]:
        raise RuntimeError("Test cases or references differ from the sealed protocol")
    cases = json.loads((TEST_SET/"cases.json").read_text(encoding="utf-8"))
    references = {row["case_id"]: row for row in read_csv(TEST_SET/"references.csv")}
    x = np.array([contract_row(c["contract"]) for c in cases])
    put = np.array([c["contract"]["option"] == "put" for c in cases])
    reference = {q: np.array([float(references[c["case_id"]][q]) for c in cases]) for q in QUANTITIES}
    return dict(x=x, put=put, split=np.array([c["split"] for c in cases]),
                case_id=[c["case_id"] for c in cases], reference=reference)


def contract_row(contract):
    """Contract row (``FIELDS``) of a sealed two-stock case."""
    values = dict(spot_1=contract["spots"][0], spot_2=contract["spots"][1], strike=contract["strike"],
                  tau=contract["tau"], rate=contract["rate"], dividend_1=contract["dividends"][0],
                  dividend_2=contract["dividends"][1], vol_1=contract["volatilities"][0],
                  vol_2=contract["volatilities"][1], correlation=contract["correlation"],
                  weight_1=contract["weights"][0])
    return [float(values[name]) for name in FIELDS]


def floor(quantity):
    return FLOORS[quantity.split("_")[0]]


def metrics(values, reference, mask):
    rows = {}
    for q in QUANTITIES:
        error = values[q][mask]-reference[q][mask]
        relative = np.abs(error)/np.maximum(np.abs(reference[q][mask]), floor(q))
        rows[q] = dict(rmse=float(np.sqrt(np.mean(error**2))), max_abs=float(np.max(np.abs(error))),
                       median_abs=float(np.median(np.abs(error))), median_relative=float(np.median(relative)),
                       max_relative=float(np.max(relative)))
    return rows


def verify_selection():
    selection = json.loads((RESULTS/"selection.json").read_text(encoding="utf-8"))
    for name, expected in selection["settings_sha256"].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError(f"Study settings changed after the selection was frozen: {name}")
    for name, expected in selection["evaluation_sources_sha256"].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError(f"Evaluation source changed after the selection was frozen: {name}")
    for entry in list(selection["selected"].values())+list(selection["ladder"].values()):
        if digest(ROOT/entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed after the selection was frozen: {entry['checkpoint']}")
    return selection


def timed(price_fn, x, put):
    start = perf_counter()
    values = price_and_greeks(price_fn, x, put)
    batch = perf_counter()-start
    singles = []
    for i in range(min(len(x), 32)):
        start = perf_counter()
        price_and_greeks(price_fn, x[i:i+1], put[i:i+1])
        singles.append(perf_counter()-start)
    return values, dict(batch_seconds_per_contract=batch/len(x), single_contract_seconds=float(np.median(singles)))


def test_set_checks(test):
    """Validity of the test set for this study, recorded before any model is scored.

    Independence: for every development contract set, the smallest over test cases of
    the largest parameter difference (in units of the parameter's range) to any
    development contract. Reference: the batched quadrature that labelled the training
    data, evaluated on the test cases, against the sealed adaptive references.
    """
    scale = np.array([100, 100, 1, 2, 0.12, 0.06, 0.06, 0.5, 0.5, 1.7, 0.6])
    pools = {"data_32768": training_data(32768)["x"].numpy(), "anchors_4096": training_data(4096, "anchors")["x"].numpy()}
    for kind in ("monitor", "validation", "short_validation"):
        pools[kind] = evaluation_set(kind)["inputs"][0]
    independence = {}
    for name, pool in pools.items():
        nearest = [np.min(np.max(np.abs((row-pool)/scale), axis=1)) for row in test["x"]]
        independence[name] = float(np.min(nearest))
    batched = reference_values(test["x"], test["put"])
    agreement = {q: float(np.max(np.abs(batched[q]-test["reference"][q])
                                 / np.maximum(np.abs(test["reference"][q]), floor(q)))) for q in QUANTITIES}
    return dict(seal_verified=True, cases=len(test["x"]), puts=int(test["put"].sum()),
                split_counts={s: int((test["split"] == s).sum()) for s in SPLITS},
                smallest_scaled_distance_to_development=independence,
                batched_reference_worst_relative_difference=agreement)


def main():
    torch.set_num_threads(1)
    selection = verify_selection()
    test = load_test_set()
    checks = test_set_checks(test)
    (RESULTS/"test_set_checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    methods = {"pinn_selected": selection["selected"]["pinn"]["checkpoint"],
               "deeponet_selected": selection["selected"]["deeponet"]["checkpoint"],
               "levy_approximation": None}
    methods.update({level: entry["checkpoint"] for level, entry in selection["ladder"].items()})
    prediction_rows, metric_rows, timing = [], [], {}
    for method, checkpoint in methods.items():
        price_fn = call_pricer(levy_call) if checkpoint is None else load_model(ROOT/checkpoint).price
        values, timing[method] = timed(price_fn, test["x"], test["put"])
        for i, case_id in enumerate(test["case_id"]):
            prediction_rows.append(dict(case_id=case_id, split=test["split"][i], method=method,
                                        **{q: values[q][i] for q in QUANTITIES}))
        for split in SPLITS+("all",):
            mask = np.ones(len(test["x"]), bool) if split == "all" else test["split"] == split
            for q, row in metrics(values, test["reference"], mask).items():
                metric_rows.append(dict(method=method, split=split, cases=int(mask.sum()), quantity=q, **row))
        print(f"{method}: price RMSE {metric_rows[-11]['rmse']:.3g} over all cases")
    write_csv(RESULTS/"test_predictions.csv", prediction_rows)
    write_csv(RESULTS/"test_metrics.csv", metric_rows)
    (RESULTS/"test_timing.json").write_text(json.dumps(dict(
        threads=1, contracts=len(test["x"]), torch=torch.__version__, methods=timing), indent=2)+"\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
