"""Apply the pre-declared selection rule and freeze the selected pricers by hash.

Writes ladder.csv and selection.json. The selection pins each selected checkpoint, the
study settings and the evaluation sources before any network is scored on the test set.

Usage (from heston/): python -m experiments.neural.select_models [--overwrite]
"""

import argparse
import json
from datetime import datetime, timezone

from experiments.neural.common import LADDER, RESULTS, SETTINGS_FILES, load_settings
from experiments.records import ROOT, digest, write_csv
from src.heston.model import QUANTITIES

EVALUATION_SOURCES = ("experiments/neural/evaluate_test.py", "experiments/neural/common.py", "experiments/records.py",
                      "experiments/test_set/load.py", "src/neural/coordinates.py", "src/neural/networks.py",
                      "src/neural/pinn.py", "src/neural/deeponet.py", "src/neural/reference.py")


def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")


def ladder_rows(settings):
    rows = []
    seed = settings["seeds"]["training"]
    for order, level in enumerate(settings["levels"]):
        path = LADDER / f"{level}_seed{seed}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        v, s = r["scores"]["validation"], r["scores"]["short_validation"]
        rows.append(dict(order=order, level=level, family=settings["levels"][level]["family"], technique=r["technique"],
                         parameters=r["parameters"], validation_score=v["score"], short_validation_score=s["score"],
                         selection_score=0.5 * (v["score"] + s["score"]),
                         training_seconds=round(r["total_seconds"], 1), label_seconds=round(r.get("label_seconds", 0.0), 1),
                         stopped_early=r.get("stopped_early"),
                         **{f"rmse_{q}": v["rmse"][q] for q in QUANTITIES},
                         **{f"short_rmse_{q}": s["rmse"][q] for q in QUANTITIES}))
    return rows


def select(rows, family):
    """Lowest selection score; a level within 1% of the best yields to an earlier one."""
    candidates = [r for r in rows if r["family"] == family]
    best = min(r["selection_score"] for r in candidates)
    return min((r for r in candidates if r["selection_score"] <= 1.01 * best), key=lambda r: r["order"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = RESULTS / "selection.json"
    if path.exists() and not args.overwrite:
        raise FileExistsError("selection.json is frozen")
    if (RESULTS / "test_metrics.csv").exists():
        raise RuntimeError("the test set has been evaluated; the selection can no longer change")
    settings = load_settings()
    rows = ladder_rows(settings)
    write_csv(RESULTS / "ladder.csv", rows)
    seed = settings["seeds"]["training"]
    selected = {}
    for family in ("pinn", "deeponet"):
        row = select(rows, family)
        ckpt = LADDER / f"{row['level']}_seed{seed}.pt"
        selected[family] = dict(level=row["level"], selection_score=row["selection_score"],
                                validation_score=row["validation_score"],
                                short_validation_score=row["short_validation_score"],
                                checkpoint=rel(ckpt), checkpoint_sha256=digest(ckpt))
        print(f"{family}: {row['level']} (selection score {row['selection_score']:.4g})")
    test = ROOT / "results/test_set_v1"
    record = dict(
        frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        selection_rule=settings["selection_rule"], selected=selected,
        ladder={r["level"]: dict(checkpoint=rel(LADDER / f"{r['level']}_seed{seed}.pt"),
                                 checkpoint_sha256=digest(LADDER / f"{r['level']}_seed{seed}.pt")) for r in rows},
        settings_sha256={rel(p): digest(p) for p in SETTINGS_FILES},
        evaluation_sources_sha256={name: digest(ROOT / name) for name in EVALUATION_SOURCES},
        test_set=dict(folder="results/test_set_v1", protocol_sha256=(test / "protocol.sha256").read_text().strip()),
        evaluation_plan="Score the selected PINN and DeepONet once on every case of every split of heston_test_v1, "
                        "with the Black-Scholes base and every ladder level alongside for context. Report RMSE, median "
                        "and largest absolute error for the price and all 11 sensitivities, by split. No case is dropped.")
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
