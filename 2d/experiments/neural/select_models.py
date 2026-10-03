"""Apply the pre-declared selection rule to the ladders and freeze the selected pricers.

Writes ``ladder.csv`` and ``selection.json``. The selection pins each selected
checkpoint, the study settings and the evaluation sources by SHA-256 before any model
is scored on the frozen test set, as the test-set protocol requires.

Usage: python -m experiments.neural.select_models [--overwrite]
"""

import argparse
from datetime import datetime, timezone
import json

from experiments.neural.common import LADDER, RESULTS, SETTINGS_FILES, load_settings
from experiments.records import ROOT, digest, relative, write_csv
from src.neural.reference import QUANTITIES

EVALUATION_SOURCES = ("experiments/neural/evaluate_test.py", "experiments/neural/common.py", "experiments/records.py",
                      "src/neural/coordinates.py", "src/neural/networks.py", "src/neural/pinn.py",
                      "src/neural/deeponet.py", "src/neural/reference.py")


def ladder_rows(settings):
    rows = []
    for order, level in enumerate(settings["levels"]):
        path = LADDER/f"{level}_seed{settings['seeds']['training']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        validation = record["scores"]["validation"]
        rows.append(dict(order=order, level=level, family=settings["levels"][level]["family"],
                         technique=record["technique"], parameters=record["parameters"],
                         validation_score=validation["score"],
                         short_validation_score=record["scores"]["short_validation"]["score"],
                         training_seconds=round(record["total_seconds"], 1),
                         **{f"rmse_{q}": validation["rmse"][q] for q in QUANTITIES}))
    return rows


def select(rows, family):
    """Lowest validation score; a level within 1% of the best yields to an earlier one."""
    candidates = [r for r in rows if r["family"] == family]
    best = min(r["validation_score"] for r in candidates)
    return min((r for r in candidates if r["validation_score"] <= 1.01*best), key=lambda r: r["order"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = RESULTS/"selection.json"
    if path.exists() and not args.overwrite:
        raise FileExistsError("selection.json is frozen; pass --overwrite only before any test evaluation")
    if (RESULTS/"test_metrics.csv").exists():
        raise RuntimeError("The test set has been evaluated; the selection can no longer change")
    settings = load_settings()
    rows = ladder_rows(settings)
    write_csv(RESULTS/"ladder.csv", rows)
    seed = settings["seeds"]["training"]
    selected = {}
    for family in ("pinn", "deeponet"):
        row = select(rows, family)
        checkpoint = LADDER/f"{row['level']}_seed{seed}.pt"
        selected[family] = dict(level=row["level"], validation_score=row["validation_score"],
                                checkpoint=relative(checkpoint), checkpoint_sha256=digest(checkpoint))
        print(f"{family}: {row['level']} (validation score {row['validation_score']:.4g})")
    test_set = ROOT/"results/test_set_v1"
    record = dict(
        frozen_at=datetime.now(timezone.utc).isoformat(),
        selection_rule=settings["selection_rule"],
        selected=selected,
        ladder={r["level"]: dict(checkpoint=relative(LADDER/f"{r['level']}_seed{seed}.pt"),
                                 checkpoint_sha256=digest(LADDER/f"{r['level']}_seed{seed}.pt"))
                for r in rows},
        settings_sha256={relative(path): digest(path) for path in SETTINGS_FILES},
        evaluation_sources_sha256={name: digest(ROOT/name) for name in EVALUATION_SOURCES},
        test_set=dict(folder="results/test_set_v1",
                      protocol_sha256=(test_set/"protocol.sha256").read_text(encoding="utf-8").strip()),
        evaluation_plan=("Score the selected PINN and DeepONet once on every case of every split of the frozen "
                         "test set, with the Levy approximation and every ladder level alongside for context. "
                         "Report RMSE, largest absolute error and the test set's floored relative errors for "
                         "the price and all ten Greeks, by split. No case is dropped."),
    )
    path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
