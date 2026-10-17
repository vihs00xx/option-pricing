"""Summarize the Phase 0 rule check into results/reference_check_v1/report.md and CSV tables.

Run from 8d/ after experiments.reference.rule_check.
"""

from . import common  # noqa: F401  (pins BLAS threads before NumPy)

import json
from pathlib import Path

import numpy as np

from experiments.records import pricing_sources, write_metadata
from src.pricing import quantity_names

from .common import OUTPUT, ROOT, load_contracts, read_jsonl, write_csv
from .rule_check import ENGINE, PROTOCOL, RULES, STUDY

FLOORS = {"price": 1e-4, "delta": 1e-4, "gamma": 1e-7, "vega": 1e-4, "theta": 1e-4, "rho": 1e-4,
          "correlation": 1e-4}


def floor(quantity):
    return FLOORS[quantity.split("_")[0]]


def main():
    contracts = {name: (c, info) for name, c, info in load_contracts()}
    names = quantity_names(next(iter(contracts.values()))[0])
    floors = np.array([floor(q) for q in names])
    engine = read_jsonl(ENGINE)
    top = max(r["points"] for r in engine)
    truth = {r["name"]: r for r in engine if r["points"] == top}
    ladder = []
    for points in sorted({r["points"] for r in engine}):
        rows = [r for r in engine if r["points"] == points]
        se = np.array([r["standard_errors"][0] for r in rows])
        ladder.append(dict(points=points, contracts=len(rows), median_price_se=float(np.median(se)),
                           max_price_se=float(se.max()), share_below_1e4=float(np.mean(se <= 1e-4)),
                           median_seconds=float(np.median([r["seconds"] for r in rows])),
                           max_seconds=float(max(r["seconds"] for r in rows)),
                           max_peak_gb=float(max(r["peak_gb"] for r in rows))))
    write_csv(OUTPUT / "engine_ladder.csv", ladder)
    errors = []
    for r in read_jsonl(RULES):
        if r["name"] not in truth:
            continue
        t = truth[r["name"]]
        diff = np.array(r["values"]) - np.array(t["values"])
        se = np.array(t["standard_errors"])
        relative = np.abs(diff) / np.maximum(np.abs(t["values"]), floors)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(se > 0, np.abs(diff) / se, np.inf)
        k = int(np.argmax(relative))
        errors.append(dict(name=r["name"], group=contracts[r["name"]][1]["group"], purpose=r["purpose"],
                           points=r["points"], seed=r["seed"], seconds_per_contract=r["seconds_per_contract"],
                           price_error=float(diff[0]), price_error_over_truth_se=float(ratio[0]),
                           price_relative_error=float(relative[0]), max_relative_error=float(relative[k]),
                           worst_quantity=names[k], median_relative_error=float(np.median(relative)),
                           truth_max_relative_se=float(np.max(se / np.maximum(np.abs(t["values"]), floors)))))
    write_csv(OUTPUT / "torch_rules.csv", errors)
    summary = []
    for purpose in ("labels", "evaluation"):
        for seed in sorted({e["seed"] for e in errors if e["purpose"] == purpose}):
            rows = [e for e in errors if e["purpose"] == purpose and e["seed"] == seed]
            summary.append(dict(purpose=purpose, points=rows[0]["points"], seed=seed, contracts=len(rows),
                                rms_price_relative=float(np.sqrt(np.mean([e["price_relative_error"] ** 2 for e in rows]))),
                                max_price_relative=float(max(e["price_relative_error"] for e in rows)),
                                median_of_median_relative=float(np.median([e["median_relative_error"] for e in rows])),
                                median_price_error_over_truth_se=float(np.median(
                                    [e["price_error_over_truth_se"] for e in rows])),
                                seconds_per_contract=float(rows[0]["seconds_per_contract"])))
    write_csv(OUTPUT / "torch_rules_summary.csv", summary)
    lines = ["# Phase 0: eight-stock reference check", "",
             "Development contracts only. Truth = engine RQMC at the top points level of the test-set protocol "
             f"({top} points x 16 replicates, geometric control), with its standard error.", "",
             "## Engine RQMC ladder", "", "| Points | Contracts | Median price SE | Max price SE | Share <= 1e-4 | "
             "Median s | Max s | Peak GB |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['points']} | {r['contracts']} | {r['median_price_se']:.2e} | {r['max_price_se']:.2e} | "
              f"{r['share_below_1e4']:.0%} | {r['median_seconds']:.0f} | {r['max_seconds']:.0f} | {r['max_peak_gb']:.2f} |"
              for r in ladder]
    lines += ["", "## Torch rules against the truth", "",
              "| Rule | Points | Seed | Contracts | RMS price rel. error | Max price rel. error | "
              "Median rel. error (all quantities) | Median price error / truth SE | s per contract |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['purpose']} | {r['points']} | {r['seed']} | {r['contracts']} | {r['rms_price_relative']:.1e} | "
              f"{r['max_price_relative']:.1e} | {r['median_of_median_relative']:.1e} | "
              f"{r['median_price_error_over_truth_se']:.1f} | {r['seconds_per_contract']:.3f} |" for r in summary]
    lines += ["", "Worst contracts for the evaluation rule (study seed):", "",
              "| Contract | Max rel. error | Quantity | Price rel. error | Truth max rel. SE |", "|---|---:|---|---:|---:|"]
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    seed = study["seeds"]["reference_rule"] + 1
    worst = sorted((e for e in errors if e["purpose"] == "evaluation" and e["seed"] == seed),
                   key=lambda e: -e["max_relative_error"])[:10]
    lines += [f"| {e['name']} | {e['max_relative_error']:.1e} | {e['worst_quantity']} | "
              f"{e['price_relative_error']:.1e} | {e['truth_max_relative_se']:.1e} |" for e in worst]
    lines += ["", "A rule error well above the truth's standard error is the rule's; at or below it the truth "
              "cannot resolve the rule. Relative errors divide by max(|truth|, floor), floors 1e-7 for Gamma and "
              "1e-4 otherwise.", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_metadata(OUTPUT, PROTOCOL, pricing_sources() + [Path(__file__), ROOT / "experiments/reference/rule_check.py",
                                                         ROOT / "src/neural/reference.py", STUDY])
    print(f"Saved {OUTPUT / 'report.md'}")


if __name__ == "__main__":
    main()
