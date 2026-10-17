"""Phase 0: how accurate are the eight-stock references, and what do they cost?

Development contracts only (experiments/config/reference/dev_contracts_v1.json).

1. Engine RQMC ladder. rqmc_estimate (16 replicates, direction "volatility", greeks,
   geometric control) at each points level of the test-set protocol, seed 808400 +
   contract index: standard errors and seconds per level. The top level is the
   "truth" for step 2 and is reported with its standard error.
2. Torch rules. The training-label and evaluation rules of
   experiments/config/neural/study.json (one fixed scrambled Sobol net with the
   geometric control) on the same contracts, with the study's rule seeds and three
   other seeds (808500 + k) to show how much the error moves with the net. Error =
   rule - truth per quantity, as a ratio to the truth's standard error and relative to
   max(|truth|, floor).

Run from 8d/: python -m experiments.reference.rule_check [--workers 8]
then python -m experiments.reference.make_report. Rows are appended to
results/reference_check_v1/*.jsonl as they finish, so an interrupted run resumes.
Check the report before training: the evaluation rule's error should be well below the
networks' validation errors, and the test-set ladder's cost should fit the lab PC.
"""

from . import common  # noqa: F401  (pins BLAS threads before NumPy)

import argparse
import json
from multiprocessing import Pool
from time import perf_counter

import numpy as np

from src.pricing import quantity_names, rqmc_estimate

from .common import OUTPUT, ROOT, WORKERS, append_jsonl, load_contracts, peak_memory_gb, read_jsonl

ENGINE = OUTPUT / "engine_ladder.jsonl"
RULES = OUTPUT / "torch_rules.jsonl"
PROTOCOL = ROOT / "experiments/config/protocol/eight_stock_test_v1.json"
STUDY = ROOT / "experiments/config/neural/study.json"
EXTRA_SEEDS = (808500, 808501, 808502)


def engine_job(task):
    index, name, contract, points, replicates = task
    start = perf_counter()
    e = rqmc_estimate(contract, points, replicates, 808400 + index, greeks=True, control=True)
    names = quantity_names(contract)
    row = dict(name=name, points=points, replicates=replicates, seconds=perf_counter() - start,
               values=[e.values[q] for q in names], standard_errors=[e.standard_errors[q] for q in names],
               peak_gb=peak_memory_gb())
    return row


def rule_rows(contracts, rules, threads):
    """Torch rule values for every contract, one row per (contract, rule, seed)."""
    import torch
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    from src.neural.coordinates import Stocks, basket_row
    from src.neural.reference import quantities, reference_values
    stocks = Stocks(8)
    rows = []
    x = np.array([basket_row(c) for _, c, _ in contracts])
    put = np.array([c.option == "put" for _, c, _ in contracts])
    for purpose, rule, seeds in rules:
        for seed in seeds:
            settings = dict(rule, seed=seed)
            start = perf_counter()
            values = reference_values(stocks, x, put, nodes=settings)
            seconds = (perf_counter() - start) / len(x)
            for k, (name, _, _) in enumerate(contracts):
                rows.append(dict(name=name, purpose=purpose, points=rule["points"], seed=seed,
                                 seconds_per_contract=seconds,
                                 values=[float(values[q][k]) for q in quantities(stocks)]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--threads", type=int, default=WORKERS, help="torch threads for the rule step")
    parser.add_argument("--limit", type=int, default=None, help="first N contracts only (a smoke run)")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))["reference"]
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    contracts = load_contracts()[:args.limit]
    done = {(r["name"], r["points"]) for r in read_jsonl(ENGINE)}
    tasks = [(i, name, c, points, protocol["replicates"]) for i, (name, c, _) in enumerate(contracts)
             for points in protocol["points_ladder"] if (name, points) not in done]
    tasks.sort(key=lambda t: -t[3] * t[2].tau)
    start = perf_counter()
    if tasks:
        with Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(engine_job, tasks), 1):
                append_jsonl(ENGINE, row)
                print(f"[{perf_counter() - start:8.1f}s] engine {i}/{len(tasks)}: {row['name']} at "
                      f"{row['points']} points, price SE {row['standard_errors'][0]:.2e}", flush=True)
    base = study["seeds"]["reference_rule"]
    rules = [(purpose, study["reference_rule"][purpose], (base + (purpose == "evaluation"),) + EXTRA_SEEDS)
             for purpose in ("labels", "evaluation")]
    done = {(r["name"], r["purpose"], r["seed"]) for r in read_jsonl(RULES)}
    missing = [(name, c, info) for name, c, info in contracts
               if any((name, purpose, seed) not in done for purpose, _, seeds in rules for seed in seeds)]
    if missing:
        for row in rule_rows(missing, rules, args.threads):
            if (row["name"], row["purpose"], row["seed"]) not in done:
                append_jsonl(RULES, row)
        print(f"[{perf_counter() - start:8.1f}s] torch rules done for {len(missing)} contracts", flush=True)


if __name__ == "__main__":
    main()
