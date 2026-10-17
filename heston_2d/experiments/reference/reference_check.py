"""Phase 0: accuracy and cost of the Heston basket references, on development contracts only.

1. Time step. Price (16 replicates of 16,384 Sobol paths, the same seed for every step) at
   QE steps 1/8, 1/16, 1/32, 1/64 and 1/128: the change between steps shows the QE bias.
2. Points ladder. Every quantity at the test-set protocol's dt and points levels: standard
   errors and seconds; the top level is the "truth" for step 3.
3. Labels. The training-label simulation of experiments/config/neural/study.json (one
   shared Sobol net, fixed steps) against the truth's price, and the evaluation rule's
   standard errors against the truth.

Run from this folder: python -m experiments.reference.reference_check [--workers 8] [--limit N]
then python -m experiments.reference.reference_check --report. Rows are appended to
results/reference_check_v1/*.jsonl as they finish, so an interrupted run resumes.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.records import pricing_sources, write_metadata
from src.heston_basket.engine import heston_estimate
from src.heston_basket.model import HestonBasket, quantity_names
from src.neural.coordinates import protocol_path

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT/"experiments/config/reference/dev_contracts_v1.json"
STUDY = ROOT/"experiments/config/neural/study.json"
OUTPUT = ROOT/"results/reference_check_v1"
STEPS = (1/8, 1/16, 1/32, 1/64, 1/128)
WORKERS = 8


def read_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row)+"\n")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_contracts(limit=None):
    rows = json.loads(CONTRACTS.read_text(encoding="utf-8"))["contracts"]
    return [(r["name"], r["group"], HestonBasket(**r["contract"])) for r in rows][:limit]


def job(task):
    kind, index, name, c, setting = task
    start = perf_counter()
    if kind == "time_step":
        e = heston_estimate(c, 16384, 16, 909400+index, dt=setting, greeks=False)
    else:
        e = heston_estimate(c, setting["points"], setting["replicates"], 909600+index, dt=setting["dt"])
    return dict(kind=kind, name=name, setting=setting, seconds=perf_counter()-start,
                values=[e.values[q] for q in e.names], standard_errors=[e.standard_errors[q] for q in e.names])


def run(args):
    protocol = json.loads(protocol_path().read_text(encoding="utf-8"))["reference"]
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    contracts = load_contracts(args.limit)
    raw = OUTPUT/"engine.jsonl"
    done = {(r["kind"], r["name"], json.dumps(r["setting"])) for r in read_jsonl(raw)}
    tasks = [("time_step", i, name, c, dt) for i, (name, _, c) in enumerate(contracts) for dt in STEPS]
    evaluation = {k: v for k, v in study["reference"]["evaluation"].items() if k != "note"}
    tasks += [("ladder", i, name, c, dict(points=p, replicates=protocol["replicates"], dt=protocol["dt"]))
              for i, (name, _, c) in enumerate(contracts) for p in protocol["points_ladder"]]
    tasks += [("evaluation", i, name, c, evaluation) for i, (name, _, c) in enumerate(contracts)]
    tasks = [t for t in tasks if (t[0], t[2], json.dumps(t[4])) not in done]
    start = perf_counter()
    if tasks:
        with Pool(args.workers) as pool:
            for k, row in enumerate(pool.imap_unordered(job, tasks), 1):
                append(raw, row)
                print(f"[{perf_counter()-start:8.1f}s] {k}/{len(tasks)} {row['kind']} {row['name']}", flush=True)
    labels = OUTPUT/"labels.jsonl"
    if not any(r for r in read_jsonl(labels)):
        import torch
        torch.set_default_dtype(torch.float64)
        from src.neural.coordinates import Stocks, reduced_coordinates, row_of
        from src.neural.reference import reference_labels
        stocks = Stocks(contracts[0][2].dimension)
        x = torch.as_tensor(np.array([row_of(c) for _, _, c in contracts]))
        features, scale = reduced_coordinates(stocks, x)
        setting = dict(points=study["reference"]["labels"]["points"], steps=study["reference"]["labels"]["steps"],
                       seed=study["seeds"]["label_normals"])
        t = perf_counter()
        values = reference_labels(stocks, features, setting)
        seconds = (perf_counter()-t)/len(contracts)
        call = (scale*values[:, 0]).numpy()
        parity = np.array([c.forward_value()-c.discounted_strike() for _, _, c in contracts])
        for k, (name, _, c) in enumerate(contracts):
            price = call[k]-(parity[k] if c.option == "put" else 0.0)
            append(labels, dict(name=name, setting=setting, price=float(price), seconds_per_contract=seconds))


def report():
    contracts = {name: (group, c) for name, group, c in load_contracts()}
    engine = read_jsonl(OUTPUT/"engine.jsonl")
    protocol = json.loads(protocol_path().read_text(encoding="utf-8"))["reference"]
    top = max(protocol["points_ladder"])
    truth = {r["name"]: r for r in engine if r["kind"] == "ladder" and r["setting"]["points"] == top}
    names = quantity_names(next(iter(contracts.values()))[1].dimension)
    lines = ["# Phase 0: Heston basket reference check", "", "Development contracts only.", "",
             "## Time step (price, 16 x 16,384 Sobol paths, same normals)", "",
             "| Contract | " + " | ".join(f"1/{round(1/s)} - 1/128" for s in STEPS[:-1]) + " | price at 1/128 | SE |",
             "|---|" + "---:|"*(len(STEPS)+1)]
    step_rows = []
    for name in contracts:
        rows = {r["setting"]: r for r in engine if r["kind"] == "time_step" and r["name"] == name}
        if len(rows) == len(STEPS):
            fine = rows[STEPS[-1]]["values"][0]
            lines.append(f"| {name} | " + " | ".join(f"{rows[s]['values'][0]-fine:+.1e}" for s in STEPS[:-1])
                         + f" | {fine:.4f} | {rows[STEPS[-1]]['standard_errors'][0]:.1e} |")
            step_rows.append(dict(name=name, **{f"dt_1_{round(1/s)}": rows[s]["values"][0] for s in STEPS}))
    lines += ["", "## Points ladder (every quantity, protocol dt)", "",
              "| Points | Contracts | Median price SE | Max price SE | Share <= target | Median s | Max s |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for points in protocol["points_ladder"]:
        rows = [r for r in engine if r["kind"] == "ladder" and r["setting"]["points"] == points]
        if rows:
            se = np.array([r["standard_errors"][0] for r in rows])
            secs = [r["seconds"] for r in rows]
            lines.append(f"| {points} | {len(rows)} | {np.median(se):.1e} | {se.max():.1e} | "
                         f"{np.mean(se <= protocol['price_standard_error_target']):.0%} | {np.median(secs):.0f} | "
                         f"{max(secs):.0f} |")
    labels = {r["name"]: r for r in read_jsonl(OUTPUT/"labels.jsonl")}
    evaluation = {r["name"]: r for r in engine if r["kind"] == "evaluation"}
    rel = [abs(labels[n]["price"]-truth[n]["values"][0])/max(abs(truth[n]["values"][0]), 1e-2)
           for n in truth if n in labels]
    if rel:
        lines += ["", "## Training labels and evaluation references against the truth", "",
                  f"Label price relative error (max(|truth|, 0.01)): RMS {np.sqrt(np.mean(np.square(rel))):.1e}, "
                  f"max {max(rel):.1e}; {list(labels.values())[0]['seconds_per_contract']:.3f} s per contract."]
    floors = []
    for n in truth:
        if n in evaluation:
            se = np.array(evaluation[n]["standard_errors"])
            floors.append(se/np.maximum(np.abs(truth[n]["values"]), 1e-4))
    if floors:
        floors = np.median(np.array(floors), axis=0)
        worst = np.argsort(-floors)[:6]
        lines += ["", "Evaluation rule: median relative standard error per quantity, largest six: "
                  + ", ".join(f"{names[k]} {floors[k]:.1e}" for k in worst)
                  + f". Seconds per contract: median {np.median([r['seconds'] for r in evaluation.values()]):.1f}."]
    lines.append("")
    (OUTPUT/"report.md").write_text("\n".join(lines), encoding="utf-8")
    if step_rows:
        write_csv(OUTPUT/"time_step.csv", step_rows)
    write_metadata(OUTPUT, protocol_path(), pricing_sources()+[Path(__file__), STUDY, CONTRACTS,
                                                              ROOT/"src/neural/reference.py"])
    print(f"Saved {OUTPUT/'report.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="first N contracts only (a smoke run)")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    report() if args.report else run(args)


if __name__ == "__main__":
    main()
