"""Score every finished ladder level and the simulation pricers on the sealed test set.

Usage (from this folder): python -m experiments.evaluate_test [--workers 3] [--skip-classical]

Reads results/test_set_v1 (its seal is checked first) and every checkpoint in
results/neural/pinn_deeponet_v1/ladder. Each method gets, per split and overall, the score
(mean over price and every Greek of RMSE / RMS(reference)), the price RMSE and seconds per
contract. The classical rows are the engine at small budgets: conditional QE-RQMC with
8 x 1,024 Sobol paths (the training study's evaluation rule) and plain QE Monte Carlo with
8 x 1,024 pseudo-random paths, both at QE step 1/32. The reference noise floor
RMS(standard error) / RMS(reference) is reported alongside: no method can be told apart
from the references below it. Checkpoint hashes are written with the scores, and the
test set counts as observed from the first run on.
"""

import argparse
import json
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.neural.common import LADDER, STOCKS, keep_full_speed
from experiments.records import ROOT, digest, relative, write_csv
from src.heston_basket.engine import heston_estimate
from src.heston_basket.model import HestonBasket, quantity_names
from src.neural.coordinates import row_of
from src.neural.networks import load_model
from src.neural.reference import price_and_greeks

TEST = ROOT/"results/test_set_v1"
OUTPUT = ROOT/"results/neural/test_v1"
CLASSICAL = {"QE-RQMC 8 x 1024": dict(points=1024, replicates=8, dt=1/32, sampler="sobol"),
             "QE Monte Carlo 8 x 1024": dict(points=1024, replicates=8, dt=1/32, sampler="mc")}


def load_test():
    protocol = TEST/"protocol.json"
    if digest(protocol) != (TEST/"protocol.sha256").read_text(encoding="utf-8").strip():
        raise RuntimeError("test-set seal mismatch")
    frozen = json.loads(protocol.read_text(encoding="utf-8"))
    for name, key in (("cases.json", "cases_sha256"), ("references.csv", "references_sha256"),
                      ("reference_standard_errors.csv", "reference_standard_errors_sha256")):
        if digest(TEST/name) != frozen[key]:
            raise RuntimeError(f"{name} does not match the sealed protocol")
    cases = json.loads((TEST/"cases.json").read_text(encoding="utf-8"))
    read = lambda name: np.genfromtxt(TEST/name, delimiter=",", names=True, dtype=None, encoding="utf-8")
    names = quantity_names(STOCKS.d)
    refs, errors = read("references.csv"), read("reference_standard_errors.csv")
    reference = {q: np.asarray(refs[q], float) for q in names}
    standard_errors = {q: np.asarray(errors[q], float) for q in names}
    return cases, reference, standard_errors


def scores(values, reference, errors, splits):
    out = []
    for split in ["all"]+sorted(set(splits)):
        mask = np.ones(len(splits), bool) if split == "all" else np.asarray(splits) == split
        ratios = {q: float(np.sqrt(np.mean((values[q][mask]-reference[q][mask])**2))
                       / np.sqrt(np.mean(reference[q][mask]**2))) for q in reference}
        floor = {q: float(np.sqrt(np.mean(errors[q][mask]**2))/np.sqrt(np.mean(reference[q][mask]**2)))
                 for q in reference}
        out.append(dict(split=split, cases=int(mask.sum()), score=float(np.mean(list(ratios.values()))),
                        price_rmse=float(np.sqrt(np.mean((values["price"][mask]-reference["price"][mask])**2))),
                        price_ratio=ratios["price"], noise_floor=float(np.mean(list(floor.values())))))
    return out


def _classical(task):
    keep_full_speed()
    contract, settings, seed = task
    c = HestonBasket(**contract)
    start = perf_counter()
    e = heston_estimate(c, settings["points"], settings["replicates"], seed, dt=settings["dt"],
                        sampler=settings["sampler"])
    return [e.values[q] for q in e.names], perf_counter()-start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-classical", action="store_true")
    args = parser.parse_args()
    keep_full_speed()
    torch.set_num_threads(args.threads)
    cases, reference, errors = load_test()
    splits = [c["split"] for c in cases]
    x = np.array([row_of(HestonBasket(**c["contract"])) for c in cases])
    put = np.array([c["option"] == "put" for c in cases])
    names = quantity_names(STOCKS.d)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, checkpoints = [], {}
    for path in sorted(LADDER.glob("*_seed*.pt")):
        model = load_model(path)
        price_and_greeks(STOCKS, model.price, x[:8], put[:8])            # warm-up
        start = perf_counter()
        values = price_and_greeks(STOCKS, model.price, x, put, chunk=len(x))
        seconds = (perf_counter()-start)/len(x)
        checkpoints[relative(path)] = digest(path)
        family = "PINN" if path.name.startswith("P") else "DeepONet"
        result = scores(values, reference, errors, splits)
        rows += [dict(method=f"{family} {path.stem}", kind="neural", seconds_per_contract=seconds, **row)
                 for row in result]
        print(f"{path.stem}: score {result[0]['score']:.4g}", flush=True)
    classical_file = OUTPUT/"classical_values.json"
    if not args.skip_classical:
        saved = json.loads(classical_file.read_text(encoding="utf-8")) if classical_file.exists() else {}
        for label, settings in CLASSICAL.items():
            if label not in saved:
                tasks = [(c["contract"], settings, 7_000_000+k) for k, c in enumerate(cases)]
                with Pool(args.workers) as pool:
                    out = pool.map(_classical, tasks)
                saved[label] = dict(values=[o[0] for o in out], seconds=[o[1] for o in out], settings=settings)
                classical_file.write_text(json.dumps(saved), encoding="utf-8")
        for label, record in saved.items():
            matrix = np.array(record["values"])
            values = {q: matrix[:, k] for k, q in enumerate(names)}
            for row in scores(values, reference, errors, splits):
                rows.append(dict(method=label, kind="classical", seconds_per_contract=float(np.mean(record["seconds"])),
                                 **row))
    if not rows:
        raise SystemExit("nothing to score yet")
    write_csv(OUTPUT/"scores.csv", rows)
    overall = sorted((r for r in rows if r["split"] == "all"), key=lambda r: r["score"])
    lines = [f"# {STOCKS.d}-stock Heston basket: test-set scores", "",
             f"Scored {datetime.now(timezone.utc).isoformat()} on results/test_set_v1 ({len(cases)} cases). "
             "Score = mean over price and every Greek of RMSE / RMS(reference); noise floor = the same ratio for "
             "the references' own standard errors. Network seconds are per contract in one batch with every Greek; "
             "classical seconds per contract on one process.", "",
             "| Method | Score | Price RMSE | Seconds per contract | Noise floor |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['method']} | {r['score']:.4g} | {r['price_rmse']:.3g} | {r['seconds_per_contract']:.2e} | "
              f"{r['noise_floor']:.2g} |" for r in overall]
    lines += ["", "By split (score):", "", "| Method | " + " | ".join(sorted(set(splits))) + " |",
              "|---|" + "---:|"*len(set(splits))]
    for r in overall:
        per = {s["split"]: s["score"] for s in rows if s["method"] == r["method"]}
        lines.append(f"| {r['method']} | " + " | ".join(f"{per[s]:.4g}" for s in sorted(set(splits))) + " |")
    lines.append("")
    (OUTPUT/"report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT/"checkpoints.json").write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT/'report.md'}")


if __name__ == "__main__":
    main()
