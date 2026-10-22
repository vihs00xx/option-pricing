"""Score every finished ladder level and two classical pricers on the sealed four-stock test set.

Usage (from 4d/): python -m experiments.evaluate_test [--workers 4] [--skip-classical]

Reads results/test_set_v1 (its seal is checked first) and every checkpoint in
results/neural/pinn_deeponet_v1/ladder. Each method gets, per split and overall, the score
(mean over price and the 26 Greeks of RMSE / RMS(reference)), the price RMSE and seconds per
contract. The references are converged Gauss-Hermite values, so they carry no noise floor.
The classical rows are conditional RQMC with the geometric control (16 x 4,096 Sobol points,
src/pricing) and plain Monte Carlo (100,000 pseudo-random paths, src/neural/monte_carlo).
Checkpoint hashes are written with the scores, and the test set counts as observed from the
first run on.
"""

import argparse
import json
from datetime import datetime, timezone
from multiprocessing import Pool
from time import perf_counter

import numpy as np
import torch

from experiments.neural.common import LADDER, STOCKS, keep_full_speed
from experiments.records import ROOT, digest, relative, write_csv
from src.neural.monte_carlo import monte_carlo_estimates
from src.neural.networks import load_model
from src.neural.reference import price_and_greeks, quantities
from src.pricing import Basket, rqmc_estimate

TEST = ROOT/"results/test_set_v1"
OUTPUT = ROOT/"results/neural/test_v1"
CLASSICAL = ("RQMC 16 x 4096 (control)", "Monte Carlo 100,000 paths")


def load_test():
    protocol = TEST/"protocol.json"
    if digest(protocol) != (TEST/"protocol.sha256").read_text(encoding="utf-8").strip():
        raise RuntimeError("test-set seal mismatch")
    frozen = json.loads(protocol.read_text(encoding="utf-8"))
    for name, key in (("cases.json", "cases_sha256"), ("references.csv", "references_sha256")):
        if digest(TEST/name) != frozen[key]:
            raise RuntimeError(f"{name} does not match the sealed protocol")
    cases = json.loads((TEST/"cases.json").read_text(encoding="utf-8"))
    refs = np.genfromtxt(TEST/"references.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    return cases, {q: np.asarray(refs[q], float) for q in quantities(STOCKS)}


def row(fields):
    """Contract row (STOCKS.fields order) of a case's contract fields."""
    d = STOCKS.d
    values = dict(strike=fields["strike"], tau=fields["tau"], rate=fields["rate"])
    for i in range(d):
        values.update({f"spot_{i+1}": fields["spots"][i], f"dividend_{i+1}": fields["dividends"][i],
                       f"vol_{i+1}": fields["volatilities"][i], f"weight_{i+1}": fields["weights"][i]})
    for i, j in STOCKS.pairs:
        values[f"correlation_{i+1}{j+1}"] = fields["correlation"][i][j]
    return np.array([values[f] for f in STOCKS.fields])


def scores(values, reference, splits):
    out = []
    for split in ["all"]+sorted(set(splits)):
        mask = np.ones(len(splits), bool) if split == "all" else np.asarray(splits) == split
        ratios = {q: float(np.sqrt(np.mean((values[q][mask]-reference[q][mask])**2))
                       / np.sqrt(np.mean(reference[q][mask]**2))) for q in reference}
        out.append(dict(split=split, cases=int(mask.sum()), score=float(np.mean(list(ratios.values()))),
                        price_rmse=float(np.sqrt(np.mean((values["price"][mask]-reference["price"][mask])**2))),
                        price_ratio=ratios["price"]))
    return out


def _classical(task):
    keep_full_speed()
    label, fields, seed = task
    start = perf_counter()
    if label == CLASSICAL[0]:
        e = rqmc_estimate(Basket(**fields), 4096, 16, seed, greeks=True, control=True)
        values = [e.values[q] for q in quantities(STOCKS)]
    else:
        e = monte_carlo_estimates(STOCKS, row(fields), fields["option"] == "put", 100_000, seed)
        values = [e[q][0] for q in quantities(STOCKS)]
    return values, perf_counter()-start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-classical", action="store_true")
    args = parser.parse_args()
    keep_full_speed()
    torch.set_num_threads(args.threads)
    cases, reference = load_test()
    splits = [c["split"] for c in cases]
    x = np.array([row(c["contract"]) for c in cases])
    put = np.array([c["option"] == "put" for c in cases])
    names = quantities(STOCKS)
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
        result = scores(values, reference, splits)
        rows += [dict(method=f"{family} {path.stem}", kind="neural", seconds_per_contract=seconds, **r) for r in result]
        print(f"{path.stem}: score {result[0]['score']:.4g}", flush=True)
    classical_file = OUTPUT/"classical_values.json"
    if not args.skip_classical:
        saved = json.loads(classical_file.read_text(encoding="utf-8")) if classical_file.exists() else {}
        for label in CLASSICAL:
            if label not in saved:
                tasks = [(label, c["contract"], 7_000_000+k) for k, c in enumerate(cases)]
                with Pool(args.workers) as pool:
                    out = pool.map(_classical, tasks)
                saved[label] = dict(values=[o[0] for o in out], seconds=[o[1] for o in out])
                classical_file.write_text(json.dumps(saved), encoding="utf-8")
        for label, record in saved.items():
            matrix = np.array(record["values"])
            values = {q: matrix[:, k] for k, q in enumerate(names)}
            rows += [dict(method=label, kind="classical", seconds_per_contract=float(np.mean(record["seconds"])), **r)
                     for r in scores(values, reference, splits)]
    if not rows:
        raise SystemExit("nothing to score yet")
    write_csv(OUTPUT/"scores.csv", rows)
    overall = sorted((r for r in rows if r["split"] == "all"), key=lambda r: r["score"])
    lines = ["# Four-stock basket: test-set scores", "",
             f"Scored {datetime.now(timezone.utc).isoformat()} on results/test_set_v1 ({len(cases)} cases). "
             "Score = mean over price and the 26 Greeks of RMSE / RMS(reference); references are converged "
             "Gauss-Hermite values. Network seconds are per contract in one batch with every Greek; classical "
             "seconds per contract on one process.", "",
             "| Method | Score | Price RMSE | Seconds per contract |", "|---|---:|---:|---:|"]
    lines += [f"| {r['method']} | {r['score']:.4g} | {r['price_rmse']:.3g} | {r['seconds_per_contract']:.2e} |"
              for r in overall]
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
