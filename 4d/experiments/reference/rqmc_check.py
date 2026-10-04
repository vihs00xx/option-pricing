"""Randomized QMC against the accepted Gauss-Hermite references.

Contracts: the 10 accepted contracts that took longest to accept, plus 10
other accepted contracts drawn at random. Each run is rqmc_estimate with 16
replicates, direction "volatility", greeks and the geometric control. Seeds:
404400 + 16 * contract_index + run (runs 0-12 at 4096 points, 13-15 at 65536).
The tail study pools price-only replicate means of 100 runs each for C* at
tau 2 (seeds 404750-404849) and the fastest-accepted random contract (seeds
404850-404949). Each run fits its own control coefficient, so pooled replicate
means are not quite identically distributed; 50 runs of C* at tau 2 without the
control (seeds 404950-404999) give a clean independent sample.
"""

from . import common  # noqa: F401  (pins BLAS threads before NumPy)

import argparse
import csv
from multiprocessing import Pool
from time import perf_counter

import numpy as np
from scipy import stats

from src.pricing import quantity_names, rqmc_estimate

from .common import OUTPUT, WORKERS, append_jsonl, load_contracts, read_jsonl, write_csv

RAW = OUTPUT / "rqmc.jsonl"
TAIL_RAW = OUTPUT / "rqmc_tails.jsonl"
CONFIGS = ((4096, range(0, 13)), (65536, range(13, 16)))
SELECTION_SEED = 404301
TAIL_RUNS = 100
TAIL_SEEDS = {"cstar_tau2": 404750, "easy": 404850}
NO_CONTROL_SEED, NO_CONTROL_RUNS = 404950, 50
GROUPS = ("price", "delta", "gamma", "vega", "theta", "rho", "correlation")


def group(name):
    return next(g for g in GROUPS if name.startswith(g))


def references():
    """{contract: (accepted nodes, values)} from the convergence study."""
    summary = list(csv.DictReader((OUTPUT / "summary_by_contract.csv").open(encoding="utf-8")))
    raw = {(r["contract"], r["nodes"]): r for r in read_jsonl(OUTPUT / "convergence.jsonl")
           if r["direction"] == "volatility"}
    accepted = {}
    for row in summary:
        if row["accepted_nodes"]:
            n = int(row["accepted_nodes"])
            accepted[row["contract"]] = dict(nodes=n, values=np.array(raw[(row["contract"], n)]["values"]),
                                             seconds=float(row["seconds_to_accept"]))
    return accepted


def select(accepted):
    slow = sorted(accepted, key=lambda name: -accepted[name]["seconds"])[:10]
    rest = sorted(set(accepted) - set(slow))
    rng = np.random.default_rng(SELECTION_SEED)
    chosen = slow + sorted(rng.choice(rest, 10, replace=False).tolist())
    return slow, chosen


def run(task):
    name, c, points, seed, greeks, control = task
    start = perf_counter()
    e = rqmc_estimate(c, points, 16, seed, direction="volatility", greeks=greeks, control=control)
    return dict(contract=name, points=points, seed=seed, greeks=greeks, control=control,
                seconds=perf_counter() - start,
                values=[e.values[k] for k in e.names], standard_errors=[e.standard_errors[k] for k in e.names],
                intervals=[list(e.intervals[k]) for k in e.names],
                replicate_means=e.replicate_means[:, 0].tolist() if not greeks else None)


def execute(tasks, path, workers):
    done = {(r["contract"], r["points"], r["seed"]) for r in read_jsonl(path)}
    todo = [t for t in tasks if (t[0], t[2], t[3]) not in done]
    start = perf_counter()
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run, todo), 1):
            append_jsonl(path, row)
            if i % 20 == 0 or i == len(todo):
                print(f"[{perf_counter() - start:7.1f}s] {path.name} {i}/{len(todo)}", flush=True)


def coverage_tables(accepted, slow, contracts):
    rows = [r for r in read_jsonl(RAW)]
    names = {name: quantity_names(c) for name, c, _ in contracts}
    overall, by_group, per_contract = [], [], []
    for points, _ in CONFIGS:
        sub = [r for r in rows if r["points"] == points]
        z, inside, groups, price_errors = [], [], [], []
        for r in sub:
            ref = accepted[r["contract"]]["values"]
            v, se, iv = np.array(r["values"]), np.array(r["standard_errors"]), np.array(r["intervals"])
            z.extend((v - ref) / se)
            inside.extend((iv[:, 0] <= ref) & (ref <= iv[:, 1]))
            groups.extend(group(n) for n in names[r["contract"]])
            price_errors.append(v[0] - ref[0])
        z, inside, groups = np.array(z), np.array(inside), np.array(groups)
        overall.append(dict(points=points, runs=len(sub), price_rmse=float(np.sqrt(np.mean(np.square(price_errors)))),
                            rms_z=float(np.sqrt(np.mean(z ** 2))), coverage_95=float(inside.mean()),
                            max_abs_z=float(np.abs(z).max()),
                            mean_seconds=float(np.mean([r["seconds"] for r in sub]))))
        for g in GROUPS:
            mask = groups == g
            by_group.append(dict(points=points, group=g, n=int(mask.sum()), rms_z=float(np.sqrt(np.mean(z[mask] ** 2))),
                                 coverage_95=float(inside[mask].mean()), max_abs_z=float(np.abs(z[mask]).max())))
        for name in dict.fromkeys(r["contract"] for r in sub):
            mine = [r for r in sub if r["contract"] == name]
            ref = accepted[name]["values"]
            zz = np.concatenate([(np.array(r["values"]) - ref) / np.array(r["standard_errors"]) for r in mine])
            per_contract.append(dict(points=points, contract=name, selection="slowest" if name in slow else "random",
                                     reference_nodes=accepted[name]["nodes"], runs=len(mine),
                                     price_rmse=float(np.sqrt(np.mean([(r["values"][0] - ref[0]) ** 2 for r in mine]))),
                                     mean_price_se=float(np.mean([r["standard_errors"][0] for r in mine])),
                                     rms_z=float(np.sqrt(np.mean(zz ** 2))), max_abs_z=float(np.abs(zz).max()),
                                     mean_seconds=float(np.mean([r["seconds"] for r in mine]))))
    write_csv(OUTPUT / "rqmc_coverage.csv", overall)
    write_csv(OUTPUT / "rqmc_coverage_by_group.csv", by_group)
    write_csv(OUTPUT / "rqmc_by_contract.csv", per_contract)
    return overall, by_group


def tail_table(accepted):
    rows = []
    for (name, control), rs in _group_by_contract(read_jsonl(TAIL_RAW)).items():
        means = np.concatenate([np.array(r["replicate_means"]) for r in rs])
        reference = accepted[name]["values"][0]
        # Standardise each run by its own replicate spread so that the per-run
        # control coefficient does not mix scales across runs.
        standardized = np.concatenate([(np.array(r["replicate_means"]) - np.mean(r["replicate_means"]))
                                       / np.std(r["replicate_means"], ddof=1) for r in rs])
        rng = np.random.default_rng(0)
        boot = [stats.kurtosis(rng.choice(means, means.size)) for _ in range(2000)]
        inside = [r["intervals"][0][0] <= reference <= r["intervals"][0][1] for r in rs]
        rows.append(dict(contract=name, control=control, runs=len(rs), replicate_means=means.size,
                         excess_kurtosis=float(stats.kurtosis(means)),
                         excess_kurtosis_ci95=f"{np.percentile(boot, 2.5):.2f} to {np.percentile(boot, 97.5):.2f}",
                         excess_kurtosis_standardized=float(stats.kurtosis(standardized)),
                         skewness=float(stats.skew(means)),
                         price_coverage_95=float(np.mean(inside)),
                         mean_minus_reference_over_se=float((np.mean([r["values"][0] for r in rs]) - reference)
                                                            / (np.std([r["values"][0] for r in rs], ddof=1)
                                                               / np.sqrt(len(rs))))))
    write_csv(OUTPUT / "rqmc_tails.csv", rows)
    return rows


def _group_by_contract(rows):
    out = {}
    for r in rows:
        out.setdefault((r["contract"], r.get("control", True)), []).append(r)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    contracts = load_contracts()
    by_name = {name: c for name, c, _ in contracts}
    accepted = references()
    slow, chosen = select(accepted)
    tasks = [(name, by_name[name], points, 404400 + 16 * i + run, True, True)
             for i, name in enumerate(chosen) for points, runs in CONFIGS for run in runs]
    # Big runs first so the pool does not end on them.
    tasks.sort(key=lambda t: -t[2])
    execute(tasks, RAW, args.workers)
    easy = min((n for n in accepted if n.startswith("random") and "_tau" not in n), key=lambda n: accepted[n]["seconds"])
    tail_tasks = [(name, by_name[name], 4096, TAIL_SEEDS[key] + run, False, True)
                  for key, name in (("cstar_tau2", "cstar_tau2"), ("easy", easy)) for run in range(TAIL_RUNS)]
    tail_tasks += [("cstar_tau2", by_name["cstar_tau2"], 4096, NO_CONTROL_SEED + run, False, False)
                   for run in range(NO_CONTROL_RUNS)]
    execute(tail_tasks, TAIL_RAW, args.workers)
    overall, by_group = coverage_tables(accepted, slow, contracts)
    for row in overall:
        print(row)
    for row in tail_table(accepted):
        print(row)
    print("slowest:", slow)
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
