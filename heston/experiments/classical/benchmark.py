"""Classical benchmark on the sealed test set: COS, ADI, QE Monte Carlo and QE-RQMC at every setting.

Run from heston/: python -m experiments.classical.benchmark [--workers 12] [--methods cos,adi,mc,rqmc]
Per-case rows stream to artifacts/classical_v1/<method>.jsonl (resumable); summary tables go to
results/classical_v1/. Settings are fixed here in advance (ADI grids chosen on development
contracts, see SETTINGS); nothing is tuned on the test cases. Seconds are per contract in a
single-threaded worker process (12 workers run side by side; quiet single-process timings
are measured separately by experiments.classical.timing).
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.records import ROOT, write_csv, write_json
from experiments.test_set.load import load_test_set
from src.heston.model import QUANTITIES, Contracts

WORK = ROOT / "artifacts" / "classical_v1"
OUT = ROOT / "results" / "classical_v1"
V_MAX = 15.28359698190767       # Phase 0 check 7, P(v_tau > v_max) <= 1e-10 at the domain's worst corner
SETTINGS = {
    "cos": [{"terms": n} for n in (32, 64, 128, 256, 512, 1024)],
    "adi": [{"m1": 40, "m2": 20, "steps": 20}, {"m1": 80, "m2": 40, "steps": 40}, {"m1": 160, "m2": 80, "steps": 80}],
    "mc": [{"paths": p, "dt": dt} for dt in (1 / 8, 1 / 32) for p in (10_000, 100_000, 1_000_000)],
    "rqmc": [{"log2_points": m, "dt": 1 / 32, "scrambles": 16} for m in (8, 10, 12, 14, 16)],
}


def setting_name(method, s):
    if method == "cos":
        return f"cos_{s['terms']}"
    if method == "adi":
        return f"adi_{s['m1']}x{s['m2']}x{s['steps']}"
    if method == "mc":
        return f"mc_{s['paths']}_dt{round(1 / s['dt'])}"
    return f"rqmc_2^{s['log2_points']}x{s['scrambles']}_dt{round(1 / s['dt'])}"


def run_job(job):
    method, setting, case_id, record, seed = job
    import torch
    torch.set_num_threads(1)
    c = Contracts.from_records([record])
    t = perf_counter()
    se = None
    if method == "adi":
        from src.heston.adi import GridSize, adi_quantities
        v = adi_quantities(c, GridSize(setting["m1"], setting["m2"], setting["steps"]), V_MAX)
        vals = {q: float(v[q][0]) for q in QUANTITIES}
    elif method == "mc":
        from src.heston.monte_carlo import mc_quantities
        vals, se = mc_quantities(c, setting["paths"], setting["dt"], seed)
    else:
        from src.heston.monte_carlo import rqmc_quantities
        vals, se = rqmc_quantities(c, setting["log2_points"], setting["dt"], seed, setting["scrambles"])
    return dict(method=method, setting=setting_name(method, setting), case_id=case_id, values=vals, se=se,
                seconds=perf_counter() - t)


def done_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                keys.add((r["setting"], r["case_id"]))
            except json.JSONDecodeError:
                pass
    return keys


def cos_rows(test):
    from src.heston.cos import cos_quantities
    rows = []
    c = test["contracts"]
    for s in SETTINGS["cos"]:
        cos_quantities(c.take([0]), s["terms"])          # warm-up
        t = perf_counter()
        v = cos_quantities(c, s["terms"])
        batch = (perf_counter() - t) / len(c)
        single = []
        for i in range(len(c)):
            t = perf_counter()
            cos_quantities(c.take([i]), s["terms"])
            single.append(perf_counter() - t)
        for i, cid in enumerate(test["case_id"]):
            rows.append(dict(method="cos", setting=setting_name("cos", s), case_id=cid,
                             values={q: float(v[q][i]) for q in QUANTITIES}, se=None,
                             seconds=single[i], batch_seconds=batch))
    return rows


def metrics(values, ref, mask):
    out = {}
    for q in QUANTITIES:
        e = np.abs(values[q][mask] - ref[q][mask])
        rms = np.sqrt(np.mean(ref[q][mask] ** 2))
        out[q] = dict(rmse=float(np.sqrt(np.mean(e**2))), median=float(np.median(e)), max=float(e.max()),
                      relative_rmse=float(np.sqrt(np.mean(e**2)) / rms) if rms > 0 else float("nan"))
    return out


def summarize(test, rows):
    ids = list(test["case_id"])
    pos = {cid: i for i, cid in enumerate(ids)}
    ref = test["reference"]
    by_setting = {}
    for r in rows:
        by_setting.setdefault((r["method"], r["setting"]), []).append(r)
    table, split_table = [], []
    saved = {}
    for (method, setting), rs in by_setting.items():
        if len(rs) < len(ids):
            continue
        vals = {q: np.full(len(ids), np.nan) for q in QUANTITIES}
        ses = {q: np.full(len(ids), np.nan) for q in QUANTITIES}
        secs = np.zeros(len(ids))
        for r in rs:
            i = pos[r["case_id"]]
            for q in QUANTITIES:
                vals[q][i] = r["values"][q]
                if r["se"]:
                    ses[q][i] = r["se"][q]
            secs[i] = r["seconds"]
        saved.update({f"{setting}__{q}": vals[q] for q in QUANTITIES})
        if method in ("mc", "rqmc"):
            saved.update({f"{setting}__se_{q}": ses[q] for q in QUANTITIES})
        for split in ("all", "interpolation", "short_maturity", "extrapolation", "anchor"):
            mask = np.ones(len(ids), bool) if split == "all" else test["split"] == split
            m = metrics(vals, ref, mask)
            score = float(np.mean([m[q]["relative_rmse"] for q in QUANTITIES]))
            row = dict(method=method, setting=setting, split=split, cases=int(mask.sum()), score=score,
                       seconds_per_contract=float(secs[mask].mean()),
                       batch_seconds_per_contract=float(np.mean([r.get("batch_seconds", np.nan) for r in rs])))
            for q in QUANTITIES:
                row[f"rmse_{q}"] = m[q]["rmse"]
                row[f"median_{q}"] = m[q]["median"]
                row[f"max_{q}"] = m[q]["max"]
            (table if split == "all" else split_table).append(row)
            if method in ("mc", "rqmc") and split == "all":
                err = np.array([vals[q] - ref[q] for q in QUANTITIES])
                se = np.array([ses[q] for q in QUANTITIES])
                ok = se > 0
                z = np.where(ok, err / np.where(ok, se, 1.0), np.nan)
                row["rms_z"] = float(np.sqrt(np.nanmean(z**2)))
                row["inside_1p96"] = float(np.nanmean(np.abs(z) <= 1.96))
                row["median_abs_z"] = float(np.nanmedian(np.abs(z)))
                row["zero_se_values"] = int((~ok).sum())
                row["zero_se_with_error_above_1e-10"] = int(((~ok) & (np.abs(err) > 1e-10)).sum())
    return table, split_table, saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--methods", default="cos,adi,rqmc,mc")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    test = load_test_set()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    records = test["contracts"].records()
    methods = [] if args.summarize_only else args.methods.split(",")
    if "cos" in methods:
        rows = cos_rows(test)
        with (WORK / "cos.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print("cos done", flush=True)
    for method in [m for m in methods if m != "cos"]:
        path = WORK / f"{method}.jsonl"
        have = done_keys(path)
        jobs = []
        for k, s in enumerate(SETTINGS[method]):
            for i, cid in enumerate(test["case_id"]):
                if (setting_name(method, s), cid) not in have:
                    jobs.append((method, s, cid, records[i], 20261007 + 100_000 * (k + 1) + i))
        jobs.sort(key=lambda j: -j[3]["tau"])          # long maturities first
        start = perf_counter()
        with ProcessPoolExecutor(args.workers) as pool, path.open("a", encoding="utf-8", newline="\n") as fh:
            for n, row in enumerate(pool.map(run_job, jobs, chunksize=1), 1):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if n % 50 == 0 or n == len(jobs):
                    print(f"{method}: {n}/{len(jobs)} [{perf_counter() - start:.0f}s]", flush=True)
    rows = []
    for method in ("cos", "adi", "mc", "rqmc"):
        path = WORK / f"{method}.jsonl"
        if path.exists():
            rows += [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    table, split_table, saved = summarize(test, rows)
    write_csv(OUT / "benchmark.csv", table)
    write_csv(OUT / "benchmark_by_split.csv", split_table)
    np.savez_compressed(OUT / "values.npz", case_id=np.array(test["case_id"]), **saved)
    write_json(OUT / "settings.json", {"settings": SETTINGS, "v_max": V_MAX})
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
