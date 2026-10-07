"""Phase 0 reference checks 1-8 on the development contracts (seed 20261006).

Run from heston/:  python -m experiments.reference.check_reference [--workers 12]
Writes results/reference_check_v1/{report.md, checks.json, summary_by_contract.csv}.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.stats import ncx2

from src.heston import torch_reference as tr
from src.heston.black_scholes import bs_price
from src.heston.cos import cos_quantities
from src.heston.model import (HARD, PARAMETERS, QUANTITIES, Contracts, average_variance,
                              development_contracts, feller_violated, load_domain)
from src.heston.monte_carlo import qe_price
from src.heston.reference import gil_pelaez_price, reference, richardson

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reference_check_v1"
SEED = 20261006


def _chunks(n, k):
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _ref_worker(args):
    records, full, keep = args
    c = Contracts.from_records(records)
    out, info = reference(c, full=full, keep_nodes=keep)
    return out, info


def parallel_reference(c, workers, full=True, keep_nodes=False):
    parts = _chunks(len(c), workers * 4)
    jobs = [(c.take(p).records(), full, keep_nodes) for p in parts]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_ref_worker, jobs))
    out = {q: np.concatenate([r[0][q] for r in results]) for q in results[0][0]}
    info = {}
    for k in results[0][1]:
        if k == "nodes":
            info[k] = [nd for r in results for nd in r[1][k]]
        else:
            info[k] = np.concatenate([np.atleast_1d(r[1][k]) for r in results])
    return out, info


def _gp_worker(records):
    c = Contracts.from_records(records)
    return [gil_pelaez_price(c.take([i])) for i in range(len(c))]


def _mc_worker(args):
    record, n_paths, steps, seed = args
    c = Contracts.from_records([record])
    t = time.perf_counter()
    mean, se, fails = qe_price(c, n_paths, steps, np.random.default_rng(seed))
    return mean, se, fails, time.perf_counter() - t


def quantlib_prices(c):
    import QuantLib as ql
    today = ql.Date(6, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    prices, taus = [], []
    for i in range(len(c)):
        days = max(1, int(round(c.tau[i] * 365)))
        rts = ql.YieldTermStructureHandle(ql.FlatForward(today, float(c.r[i]), dc, ql.Continuous))
        qts = ql.YieldTermStructureHandle(ql.FlatForward(today, float(c.q[i]), dc, ql.Continuous))
        proc = ql.HestonProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(float(c.S[i]))),
                                float(c.v0[i]), float(c.kappa[i]), float(c.theta[i]),
                                float(c.xi[i]), float(c.rho[i]))
        engine = ql.AnalyticHestonEngine(
            ql.HestonModel(proc), ql.AnalyticHestonEngine.Gatheral,
            ql.AnalyticHestonEngine_Integration.gaussLobatto(1e-13, 1e-13, 10_000_000))
        kind = ql.Option.Call if c.call[i] else ql.Option.Put
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(kind, 1.0), ql.EuropeanExercise(today + days))
        opt.setPricingEngine(engine)
        prices.append(opt.NPV())
        taus.append(days / 365)
    return np.array(prices), np.array(taus), ql.__version__


def _max(a):
    a = np.abs(np.asarray(a))
    return float(np.nanmax(a)) if a.size else float("nan")


def _argmax(a):
    return int(np.nanargmax(np.abs(a)))


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--mc-paths", type=int, default=1 << 20)
    args = ap.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    main_c, short_c, hard_c = development_contracts()
    allc = Contracts.concat(main_c, short_c, hard_c)
    split = np.array(["domain"] * len(main_c) + ["short"] * len(short_c) + ["hard"] * len(hard_c))
    names = [f"domain{i:04d}" for i in range(len(main_c))] + \
            [f"short{i:03d}" for i in range(len(short_c))] + list(HARD)
    n = len(allc)
    checks = {"study": "reference_check_v1", "seed": SEED, "contracts": {
        "domain": len(main_c), "short": len(short_c), "hard": len(hard_c)}}
    log(f"{n} development contracts")

    # ---- Reference (NumPy, all 12 quantities, per-contract ladder) -------------------
    t = time.perf_counter()
    ref, rinfo = parallel_reference(allc, args.workers, keep_nodes=True)
    log(f"reference done in {time.perf_counter() - t:.0f}s wall ({args.workers} workers)")
    levels, counts = np.unique(rinfo["level"], return_counts=True)
    bad = ~rinfo["converged"]
    checks["reference"] = {
        "agreement_target": 1e-9,
        "levels_used": {int(l): int(k) for l, k in zip(levels, counts)},
        "max_last_change": _max(rinfo["change"]),
        "non_converged": [names[i] for i in np.nonzero(bad)[0]],
        "U_percentiles": {p: float(np.percentile(rinfo["U"], p)) for p in (50, 90, 99, 100)},
        "panels_percentiles": {p: float(np.percentile(rinfo["panels"], p)) for p in (50, 90, 99, 100)},
        "worst_quantity_counts": {q: int((rinfo["worst_quantity"] == q).sum()) for q in QUANTITIES},
        "parity_note": "puts come from calls by parity in closed form, so parity holds to rounding by construction; check 1 prices puts independently",
    }

    # ---- 1. Lewis vs Gil-Pelaez --------------------------------------------------------
    t = time.perf_counter()
    jobs = [allc.take(p).records() for p in _chunks(n, args.workers * 4)]
    with ProcessPoolExecutor(args.workers) as pool:
        gp = np.concatenate([np.array(r) for r in pool.map(_gp_worker, jobs)])
    d1 = gp - ref["price"]
    log(f"check 1 done in {time.perf_counter() - t:.0f}s, max |Lewis - GP| = {_max(d1):.2e}")
    checks["1_lewis_vs_gil_pelaez"] = {
        "max_abs_diff": _max(d1), "by_split": {s: _max(d1[split == s]) for s in ("domain", "short", "hard")},
        "worst": names[_argmax(d1)], "integrator": "scipy.integrate.quad, epsabs 1e-14, epsrel 1e-13"}

    # ---- 2. Black-Scholes limit ---------------------------------------------------------
    t = time.perf_counter()
    base = main_c.take(np.arange(64))
    bs_rows = []
    for rho_mode in ("zero", "domain"):
        for xi in (1e-2, 1e-3, 1e-4):
            cc = base.with_values(xi=xi, **({"rho": 0.0} if rho_mode == "zero" else {}))
            pr, _ = reference(cc, full=False)
            w = average_variance(cc.v0, cc.theta, cc.kappa, cc.tau)
            err = pr["price"] - bs_price(cc.x, w, cc.tau, cc.r, cc.call)
            bs_rows.append({"rho": rho_mode, "xi": xi, "max_abs_err": _max(err),
                            "max_rel_err": _max(err / np.maximum(np.abs(pr["price"]), 1e-300))})
    for mode in ("zero", "domain"):
        rows = [r for r in bs_rows if r["rho"] == mode]
        for a, b in zip(rows[:-1], rows[1:]):
            b["ratio_to_previous"] = a["max_abs_err"] / b["max_abs_err"]
    checks["2_black_scholes_limit"] = {"contracts": 64, "rows": bs_rows,
                                       "expected_ratio": {"zero": 100, "domain": 10}}
    log(f"check 2 done in {time.perf_counter() - t:.0f}s")

    # ---- 3. COS ------------------------------------------------------------------------
    cos_res = {}
    for rule, L in (("c2", 12.0), ("c4", 10.0)):
        for N in (256, 512):
            co = cos_quantities(allc, N, rule=rule, L=L, parameter_greeks=False)
            cos_res[(rule, N)] = co
            row = {}
            for q in ("price", "delta", "gamma"):
                e = co[q] - ref[q]
                row[q] = {"max": _max(e), "worst": names[_argmax(e)],
                          "by_split": {s: _max(e[split == s]) for s in ("domain", "short", "hard")},
                          "share_below_1e-9": float(np.mean(np.abs(e) <= 1e-9))}
            checks.setdefault("3_cos", {})[f"{rule}_L{L:g}_{N}"] = row
    checks["3_cos"]["truncation"] = ("c2: L = 12 standard deviations of X around its mean (exact variance); "
                                     "c4: L = 10 times sqrt(c2 + sqrt(c4)) (Fang and Oosterlee 2008), cumulants "
                                     "from a contour integral of log phi")
    log("check 3 done")

    # ---- 4. QuantLib -------------------------------------------------------------------
    try:
        ql_idx = np.concatenate([np.arange(180), len(main_c) + np.arange(20)])
        qc = allc.take(ql_idx)
        qp, qtau, qlv = quantlib_prices(qc)
        qc = qc.with_values(tau=qtau)
        qref, _ = reference(qc, full=False)
        d4 = qp - qref["price"]
        checks["4_quantlib"] = {"version": qlv, "contracts": len(qc), "max_abs_diff": _max(d4),
                                "worst": names[ql_idx[_argmax(d4)]],
                                "engine": "AnalyticHestonEngine, Gatheral (little trap) form, Gauss-Lobatto rel/abs 1e-13",
                                "note": "tau rounded to whole days (Actual/365F); the reference is recomputed at that tau"}
        log(f"check 4 done, max |QL - ref| = {_max(d4):.2e}")
    except ImportError:
        checks["4_quantlib"] = {"skipped": "QuantLib not importable"}

    # ---- 5. QE Monte Carlo -------------------------------------------------------------
    rng = np.random.default_rng(SEED + 5)
    mc_idx = list(rng.choice(len(main_c), 24, replace=False)) + \
        [len(main_c) + len(short_c) + list(HARD).index(k) for k in
         ("feller_strong_atm", "feller_strong_skew_put", "rho_m09_long_itm_call",
          "v0_high_theta_low_fast", "v0_high_theta_low_slow_otm", "short_atm_max_gamma")]
    mc_jobs = []
    for j, i in enumerate(mc_idx):
        steps = max(32, int(np.ceil(256 * allc.tau[i])))
        mc_jobs.append((allc.take([i]).records()[0], args.mc_paths, steps, SEED + 1000 + j))
    t = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        mc = list(pool.map(_mc_worker, mc_jobs))
    z = np.array([(m[0] - ref["price"][i]) / m[1] if m[1] > 0 else 0.0 for m, i in zip(mc, mc_idx)])
    checks["5_qe_monte_carlo"] = {
        "paths": args.mc_paths, "steps_rule": "max(32, ceil(256 tau))", "psi_c": 1.5,
        "contracts": len(mc_idx), "coverage_95": float(np.mean(np.abs(z) <= 1.96)),
        "coverage_99": float(np.mean(np.abs(z) <= 2.576)), "rms_z": float(np.sqrt(np.mean(z**2))),
        "max_abs_z": _max(z), "mean_z": float(z.mean()),
        "martingale_correction_failures": int(sum(m[2] for m in mc)),
        "rows": [{"contract": names[i], "tau": float(allc.tau[i]), "steps": job[2], "mc": m[0], "se": m[1],
                  "reference": float(ref["price"][i]), "z": float(zz), "seconds": m[3]}
                 for i, m, zz, job in zip(mc_idx, mc, z, mc_jobs)]}
    log(f"check 5 done in {time.perf_counter() - t:.0f}s, rms z {checks['5_qe_monte_carlo']['rms_z']:.2f}")

    # ---- 6. Torch twin, finite differences, Theta ---------------------------------------
    torch.set_num_threads(max(1, args.workers))
    t = time.perf_counter()
    tw, tinfo = tr.twin(allc)
    torch.set_num_threads(1)
    log(f"torch twin ladder done in {time.perf_counter() - t:.0f}s ({args.workers} threads)")
    diff6 = {q: tw[q] - ref[q] for q in QUANTITIES}
    nodes = tinfo["nodes"]
    checks["6_torch_twin"] = {
        "max_abs_diff": {q: _max(diff6[q]) for q in QUANTITIES},
        "worst_contract": {q: names[_argmax(diff6[q])] for q in QUANTITIES},
        "ladder_non_converged": [names[i] for i in np.nonzero(~tinfo["converged"])[0]],
        "nodes_needed": {s: {str(k): int(v) for k, v in zip(*np.unique(nodes[split == s], return_counts=True))}
                         for s in ("domain", "short", "hard")},
        "nodes_percentiles_domain": {p: float(np.percentile(nodes[split == "domain"], p)) for p in (50, 90, 99, 100)},
    }
    # Richardson FD on a subset with the reference's fixed final node sets
    fd_idx = list(range(0, len(main_c), 16)) + list(range(len(main_c), len(main_c) + len(short_c), 8)) + \
        list(range(len(main_c) + len(short_c), n))
    steps = {"kappa": 1e-3, "theta": 1e-4, "xi": 1e-3, "rho": 1e-3, "r": 1e-4}
    fd = {f"d_{p}": [] for p in PARAMETERS}
    fd["rho_rate"], fd["theta_cal"] = [], []
    for i in fd_idx:
        c1 = allc.take([i])
        nd, wt = rinfo["nodes"][i]
        for p in PARAMETERS:
            fd[f"d_{p}"].append(richardson(c1, p, steps[p], nd, wt))
        fd["rho_rate"].append(richardson(c1, "r", steps["r"], nd, wt))
        fd["theta_cal"].append(-richardson(c1, "tau", min(1e-4, 0.05 * c1.tau[0]), nd, wt))
    fd_rows = {}
    for q, vals in fd.items():
        vals = np.array(vals)
        fd_rows[q] = {"torch_vs_fd": _max(tw[q][fd_idx] - vals), "numpy_vs_fd": _max(ref[q][fd_idx] - vals),
                      "worst": names[fd_idx[_argmax(ref[q][fd_idx] - vals)]]}
    checks["6_torch_twin"]["finite_differences"] = {
        "contracts": len(fd_idx), "steps": steps, "theta_step": "min(1e-4, tau/20)", "rows": fd_rows,
        "method": "Richardson central differences of the NumPy price on the reference's final node set"}
    log("check 6 done")

    # ---- 7. v_max ----------------------------------------------------------------------
    rng_ = load_domain()["ranges"]
    grids = {k: np.linspace(*rng_[k], 6) for k in ("v0", "theta", "kappa", "xi")}
    taus = np.concatenate([np.linspace(*rng_["tau_short"], 4), np.linspace(*rng_["tau"], 16)])
    V0, TH, KA, XI, TA = np.meshgrid(grids["v0"], grids["theta"], grids["kappa"], grids["xi"], taus, indexing="ij")
    om = -np.expm1(-KA * TA)
    cfac = XI**2 * om / (4 * KA)
    df = 4 * KA * TH / XI**2
    nc = 4 * KA * np.exp(-KA * TA) * V0 / (XI**2 * om)
    vmax = {}
    for eps in (1e-8, 1e-10):
        q = cfac * ncx2.isf(eps, df, nc)
        k = np.unravel_index(np.nanargmax(q), q.shape)
        vm = float(q[k])
        corner = {"v0": float(V0[k]), "theta": float(TH[k]), "kappa": float(KA[k]), "xi": float(XI[k]), "tau": float(TA[k])}
        tail = float(ncx2.sf(vm / cfac[k], df[k], nc[k]))
        vmax[str(eps)] = {"v_max": vm, "worst_corner": corner, "tail_at_v_max": tail}
    checks["7_v_max"] = {"method": "max over a 6^4 x 20 grid of the CIR terminal quantile c * ncx2.isf(eps, 4 kappa theta/xi^2, lambda)",
                         **vmax}
    log(f"check 7: {vmax}")

    # ---- 8. Timing (one thread) --------------------------------------------------------
    tm_idx = np.arange(0, len(main_c), 20)
    tc = main_c.take(tm_idx)
    t = time.perf_counter()
    _, tm_info = reference(tc)
    numpy_per = (time.perf_counter() - t) / len(tc)
    t = time.perf_counter()
    reference(tc, full=False)
    numpy_price_per = (time.perf_counter() - t) / len(tc)
    batch = main_c.take(np.arange(4096))
    timing_nodes = int(np.percentile(nodes[split == "domain"], 99))
    U_b = tr.truncation_point(batch, tr.TRUNCATION)
    t = time.perf_counter()
    tr.quantities(batch, timing_nodes // 16, U_b)
    torch_fixed = (time.perf_counter() - t) / len(batch)
    t = time.perf_counter()
    _, tl_info = tr.twin(batch)
    torch_ladder = (time.perf_counter() - t) / len(batch)
    co_t = {}
    for N in (256, 512):
        t = time.perf_counter()
        cos_quantities(batch, N, rule="c2", L=12.0, parameter_greeks=False)
        co_t[f"batch_{N}"] = (time.perf_counter() - t) / len(batch)
        one = batch.take([0])
        t = time.perf_counter()
        for _ in range(200):
            cos_quantities(one, N, rule="c2", L=12.0, parameter_greeks=False)
        co_t[f"single_{N}"] = (time.perf_counter() - t) / 200
    checks["8_timing_seconds_per_contract"] = {
        "numpy_reference_12_quantities_with_ladder": numpy_per,
        "numpy_reference_price_only_with_ladder": numpy_price_per,
        "numpy_contracts_timed": len(tc),
        "torch_twin_fixed_nodes": {"nodes": timing_nodes, "seconds": torch_fixed, "batch": len(batch)},
        "torch_twin_ladder": {"seconds": torch_ladder, "batch": len(batch)},
        "cos_price_delta_gamma": co_t, "threads": 1}
    log(f"check 8: numpy {numpy_per:.4f}s, torch fixed {torch_fixed:.5f}s, ladder {torch_ladder:.5f}s, cos {co_t}")

    # ---- Domain facts and hardest regions ------------------------------------------------
    fel = feller_violated(main_c)
    checks["feller_violation_share_domain_sample"] = float(fel.mean())
    checks["feller_violation_share_short_sample"] = float(feller_violated(short_c).mean())
    tau_bins = [0.005, 0.05, 0.25, 1.0, 2.0]
    xi_bins = [0.1, 0.4, 0.7, 1.0]
    region = []
    for a, b in zip(tau_bins[:-1], tau_bins[1:]):
        for c_, d_ in zip(xi_bins[:-1], xi_bins[1:]):
            m = (allc.tau >= a) & (allc.tau < b + 1e-12) & (allc.xi >= c_) & (allc.xi <= d_) & (split != "hard")
            if m.any():
                region.append({"tau": [a, b], "xi": [c_, d_], "n": int(m.sum()),
                               "reference_U_median": float(np.median(rinfo["U"][m])),
                               "torch_nodes_max": int(nodes[m].max()),
                               "torch_nodes_median": float(np.median(nodes[m])),
                               "cos256_price_err_max": _max((cos_res[("c2", 256)]["price"] - ref["price"])[m]),
                               "cos256_gamma_err_max": _max((cos_res[("c2", 256)]["gamma"] - ref["gamma"])[m])})
    checks["regions_tau_by_xi"] = region

    meta = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_head": git("rev-parse", "HEAD"), "git_head_subject": git("log", "-1", "--format=%s"),
            "platform": platform.platform(), "processor": platform.processor(),
            "packages": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "torch": torch.__version__},
            "workers": args.workers}
    checks["metadata"] = meta
    with open(OUT / "checks.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(checks, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(OUT / "summary_by_contract.csv", "w", encoding="utf-8", newline="\n") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["name", "split", "S", "tau", "r", "q", "v0", "theta", "kappa", "xi", "rho", "call", "feller_violated",
                     *QUANTITIES, "ref_level", "ref_change", "ref_U", "torch_nodes", "torch_max_diff",
                     "gp_diff", "cos256_price_err", "cos256_delta_err", "cos256_gamma_err"])
        fel_all = feller_violated(allc)
        tmax = np.max([np.abs(diff6[q]) for q in QUANTITIES], axis=0)
        for i in range(n):
            wr.writerow([names[i], split[i], *[f"{getattr(allc, f)[i]:.17g}" for f in
                                                  ("S", "tau", "r", "q", "v0", "theta", "kappa", "xi", "rho")],
                         int(allc.call[i]), int(fel_all[i]), *[f"{ref[q][i]:.17g}" for q in QUANTITIES],
                         rinfo["level"][i], f"{rinfo['change'][i]:.3g}", f"{rinfo['U'][i]:.6g}", nodes[i],
                         f"{tmax[i]:.3g}", f"{d1[i]:.3g}",
                         *[f"{cos_res[('c2', 256)][q][i] - ref[q][i]:.3g}" for q in ("price", "delta", "gamma")]])
    with open(OUT / "run.log", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log_lines) + "\n")
    log("written checks.json, summary_by_contract.csv, run.log")


if __name__ == "__main__":
    main()
