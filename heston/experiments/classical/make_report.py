"""Write results/classical_v1/report.md (Phase 1) from the benchmark, the quiet timings and the test set.

Run from heston/: python -m experiments.classical.make_report
"""

import csv
import json

import numpy as np

from experiments.records import ROOT, write_csv, write_metadata
from experiments.test_set.load import load_test_set
from src.heston.model import QUANTITIES

OUT = ROOT / "results/classical_v1"
LABEL = {"cos": "COS (c2 rule, L = 12)", "adi": "ADI (Hundsdorfer-Verwer)", "mc": "QE Monte Carlo", "rqmc": "QE-RQMC (Sobol, bridge, 16 scrambles)"}


def f(x):
    return f"{x:.2g}" if x == x else "-"


def time_step_bias(test, saved):
    """Per quantity: MC 1e6 paths at dt 1/8 minus dt 1/32 (same seeds differ, so the SEs add)."""
    rows = []
    ref = test["reference"]
    for q in QUANTITIES:
        a, b = saved["mc_1000000_dt8__" + q], saved["mc_1000000_dt32__" + q]
        sa, sb = saved["mc_1000000_dt8__se_" + q], saved["mc_1000000_dt32__se_" + q]
        d = a - b
        e8, e32 = a - ref[q], b - ref[q]
        both = (sa > 0) & (sb > 0)     # SE = 0 (no path in the money) is counted in benchmark.csv
        z = np.where(both, d / np.sqrt(np.where(both, sa**2 + sb**2, 1.0)), np.nan)
        z8 = np.where(sa > 0, e8 / np.where(sa > 0, sa, 1.0), np.nan)
        z32 = np.where(sb > 0, e32 / np.where(sb > 0, sb, 1.0), np.nan)
        rows.append(dict(quantity=q, median_abs_z_dt8=float(np.nanmedian(np.abs(z8))),
                         inside_dt8=float(np.nanmean(np.abs(z8) <= 1.96)),
                         median_abs_z_dt32=float(np.nanmedian(np.abs(z32))),
                         inside_dt32=float(np.nanmean(np.abs(z32) <= 1.96)),
                         median_abs_z_dt8_minus_dt32=float(np.nanmedian(np.abs(z))),
                         inside_dt8_minus_dt32=float(np.nanmean(np.abs(z) <= 1.96)),
                         mean_error_dt8=float(np.mean(e8)), mean_error_dt32=float(np.mean(e32)),
                         rmse_dt8=float(np.sqrt(np.mean(e8**2))), rmse_dt32=float(np.sqrt(np.mean(e32**2)))))
    return rows


def main():
    test = load_test_set()
    bench = list(csv.DictReader((OUT / "benchmark.csv").open(encoding="utf-8")))
    split = list(csv.DictReader((OUT / "benchmark_by_split.csv").open(encoding="utf-8")))
    timing = json.loads((OUT / "timing.json").read_text(encoding="utf-8"))
    saved = dict(np.load(OUT / "values.npz", allow_pickle=True))
    tsb = time_step_bias(test, saved)
    write_csv(OUT / "time_step_bias.csv", tsb)
    t = timing["seconds"]
    L = ["# Phase 1: classical benchmark on heston_test_v1", "",
         f"COS, ADI, QE Monte Carlo and QE-RQMC at every setting on the {len(test['case_id'])} sealed test cases "
         "(128 interpolation, 32 short maturity, 72 extrapolation, 20 anchors), price and 11 sensitivities per "
         "unit strike, against the Lewis reference. Settings were fixed in advance on the Phase 0 development "
         "contracts and were not tuned on the test cases. Seconds are per contract with all 12 quantities, one thread, "
         f"nothing else running, measured on {len(timing['cases'])} test cases spread evenly over maturity "
         "(`timing.json`; the benchmark workers' own seconds shared the CPU with the neural training).", "",
         "Score = mean over the 12 quantities of RMSE / RMS(reference), all cases.", "",
         "| Method | Setting | Score | Price RMSE | Delta RMSE | Gamma RMSE | Vega RMSE | Volga RMSE | Theta RMSE | d_xi RMSE | Seconds per contract |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in bench:
        L.append(f"| {LABEL[r['method']]} | `{r['setting']}` | {f(float(r['score']))} | {f(float(r['rmse_price']))} | "
                 f"{f(float(r['rmse_delta']))} | {f(float(r['rmse_gamma']))} | {f(float(r['rmse_vega']))} | "
                 f"{f(float(r['rmse_volga']))} | {f(float(r['rmse_theta_cal']))} | {f(float(r['rmse_d_xi']))} | "
                 f"{f(t.get(r['setting'], float('nan')))} |")
    cos_batch = {k: v for k, v in t.items() if k.startswith("cos") and k.endswith("_batch")}
    L += ["", "COS on the whole test set at once (vectorised): " +
          ", ".join(f"{k.replace('_batch', '')} {v * 1e6:.0f} us" for k, v in cos_batch.items()) + " per contract.",
          f" The NumPy Lewis reference with all 12 quantities takes {t['reference_numpy_12']:.4f} s per contract.", "",
          "## By split (score)", "", "| Setting | interpolation | short_maturity | extrapolation | anchor |", "|---|---:|---:|---:|---:|"]
    for r in bench:
        cells = [next(float(x["score"]) for x in split if x["setting"] == r["setting"] and x["split"] == s)
                 for s in ("interpolation", "short_maturity", "extrapolation", "anchor")]
        L.append(f"| `{r['setting']}` | " + " | ".join(f(c) for c in cells) + " |")
    L += ["", "## Monte Carlo and RQMC error bars", "",
          "| Setting | median abs z | share inside 1.96 SE | RMS z (outlier-dominated) | values with SE = 0 (of which error > 1e-10) |", "|---|---:|---:|---:|---:|"]
    for r in bench:
        if r["method"] in ("mc", "rqmc"):
            L.append(f"| `{r['setting']}` | {f(float(r['median_abs_z']))} | {f(float(r['inside_1p96']))} | {f(float(r['rms_z']))} | "
                     f"{r['zero_se_values']} ({r['zero_se_with_error_above_1e-10']}) |")
    L += ["", "RQMC caveat: the standard error over 16 scrambles is unreliable when a sensitivity comes from rare "
          "paths. On short maturities deep in or out of the money (for example Gamma of a deep in-the-money call "
          "with tau = 0.006, or Vanna of a deep in-the-money put), few or no scrambles hit the paths that cross the "
          "strike, so the estimate and its SE are both near zero while the reference is not, and z runs to "
          "hundreds or more. Plain Monte Carlo has the same SE = 0 cases at small path counts, and they disappear "
          "with more paths. With 2e5 paths it reproduces the RQMC outliers within its error bars. The RMS z "
          "column is dominated by these cases; the median abs z and the share inside 1.96 are not, and neither are "
          "the RMSEs.", "",
          "## Time-step bias (QE, 10^6 paths)", "",
          "| Quantity | dt 1/8 vs reference: median abs z / inside 1.96 | dt 1/32 vs reference | dt 1/8 minus dt 1/32 | RMSE dt 1/8 | RMSE dt 1/32 |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in tsb:
        L.append(f"| {r['quantity']} | {f(r['median_abs_z_dt8'])} / {f(r['inside_dt8'])} | "
                 f"{f(r['median_abs_z_dt32'])} / {f(r['inside_dt32'])} | "
                 f"{f(r['median_abs_z_dt8_minus_dt32'])} / {f(r['inside_dt8_minus_dt32'])} | "
                 f"{f(r['rmse_dt8'])} | {f(r['rmse_dt32'])} |")
    L += ["", "Without bias, the median abs z is about 0.67 and 95% lie inside 1.96. Larger values are bias: the time "
          "step, and for the bumped sensitivities the bump. The 'dt 1/8 minus dt 1/32' column compares the two step "
          "sizes directly, so it isolates the time-step part. Cases with SE = 0 are left out (counted above).", "",
          "## Settings", "",
          "- COS: 32, 64, 128, 256, 512, 1024 terms; truncation [c1 +- 12 sqrt(c2)] with exact cumulants, frozen on "
          "the development contracts (Phase 0); Delta, Gamma, Vega, Vanna, Volga from the expansion, rate rho by the "
          "chain rule, Theta from the PDE identity, kappa/theta/xi/rho by central bumps with [a, b] fixed.",
          "- ADI: Hundsdorfer-Verwer, theta = 1/2 + sqrt(3)/6, two Douglas (theta = 1) half-step pairs first; "
          "(S intervals, v intervals, time steps) = (40, 20, 20), (80, 40, 40), (160, 80, 80), chosen on development "
          "contracts; S grid sinh-clustered at the strike with the spot on a node; v grid sinh-clustered at 0 up to "
          "v_max = 15.28 (Phase 0) with v0 on a node; second-order upwinding of the v-convection above v = 1; "
          "V = 0 at S = 0, V_SS = 0 at S_max, the degenerate PDE at v = 0, V_v = 0 at v_max. Grid Greeks; rate rho "
          "and parameter sensitivities by central bumps re-solved on the same grid; Theta from the PDE.",
          "- QE Monte Carlo (Andersen 2008, psi_c = 1.5, martingale correction): 10^4, 10^5, 10^6 paths at dt = 1/8 "
          "and 1/32 (at least 2 steps). Pathwise Delta and rate rho; Gamma from pathwise Deltas at S +- 1%; Vega, "
          "Vanna, Volga, Theta and the parameter sensitivities by central bumps (2%; 0.02 for rho) with common "
          "random numbers.",
          "- QE-RQMC: scrambled Sobol, Brownian bridge for both Brownian motions, 16 scrambles of 2^8, 2^10, 2^12, "
          "2^14, 2^16 points, dt = 1/32; same estimators; standard errors over scrambles.", "",
          "Files: `benchmark.csv`, `benchmark_by_split.csv`, `values.npz` (every value and SE per case), "
          "`time_step_bias.csv`, `timing.json`, `settings.json`.", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    write_metadata(OUT, [ROOT / "experiments/classical/benchmark.py", ROOT / "experiments/classical/timing.py",
                         ROOT / "experiments/classical/make_report.py", ROOT / "src/heston/cos.py",
                         ROOT / "src/heston/adi.py", ROOT / "src/heston/monte_carlo.py"])
    print("wrote", OUT / "report.md")


if __name__ == "__main__":
    main()
