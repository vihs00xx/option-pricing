"""Write results/neural/pinn_deeponet_v1/report.md and metadata.json.

Tables come from the saved results; the prose findings come from
experiments/neural/report_findings.md, written after reading the results.

Run from heston/: python -m experiments.neural.make_report
"""

import csv
import json

from experiments.neural.common import LADDER, RESULTS, SETTINGS_FILES, load_settings
from experiments.records import ROOT, digest, write_metadata
from src.heston.model import QUANTITIES

CLASSICAL = ROOT / "results/classical_v1"
FINDINGS = ROOT / "experiments/neural/report_findings.md"
SHOW = ("price", "delta", "gamma", "vega", "volga", "theta_cal", "d_xi")


def g(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x != x:
        return "-"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def note(stopped):
    """Ladder-row note from a level's stopped_early field (a string, None, or a list for ensembles)."""
    if isinstance(stopped, str) and stopped.startswith("["):
        stopped = [x for x in stopped.strip("[]").split(", ") if x not in ("None", "")]
    if isinstance(stopped, list):
        return "; ".join(x for x in stopped if x)
    return stopped or ""


def read_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def main():
    study = load_settings()
    sel = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    ladder = read_csv(RESULTS / "ladder.csv")
    tm = read_csv(RESULTS / "test_metrics.csv")
    checks = json.loads((RESULTS / "test_checks.json").read_text(encoding="utf-8"))
    nt = json.loads((RESULTS / "neural_timing.json").read_text(encoding="utf-8"))
    cost = json.loads((RESULTS / "training_cost.json").read_text(encoding="utf-8"))
    be = read_csv(RESULTS / "break_even.csv")
    eq = read_csv(RESULTS / "equal_accuracy.csv")
    cal = json.loads((RESULTS / "calibration.json").read_text(encoding="utf-8"))
    bench = read_csv(CLASSICAL / "benchmark.csv")
    ctime = json.loads((CLASSICAL / "timing.json").read_text(encoding="utf-8"))["seconds"]
    vcap = json.loads((RESULTS.parent / "variance_cap_study" / "summary.json").read_text(encoding="utf-8"))
    test_proto = json.loads((ROOT / "results/test_set_v1/protocol.json").read_text(encoding="utf-8"))
    p0 = json.loads((ROOT / "results/reference_check_v1/checks.json").read_text(encoding="utf-8"))
    test_proto_seal = (ROOT / "results/test_set_v1/protocol.sha256").read_text(encoding="utf-8").strip()[:16] + "..."
    sp, sd = sel["selected"]["pinn"], sel["selected"]["deeponet"]
    all_rows = {r["pricer"]: r for r in tm if r["split"] == "all"}

    L = ["# Neural pricers for one-stock Heston options: PINN and DeepONet", "",
         "Two neural pricers for European calls and puts under the Heston model, with the four Heston parameters "
         "as inputs so that one network covers the whole parameter range. Each is built as a ladder in which every "
         "level adds one technique and is scored on the same validation panels. The pre-declared rule selected "
         f"**`{sp['level']}`** for the PINN and **`{sd['level']}`** for the DeepONet; both were frozen by hash "
         f"({sel['frozen_at'][:16].replace('T', ' ')} UTC) before being scored once on the sealed test set "
         "`heston_test_v1`, then compared with COS, ADI, QE Monte Carlo and QE-RQMC.", "",
         "## Results at a glance", "",
         "Test RMSEs on all 252 cases, price and 11 sensitivities per unit strike; microseconds per contract for "
         "the price and all 11 sensitivities, one thread (networks in a batch of 4,096; classical methods per "
         "contract, quiet machine). Training seconds include the warm-start chain and the label building, scaled "
         "to an idle machine (see Break-even).", ""]
    head = ["Pricer", "Selection score", "Test score", "Price RMSE", "Delta RMSE", "Gamma RMSE", "Vega RMSE",
            "Theta RMSE", "us per contract", "Training + labels s"]
    rows = []
    for fam, s in (("pinn", sp), ("deeponet", sd)):
        r = all_rows[f"{fam}_selected"]
        rows.append([f"{'PINN' if fam == 'pinn' else 'DeepONet'} `{s['level']}`", g(s["selection_score"]), g(r["score"]),
                     g(r["rmse_price"]), g(r["rmse_delta"]), g(r["rmse_gamma"]), g(r["rmse_vega"]), g(r["rmse_theta_cal"]),
                     g(nt[f"{fam}_selected"]["batch_seconds"] * 1e6), g(cost["networks"][fam]["total_seconds"])])
    r = all_rows["bs_base"]
    rows.append(["Black-Scholes base (no training)", "", g(r["score"]), g(r["rmse_price"]), g(r["rmse_delta"]),
                 g(r["rmse_gamma"]), g(r["rmse_vega"]), g(r["rmse_theta_cal"]), g(nt["bs_base"]["batch_seconds"] * 1e6), "0"])
    for setting in ("cos_32", "cos_64", "cos_128", "cos_256", "adi_80x40x40", "adi_160x80x80", "mc_100000_dt32",
                    "mc_1000000_dt32", "rqmc_2^12x16_dt32", "rqmc_2^16x16_dt32"):
        b = next((x for x in bench if x["setting"] == setting), None)
        if b:
            t = g(ctime[setting] * 1e6)
            if setting + "_batch" in ctime:
                t += f" ({g(ctime[setting + '_batch'] * 1e6)} batched)"
            rows.append([f"`{setting}`", "", g(b["score"]), g(b["rmse_price"]), g(b["rmse_delta"]), g(b["rmse_gamma"]),
                         g(b["rmse_vega"]), g(b["rmse_theta_cal"]), t, "0"])
    L += table(head, rows)
    L += ["", "## Findings", "", FINDINGS.read_text(encoding="utf-8").strip(), ""]

    L += ["## Problem and model", "",
          "European calls and puts on one stock under",
          "", "    dS = (r - q) S dt + sqrt(v) S dW1,   dv = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1, W2> = rho dt,", "",
          "with constant r and q. A contract has nine inputs (S/K, tau, r, q, v0, kappa, theta, xi, rho) and a "
          "call/put flag; the pricers return the price and 11 sensitivities per unit strike: Delta, Gamma, Vega and "
          "Volga with respect to v0, Vanna (d2V/dS dv0), Theta (calendar, -dV/dtau per year), rate rho, and the "
          "derivatives with respect to kappa, theta, xi and rho.", "",
          "| Input | Training domain |", "|---|---|"]
    dom = json.loads((ROOT / "experiments/config/domain.json").read_text(encoding="utf-8"))["ranges"]
    for k, v in dom.items():
        if k != "tau_short":
            L.append(f"| {k} | {v[0]:g} to {v[1]:g} |")
    L += ["", f"The Feller condition 2 kappa theta >= xi^2 is not imposed: it fails for "
          f"{100 * p0['feller_violation_share_domain_sample']:.1f}% of a uniform sample of the domain, and for "
          f"{sum(test_proto['feller_violations'].values())} of the 252 test cases. The variance process has a heavy "
          f"upper tail at the domain's worst corner (v0 = theta = 0.36, kappa = 0.5, xi = 1, tau = 2): P(v_tau > v) is "
          f"1e-8 at v = {p0['7_v_max']['1e-08']['v_max']:.1f} and 1e-10 at v = {p0['7_v_max']['1e-10']['v_max']:.1f}, ten "
          "times the plan's first guess of 1 to 1.5. The ADI grid runs to v_max = 15.28; the networks stop at v_hi = 3 "
          "(next section).", "",
          "**Exact reduction.** With F = S e^{(r-q) tau} and x = log(F/K), V = K e^{-r tau} c(x, v, tau; kappa, theta, xi, rho), "
          "and the normalized call solves the driftless PDE", "",
          "    c_tau = 1/2 v (c_xx - c_x) + rho xi v c_xv + 1/2 xi^2 v c_vv + kappa (theta - v) c_v,   c(tau = 0) = (e^x - 1)^+,", "",
          "which at v = 0 degenerates to c_tau = kappa theta c_v. Puts follow from parity. Every sensitivity of a "
          "network comes from automatic differentiation of the price through this map (`price_and_greeks` in "
          "`src/neural/reference.py`).", "",
          "**Base.** Black-Scholes at the expected average variance w = theta + (v - theta)(1 - e^{-kappa tau})/(kappa tau), "
          "exact as xi -> 0, is the base the networks correct (output `bs`: c = BS + tau N; `bs_envelope`: "
          "c = BS + tau phi(d2) N).", "",
          "**Computational box.** x in [-3, 2.5] (the domain maps into [-0.52, 0.46]); tau from 0; v in [0, 1] for "
          "the levels before the variance boundary and [0, 3] from `P4`/`D4` on. The far-v condition c_v = 0 at "
          "v_hi = 3 was chosen on development contracts with the ADI solver at a fixed fine grid: against the full "
          "domain it moves prices at v0 <= 0.49 by at most "
          f"{vcap['difference_from_full_domain']['zero_slope_3.0']['price']:.0e} (Volga "
          f"{vcap['difference_from_full_domain']['zero_slope_3.0']['volga']:.0e}); at v_hi = 2 both c_v = 0 and the "
          f"Black-Scholes base value move prices by about {vcap['difference_from_full_domain']['base_value_2.0']['price']:.0e}, "
          "and at v_hi = 1 by about 4e-3 (`results/neural/variance_cap_study`). Collocation is packed near v = 0 "
          "(v = 3 u^2) and the networks see sqrt(v + 0.01): plain sqrt(v) would make c_v infinite at v = 0, where the "
          "degenerate equation needs it finite.", ""]

    L += ["## Data, seeds and selection", "",
          "| Set | Contracts | Seed | Use |", "|---|---:|---:|---|",
          f"| DeepONet labels | 32,768 (131,072 for the scale-up; 25% short maturity from D5) | {study['seeds']['data']} | training |",
          f"| PINN anchors | 4,096 | {study['seeds']['anchors']} | `P10_anchored` only |",
          f"| Monitor | 512 | {study['seeds']['monitor']} | checkpoint choice within a run |",
          f"| Validation | 4,096 | {study['seeds']['validation']} | selection |",
          f"| Short-maturity validation | 4,096 (tau 0.005-0.05) | {study['seeds']['short_validation']} | selection |",
          "| Test set `heston_test_v1` | 252 | 20261007 | final assessment, once |", "",
          "Labels come from the batched complex128 torch twin of the Lewis reference (Phase 0: it agrees with the "
          "NumPy reference to 8e-14 on all 12 quantities), which gives the call and all its derivatives by autograd.", "",
          study["selection_rule"], ""]

    # Black-Scholes base alone on both panels: the baseline every network must beat
    from experiments.neural.common import evaluation_set
    from experiments.neural.evaluate_test import base_price
    from src.neural.reference import price_and_greeks
    from src.neural.training import score as panel_score
    base = {}
    for kind in ("validation", "short_validation"):
        panel = evaluation_set(kind)
        base[kind] = panel_score(price_and_greeks(base_price, *panel["inputs"]), panel["reference"])
    variants = []
    for path in sorted((RESULTS / "variants").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        variants.append((path.stem.replace("_seed0", ""), r))

    L += ["## How the ladders changed during the run", "",
          "The ladders were changed mid-run after a diagnosis on the validation panels only; the test set stayed "
          f"sealed throughout (its seal `{test_proto_seal}` was checked again immediately before it was opened). The "
          "first PINN and DeepONet levels reached validation scores near 0.11 and short-maturity scores of 0.25 to "
          "0.36, worse than the Black-Scholes base alone on the short panel (table below). The per-quantity breakdown "
          "showed why: price, Delta and Theta were fine, but Volga, Vanna and the xi and rho sensitivities stayed "
          "close to the base's own errors, and on the short panel the networks did not move them at all (Vanna 0.187 "
          "and Volga 0.311 for five different models). The learning curves were flat at the end, so the wall-clock "
          "cap was not the limit.", "",
          "The cause is structural. The first outputs add a correction tau N (or tau phi(d_2) N) to Black-Scholes at "
          "the expected average variance, so the correction is O(tau) near expiry. Under Heston the departure from "
          "that base at short maturity is an implied-volatility skew and smile of order one, which is "
          "O(sqrt(tau) phi(d_2)) in price, and the same smile terms carry Volga, Vanna and the xi and rho "
          "sensitivities at every maturity. Two changes followed, each piloted on the validation panels: (1) an "
          "implied-volatility output c = BS(x, sqrt(w tau) e^N), whose first-order correction sqrt(w tau) phi(d_2) N "
          "fades like the time value; (2) kappa and theta entering only through the variance drift over the option's "
          "life, as inputs kappa tau and kappa theta tau for the PINN and as a tau-weighted second branch for the "
          "DeepONet, so that their sensitivities vanish at the true O(tau^1.5) rate instead of leaking at "
          "O(sqrt(tau)). They became the envelope levels `P9` and `D8` (and so `P10` and `D9`). The runs they "
          "replaced are kept as variants below. The variance-boundary level `P4` was also re-examined with 85% of "
          "the collocation at v <= 0.6: better than the original P4 but still worse than P3, so P4 was left as "
          "configured. The wall-clock caps were raised from 2 h / 4 h to 4 h / 6 h for the levels that ran after the "
          "change.", "",
          "Per-quantity RMSE / RMS(reference) on the validation panels for the Black-Scholes base and the levels "
          "that led to the change:", ""]
    diag_levels = [("Black-Scholes base", None)] + [(n, LADDER / f"{n}_seed0.json") for n in
                   ("P2_hard_payoff", "P10_anchored", "D3_differential", "D8_envelope")]
    diag_levels[2:2] = [("P9 tau-envelope (variant)", RESULTS / "variants/P9_envelope_tau_envelope_seed0.json")]
    rows = []
    for name, path in diag_levels:
        for kind in ("validation", "short_validation"):
            if path is None:
                ratios = base[kind][1]
            elif path.exists():
                ratios = json.loads(path.read_text(encoding="utf-8"))["scores"][kind]["ratios"]
            else:
                continue
            rows.append([name, "short" if kind == "short_validation" else "validation"] + [g(ratios[q]) for q in QUANTITIES])
    L += table(["Pricer", "Panel"] + list(QUANTITIES), rows) + [""]
    L += ["Pilots behind the change (5,000 Adam steps each; `variants/output_and_vmix_pilots.txt`):", "", "```"]
    L += (RESULTS / "variants/output_and_vmix_pilots.txt").read_text(encoding="utf-8").strip().splitlines()
    L += ["```", ""]

    for fam, title in (("pinn", "PINN ladder"), ("deeponet", "DeepONet ladder")):
        L += [f"## {title}", ""]
        head = ["Level", "Technique added", "Parameters", "Validation", "Short-maturity", "Selection",
                "Price RMSE", "Gamma RMSE", "Vega RMSE", "Theta RMSE", "Training s (wall)", "Note"]
        rows = []
        for r in ladder:
            if r["family"] != fam:
                continue
            rows.append([f"`{r['level']}`", r["technique"], g(r["parameters"]), g(r["validation_score"]),
                         g(r["short_validation_score"]), g(r["selection_score"]), g(r["rmse_price"]), g(r["rmse_gamma"]),
                         g(r["rmse_vega"]), g(r["rmse_theta_cal"]), g(r["training_seconds"]),
                         note(r["stopped_early"])])
        rows.append(["Black-Scholes base", "no network", "0", g(base["validation"][0]), g(base["short_validation"][0]),
                     g(0.5 * (base["validation"][0] + base["short_validation"][0])), "", "", "", "", "0", "baseline"])
        for name, r in variants:
            if r["family"] != fam:
                continue
            v, sh = r["scores"]["validation"], r["scores"]["short_validation"]
            rows.append([f"`{name}` (variant)", r["technique"], g(r["parameters"]), g(v["score"]), g(sh["score"]),
                         g(0.5 * (v["score"] + sh["score"])), g(v["rmse"]["price"]), g(v["rmse"]["gamma"]),
                         g(v["rmse"]["vega"]), g(v["rmse"]["theta_cal"]), g(r["total_seconds"]),
                         "replaced after the diagnosis; not eligible for selection"])
        L += table(head, rows) + [""]

    L += ["## Frozen test set", "", f"Selected pricers and the Black-Scholes base on every split (score = mean over "
          "the 12 quantities of RMSE / RMS(reference) within the split).", ""]
    head = ["Pricer", "Split", "Cases", "Score"] + [f"{q} RMSE" for q in SHOW] + ["max price error"]
    rows = []
    for name in ("pinn_selected", "deeponet_selected", "bs_base"):
        for r in tm:
            if r["pricer"] == name:
                rows.append([r["level"], r["split"], r["cases"], g(r["score"])] + [g(r[f"rmse_{q}"]) for q in SHOW]
                            + [g(r["max_price"])])
    L += table(head, rows)
    L += ["", "Every ladder level on all 252 cases (context only; the selection was frozen before):", ""]
    rows = [[r["level"], g(r["score"]), g(r["rmse_price"]), g(r["rmse_gamma"]), g(r["rmse_vega"])]
            for r in tm if r["split"] == "all"]
    L += table(["Level", "Test score", "Price RMSE", "Gamma RMSE", "Vega RMSE"], rows)

    L += ["", "## Benchmark, cost at equal accuracy and break-even", "",
          "Classical results on the same 252 cases are in `results/classical_v1/report.md`. For each classical "
          "method and quantity, the cheapest setting whose test RMSE is at most the network's:", ""]
    for fam in ("pinn", "deeponet"):
        L += [f"**{('PINN' if fam == 'pinn' else 'DeepONet')} `{sel['selected'][fam]['level']}`**", ""]
        methods = list(dict.fromkeys(r["method"] for r in eq if r["network"] == fam))
        rows = []
        for q in QUANTITIES:
            cells = []
            for m in methods:
                r = next(x for x in eq if x["network"] == fam and x["quantity"] == q and x["method"] == m)
                cells.append("none" if r["setting"].startswith("none") else f"`{r['setting']}` ({g(r['seconds_per_contract'])} s)")
            net_rmse = next(x for x in eq if x["network"] == fam and x["quantity"] == q)["network_rmse"]
            rows.append([q, g(net_rmse)] + cells)
        L += table(["Quantity", "Network RMSE"] + methods, rows) + [""]
    tc = cost["networks"]
    L += ["**Break-even.** N* = (training seconds + label seconds) / (classical seconds - network seconds) per "
          "contract, against the cheapest classical setting that matches the network on the price, and on all 12 "
          "quantities at once. "
          f"Training seconds are wall-clock: the ladders ran 3 levels at a time with 4 threads each next to the "
          f"classical benchmark, so each family's wall-clock is multiplied by a load factor measured by re-running "
          f"a pilot of one level alone (PINN {cost['load_factor']['pinn']['factor']:.2f}, DeepONet "
          f"{cost['load_factor']['deeponet']['factor']:.2f}; `load_factor.json`). Process CPU seconds (all threads) "
          "are in `training_cost.json` for the levels that recorded them; they are not used because a 4-thread "
          "process's CPU seconds overstate its wall-clock on an idle machine.", ""]
    rows = []
    for fam in ("pinn", "deeponet"):
        rows.append([f"{'PINN' if fam == 'pinn' else 'DeepONet'} `{tc[fam]['level']}`", " -> ".join(reversed(tc[fam]["chain"])),
                     g(tc[fam]["wall_seconds"]), g(tc[fam]["scaled_seconds"]), g(tc[fam]["label_seconds"]),
                     g(tc[fam]["total_seconds"])])
    L += table(["Network", "Training chain", "Wall s (under load)", "Scaled s", "Label s", "Total s"], rows) + [""]
    rows = []
    for r in be:
        t = g(r["classical_seconds"])
        if r.get("classical_batch_seconds") not in (None, "", r["classical_seconds"]):
            t += f" ({g(r['classical_batch_seconds'])} batched)"
        setting = f"`{r['setting']}`" if not r["setting"].startswith("none") else "none"
        rows.append([r["network"], r["method"], r["matched_on"], setting, t, r["break_even_batch"], r["break_even_single"]])
    L += table(["Network", "Method", "Matched on", "Classical setting", "Classical s per contract",
                "N* (network batched)", "N* (network one at a time)"], rows)
    L += ["", "Batched networks are compared with COS priced as a batch and with ADI, Monte Carlo and RQMC per "
          "contract; one-at-a-time networks with every method one contract at a time. 'Classical never matches': no "
          "setting run reaches the network's RMSE on every quantity; for Monte Carlo and RQMC the quantity they "
          "never reach is Volga (a bumped second difference)."]
    L += ["", "## Calibration test", "",
          f"Fit (v0, kappa, theta, xi, rho) to {cal['options']} out-of-the-money options (S/K 0.8 to 1.2, tau 0.1 to 2) "
          f"priced by the reference at {cal['true']}, r = {cal['market']['r']}, q = {cal['market']['q']}; "
          f"{cal['starts']} random starts in the training domain; box-constrained Levenberg-Marquardt on price "
          "residuals; one thread. COS Jacobians come from its own sensitivities, the networks' exactly from autograd.", ""]
    rows = [[k, g(v["seconds_mean"]), g(v["iterations_mean"]), g(v["median_max_rel_param_error"]),
             f"{v['recovered_within_1pct']}/{cal['starts']}", f"{v['recovered_within_10pct']}/{cal['starts']}",
             g(v["median_fit_rmse"])] for k, v in cal["summary"].items()]
    L += table(["Pricer", "Seconds per calibration", "Iterations", "Median max relative parameter error",
                "Within 1%", "Within 10%", "Median fit RMSE"], rows)
    cs = cal["summary"]
    cos_t = cs["cos_128"]["seconds_mean"]
    L.append("")
    for fam in ("pinn", "deeponet"):
        key = next(k for k in cs if k.startswith(fam))
        name = "PINN" if fam == "pinn" else "DeepONet"
        gain = cos_t - cs[key]["seconds_mean"]
        if cs[key]["recovered_within_10pct"] == 0:
            L.append(f"- {name}: no calibration break-even. It is {cos_t / cs[key]['seconds_mean']:.1f} times faster "
                     "per calibration than COS-128, but it recovers the parameters from none of the starts, so it "
                     "does not do the job COS does. If speed alone counted, training would pay back after "
                     f"{tc[fam]['total_seconds'] / gain:,.0f} calibrations.")
        else:
            n = f"{tc[fam]['total_seconds'] / gain:,.0f}" if gain > 0 else "never (COS is faster)"
            L.append(f"- {name}: training pays back against COS-128 after {n} calibrations.")
    L += ["", "## No-arbitrage and Greek validation", "",
          "Counts of violations (price below discounted intrinsic, Delta outside [0, e^{-q tau}] for calls or "
          "[-e^{-q tau}, 0] for puts, negative Gamma, negative Vega):", ""]
    rows = [[k, v["cases"], v["price_below_intrinsic"], v["delta_out_of_bounds"], v["negative_gamma"], v["negative_vega"]]
            for k, v in checks["no_arbitrage"].items()]
    L += table(["Pricer / panel", "Cases", "Price < intrinsic", "Delta out of bounds", "Gamma < 0", "Vega < 0"], rows)
    cov = checks.get("monte_carlo_coverage_95") or {}
    if cov:
        L += ["", "Share of test cases inside the Monte Carlo 95% interval (QE, dt 1/32), per quantity:", ""]
        rows = []
        for setting, per in cov.items():
            for who, vals in per.items():
                if vals:
                    rows.append([setting, who] + [g(vals[q]) for q in QUANTITIES])
        L += table(["Monte Carlo", "Values"] + list(QUANTITIES), rows)
    L += ["", "## Files and reproduction", "",
          "- `ladder/`: every level's checkpoint, log and record (settings, source hashes, scores, history).",
          "- `ladder.csv`, `selection.json` (frozen selection with hashes), `test_metrics.csv`, `test_predictions.csv`, "
          "`test_checks.json`, `neural_timing.json`, `equal_accuracy.csv`, `break_even.csv`, `training_cost.json`, "
          "`load_factor.json`, `calibration.csv`, `calibration.json`.",
          "- Reproduce (from heston/): `python -m experiments.neural.run_ladders`, then `select_models`, "
          "`evaluate_test`, `load_factor`, `compare`, `calibration`, `make_report`; verify with "
          "`python -m experiments.neural.verify_results`.", ""]
    (RESULTS / "report.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    srcs = sorted((ROOT / "src/neural").glob("*.py")) + sorted((ROOT / "experiments/neural").glob("*.py")) + list(SETTINGS_FILES)
    ladder_files = {p.name: digest(p) for p in sorted(LADDER.iterdir()) if p.is_file() and p.suffix != ".stdout"}
    write_metadata(RESULTS, srcs, extra=dict(ladder_files=ladder_files))
    print("wrote", RESULTS / "report.md")


if __name__ == "__main__":
    main()
