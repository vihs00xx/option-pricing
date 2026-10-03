"""Write results/neural/two_asset_v1/report.md from the saved evaluation outputs."""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"results/neural/two_asset_v1"
LEVELS = ROOT/"experiments/config/neural/basket_neural_levels.json"
SETTINGS = ROOT/"experiments/config/neural/basket_neural_settings.json"
GREEKS = ("delta_1", "delta_2", "gamma_11", "gamma_22", "gamma_12", "vega_1", "vega_2", "theta", "rho",
          "correlation")
LABELS = {"price": "Price", "delta_1": "Delta 1", "delta_2": "Delta 2", "gamma_11": "Gamma 11",
          "gamma_22": "Gamma 22", "gamma_12": "Gamma 12", "vega_1": "Vega 1", "vega_2": "Vega 2",
          "theta": "Theta", "rho": "Rate rho", "correlation": "Correlation"}


def read(name):
    with (RESULTS/name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def g(value, digits=2):
    value = float(value)
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    return f"{value:.{digits}e}" if abs(value) < 1e-3 or abs(value) >= 1e4 else f"{value:.{digits+1}g}"


def main():
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    levels = json.loads(LEVELS.read_text(encoding="utf-8"))["levels"]
    summary = json.loads((RESULTS/"summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((RESULTS/"protocol.json").read_text(encoding="utf-8"))
    selected = summary["selected"]
    ladder = read("ladder_validation.csv")
    tests = read("test_metrics.csv")
    bench = read("monte_carlo_benchmark.csv")
    maturity = read("error_by_maturity.csv")
    cases = read("study_cases.csv")
    test_row = {r["panel"]: r for r in tests if r["level"] == selected}
    box = settings["test_box"]
    n_test = settings["sets"]["test"]["count"]
    n_short = settings["sets"]["short_maturity"]["count"]
    lines = [
        "# Two-stock neural basket pricer",
        "",
        "A neural pricer for European arithmetic basket calls and puts on two correlated stocks under "
        "risk-neutral GBM, returning the price and ten Greeks in one call. It was built as a ladder: "
        "each level adds one technique, every level is scored on the same validation set, and the "
        "pre-declared rule picks the final model. The test sets were generated and hashed before any "
        f"training (protocol frozen {protocol['frozen_at'][:16].replace('T', ' ')} UTC) and played no "
        "part in any choice.",
        "",
        f"**Selected pricer: `{selected}`.** Reference values come from the conditional quadrature in "
        "`src/neural/reference.py`, a batched twin of the repository's `quadrature_price` that agrees with "
        "it to about 1e-12 in price; its Greeks are exact derivatives of the quadrature rule.",
        "",
        "## Domain",
        "",
        f"Strike {box['strike']}, spots {box['spot'][0]}-{box['spot'][1]}, time to expiry "
        f"{box['tau'][0]}-{box['tau'][1]} years, rate {box['rate'][0]}-{box['rate'][1]}, dividend yields "
        f"{box['dividend'][0]}-{box['dividend'][1]}, volatilities {box['vol'][0]}-{box['vol'][1]}, correlation "
        f"{box['correlation'][0]} to {box['correlation'][1]}, first weight {box['weight_1'][0]}-{box['weight_1'][1]}. "
        "Prices are homogeneous in spots and strike, so any strike is covered by rescaling. Training "
        "contracts come from a slightly larger box with extra weight on short maturities. Puts are priced "
        "by exact put-call parity, so their absolute price and Gamma/Vega/correlation errors equal those of "
        "the matching call.",
        "",
        "## Accuracy of the selected pricer",
        "",
        f"RMSE and largest absolute error against the reference on {n_test:,} fresh test contracts drawn "
        f"uniformly from the domain (alternating calls and puts) and {n_short:,} short-maturity contracts "
        f"({settings['short_maturity_tau'][0]}-{settings['short_maturity_tau'][1]} years). Units: currency for "
        "price, per unit spot for Delta and Gamma, per unit of volatility, rate or correlation for Vega, rho "
        "and correlation sensitivity, per year for Theta.",
        "",
        "| Quantity | Test RMSE | Test max error | Short-maturity RMSE | Short-maturity max error |",
        "|---|---:|---:|---:|---:|",
    ]
    for q in ("price",)+GREEKS:
        lines.append(f"| {LABELS[q]} | {g(test_row['test'][f'{q}_rmse'])} | {g(test_row['test'][f'{q}_max_abs'])} "
                     f"| {g(test_row['short_maturity'][f'{q}_rmse'])} | {g(test_row['short_maturity'][f'{q}_max_abs'])} |")
    lines += [
        "",
        "## The ladder",
        "",
        "Validation score = mean over price and the ten Greeks of RMSE divided by the RMS of the reference "
        "(lower is better). Training times are wall-clock seconds on a 4-core CPU; seeds of one level ran "
        "in parallel on one thread each, and ensemble times add up their members.",
        "",
        "| Level | Technique added | Validation score | Price RMSE | Gamma 11 RMSE | Vega 1 RMSE | Theta RMSE | Correlation RMSE | Test score | Training s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ladder:
        test_score = next(r["score"] for r in tests if r["level"] == row["level"] and r["panel"] == "test")
        mark = " **(selected)**" if row["level"] == selected else ""
        lines.append(f"| `{row['level']}`{mark} | {row['technique']} | {g(row['score'])} | {g(row['price_rmse'])} "
                     f"| {g(row['gamma_11_rmse'])} | {g(row['vega_1_rmse'])} | {g(row['theta_rmse'])} "
                     f"| {g(row['correlation_rmse'])} | {g(test_score)} | {float(row['training_seconds']):.0f} |")
    lines += ["", "![Ladder](ladder.png)", ""]

    spec = settings["benchmark"]
    lines += [
        "## Benchmark against Monte Carlo",
        "",
        f"The first {spec['contracts']} test contracts, priced by the selected pricer, by plain Monte Carlo "
        "(pseudo-random exact GBM draws; pathwise first-order Greeks and likelihood-ratio/pathwise Gammas, "
        "`src/neural/monte_carlo.py`) and by the repository's Sobol RQMC estimators (conditional quadrature "
        f"with the geometric control, {spec['rqmc_replicates']} scrambles; bumped Vega, Theta, rho and "
        "correlation). Errors are RMSE against the reference. Seconds are wall-clock per contract on one "
        "machine and include all Greeks.",
        "",
        "| Method | Normal draws | Price | Delta 1 | Gamma 11 | Vega 1 | Theta | Correlation | Seconds per contract |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    methods = []
    for row in bench:
        if row["method"] not in methods:
            methods.append(row["method"])
    by = {(r["method"], r["quantity"]): r for r in bench}
    for method in methods:
        name = ("Neural (selected)" if method == "neural" else
                "Plain MC" if method.startswith("mc_") else "Sobol RQMC (repo)")
        draws = int(by[(method, "price")]["normals"])
        lines.append(f"| {name} | {draws:,} | " + " | ".join(
            g(by[(method, q)]["rmse"]) for q in ("price", "delta_1", "gamma_11", "vega_1", "theta", "correlation"))
            + f" | {float(by[(method, 'price')]['seconds_per_contract']):.2g} |")
    lines += ["", "Median absolute error (Sobol RQMC has a few heavy-tailed contracts at strongly negative "
              "correlation and high volatility, which dominate its RMSE):", "",
              "| Method | Normal draws | Price | Delta 1 | Gamma 11 | Vega 1 | Theta | Correlation |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        name = ("Neural (selected)" if method == "neural" else
                "Plain MC" if method.startswith("mc_") else "Sobol RQMC (repo)")
        lines.append(f"| {name} | {int(by[(method, 'price')]['normals']):,} | " + " | ".join(
            g(by[(method, q)]["median_abs"]) for q in ("price", "delta_1", "gamma_11", "vega_1", "theta", "correlation"))
            + " |")
    biggest = max((m for m in methods if m.startswith("mc_")), key=lambda m: int(by[(m, "price")]["normals"]))
    paths = int(by[(biggest, "price")]["normals"])
    plain = sorted((m for m in methods if m.startswith("mc_")), key=lambda m: int(by[(m, "price")]["normals"]))
    coverage = plain[-2:]
    lines += [
        "",
        f"Plain Monte Carlo paths needed to match the neural RMSE, extrapolated from {paths:,} paths with the "
        "1/sqrt(N) law:",
        "",
        "| Quantity | Neural RMSE | MC RMSE at " + f"{paths:,} paths | Equivalent MC paths |",
        "|---|---:|---:|---:|",
    ]
    for q in ("price",)+GREEKS:
        nn, mc = float(by[("neural", q)]["rmse"]), float(by[(biggest, q)]["rmse"])
        lines.append(f"| {LABELS[q]} | {g(nn)} | {g(mc)} | {paths*(mc/nn)**2:,.0f} |")
    timing = summary["neural_seconds_per_contract"]
    lines += [
        "",
        f"Neural evaluation, one thread: {timing['batch_price']*1e6:.1f} microseconds per contract for prices "
        f"and {timing['batch_greeks']*1e6:.1f} for price plus all Greeks in a batch of {n_test:,}; "
        f"{timing['single_greeks']*1e3:.2f} ms for a single contract with Greeks.",
        "",
        "![Monte Carlo benchmark](monte_carlo_benchmark.png)",
        "",
        "## Greek validation",
        "",
        f"Coverage: share of the {spec['contracts']} contracts where the value lies inside the 95% interval of "
        "plain Monte Carlo. The reference columns check the intervals themselves; a neural share near the "
        "reference share means the neural error is small next to the Monte Carlo error at that path count.",
        "",
        "| Quantity | " + " | ".join(f"Reference, {int(by[(m, 'price')]['normals']):,} paths | "
                                     f"Neural, {int(by[(m, 'price')]['normals']):,} paths" for m in coverage) + " |",
        "|---|" + "---:|"*(2*len(coverage)),
    ]
    for q in ("price",)+GREEKS:
        lines.append(f"| {LABELS[q]} | " + " | ".join(
            f"{float(by[(m, q)]['reference_inside_95ci']):.1%} | {float(by[(m, q)]['neural_inside_95ci']):.1%}"
            for m in coverage) + " |")
    gap = max(float(by[(biggest, q)]["reference_inside_95ci"])-float(by[(biggest, q)]["neural_inside_95ci"])
              for q in ("price",)+GREEKS)
    lines += ["", (
        f"Even at {paths:,} paths the selected pricer's Greeks fall inside the Monte Carlo intervals nearly as "
        f"often as the reference values (largest shortfall {gap:.1%}), so Monte Carlo at that precision cannot "
        "tell them apart from the truth; the RMSE tables above, measured against the exact reference, are the "
        "sharper test." if gap <= 0.05 else
        f"At {paths:,} paths the Monte Carlo interval is about as narrow as the neural error for some Greeks "
        f"(largest coverage shortfall {gap:.1%}), so a neural Greek can fall outside it while still being close "
        "to the truth; the RMSE tables above, measured against the exact reference, are the sharper test there.")]
    checks = summary["identity_checks"]
    lines += [
        "",
        "Identity and no-arbitrage checks on the selected pricer's Greeks:",
        "",
        "| Panel | Contracts | Homogeneity max error | PDE residual RMSE / Theta RMS | Delta outside bounds (worst excess) | Gamma not PSD (worst negative / largest eigenvalue) | Price below intrinsic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for panel, c in checks.items():
        lines.append(f"| {panel.replace('_', ' ')} | {c['contracts']:,} | {g(c['homogeneity_max_abs'])} "
                     f"| {g(c['pde_residual_over_theta_rms'])} | {c['delta_bound_violations']} "
                     f"({g(c['delta_bound_worst_excess'])}) | {c['gamma_not_psd']} "
                     f"({g(c['gamma_worst_negative_ratio'])}) | {c['price_below_intrinsic']} |")
    lines += [
        "",
        "Homogeneity (V = S1 Delta1 + S2 Delta2 + K dV/dK) and put-call parity hold by construction. The "
        "two-asset Black-Scholes PDE is not imposed, so its residual measures how consistently Theta, the "
        "Deltas and the Gamma matrix were learned. Delta bounds are 0 <= Delta_i <= w_i exp(-q_i tau) for calls "
        "(mirrored for puts); the Gamma matrix must be positive semi-definite because the payoff is convex. "
        "The basket payoff depends on the spots only through w1 S1 + w2 S2, so the true Gamma matrix is "
        "close to rank one and its smaller eigenvalue is tiny; a small learned error can push that "
        "eigenvalue below zero, and the bracket gives its worst size relative to the larger eigenvalue.",
        "",
        "Error by time to expiry (test and short-maturity panels together), relative to the RMS of each reference quantity:",
        "",
        "| Years to expiry | Contracts | Price | Delta 1 | Gamma 11 | Vega 1 | Theta | Correlation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in maturity:
        lines.append(f"| {float(row['tau_low']):g}-{min(float(row['tau_high']), 2):g} | {row['contracts']} | " + " | ".join(
            g(float(row[f'{q}_rmse'])/float(row[f'{q}_reference_rms']))
            for q in ("price", "delta_1", "gamma_11", "vega_1", "theta", "correlation")) + " |")
    lines += ["", "![Error by maturity](error_by_maturity.png)", "",
              "The eight contracts of the classical two-stock study (observed earlier, so illustrative only):", "",
              "| Case | Price ref | Price neural | Delta 1 ref | Delta 1 neural | Gamma 12 ref | Gamma 12 neural | Theta ref | Theta neural |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    table = {}
    for row in cases:
        table.setdefault(row["case"], {})[row["quantity"]] = row
    for case, rows in table.items():
        lines.append(f"| {case} | " + " | ".join(
            f"{float(rows[q]['reference']):.5g} | {float(rows[q]['neural']):.5g}"
            for q in ("price", "delta_1", "gamma_12", "theta")) + " |")
    lines += [
        "",
        "## Limits",
        "",
        "- The pricer is trained for the domain above. Outside it (for example correlation beyond +/-0.95 or "
        "expiries under one week) it extrapolates without guarantees.",
        "- The model is the repository's: constant rates, dividends, volatilities and correlation under GBM.",
        "- Accuracy is measured against the conditional quadrature reference, which is exact to about 1e-12 "
        "here; Monte Carlo is a competitor in the benchmark, not the reference.",
        "- Timings depend on the machine and on batching. Training and labelling time is a one-off cost that "
        "pays off only when many contracts are priced.",
        "",
        "## Reproduce",
        "",
        "Run from the `2d/` folder (Python 3.11, NumPy 2.4, SciPy 1.17, PyTorch 2.14 recorded in `summary.json`).",
        "",
        "```",
        "python -m experiments.neural.verify_results           # hashes + re-evaluation, no training",
        "python -m pytest tests -q",
        "python -m experiments.neural.prepare_data --overwrite  # rebuild data (test sets are frozen)",
        "for level in L0_levy L1_naive L2_scale_free L3_black_output L4_differential L5_gamma_labels L6_lbfgs; do",
        "  python -m experiments.neural.train_ladder $level --seed 0",
        "done",
        "for seed in 0 1 2 3; do python -m experiments.neural.train_ladder L7_scale_up --seed $seed; done",
        "python -m experiments.neural.train_ladder L8_ensemble  # averages the four L7 seeds",
        "python -m experiments.neural.evaluate",
        "python -m experiments.neural.make_report",
        "```",
        "",
    ]
    (RESULTS/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS/'report.md'}")


if __name__ == "__main__":
    main()
