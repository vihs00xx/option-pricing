"""Markdown text of the PINN/DeepONet study report, built from the saved outputs."""

import csv
import json
from pathlib import Path

import numpy as np

from experiments.neural.common import RESULTS
from experiments.records import ROOT
from src.neural.reference import QUANTITIES

LABEL = {"price": "Price", "delta_1": "Delta 1", "delta_2": "Delta 2", "gamma_11": "Gamma 11",
         "gamma_12": "Gamma 12", "gamma_22": "Gamma 22", "vega_1": "Vega 1", "vega_2": "Vega 2",
         "theta": "Theta", "rho": "Rate rho", "correlation_12": "Correlation"}
SHOWN = ("price", "delta_1", "gamma_11", "vega_1", "theta", "correlation_12")
SPLITS = ("interpolation", "short_maturity", "extrapolation", "anchor", "all")
SPLIT_LABEL = {"interpolation": "Interpolation", "short_maturity": "Short maturity",
               "extrapolation": "Extrapolation", "anchor": "Anchor", "all": "All cases"}
FAMILY = {"pinn": "PINN", "deeponet": "DeepONet"}


def g(value, digits=3):
    value = float(value)
    if value == 0:
        return "0"
    return f"{value:.{digits}g}"


def n(value):
    return f"{int(round(float(value))):,}"


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def table(header, rows, align=None):
    align = align or ["---"]+["---:"]*(len(header)-1)
    out = ["| "+" | ".join(header)+" |", "| "+" | ".join(align)+" |"]
    out += ["| "+" | ".join(str(c) for c in row)+" |" for row in rows]
    return "\n".join(out)


def ladder_section(settings, records, selection, family):
    chosen = selection["selected"][family]["level"]
    rows = []
    for level, record in records.items():
        if settings["levels"][level]["family"] != family:
            continue
        v = record["scores"]["validation"]
        name = f"`{level}`"+(" **(selected)**" if level == chosen else "")
        rows.append([name, settings["levels"][level]["technique"], n(record["parameters"]), g(v["score"]),
                     g(record["scores"]["short_validation"]["score"])]
                    + [g(v["rmse"][q]) for q in SHOWN]+[n(record["total_seconds"])])
    header = (["Level", "Technique added", "Parameters", "Validation score", "Short-maturity score"]
              + [f"{LABEL[q]} RMSE" for q in SHOWN]+["Training s"])
    return table(header, rows, ["---", "---"]+["---:"]*(len(header)-2))


def test_section(selection):
    metrics = read_csv(RESULTS/"test_metrics.csv")
    timing = json.loads((RESULTS/"test_timing.json").read_text(encoding="utf-8"))
    get = {(r["method"], r["split"], r["quantity"]): r for r in metrics}
    out = []
    for family in ("pinn", "deeponet"):
        method = f"{family}_selected"
        level = selection["selected"][family]["level"]
        out.append(f"**{FAMILY[family]} (`{level}`).** RMSE by split, with the largest absolute error and the "
                   "median relative error (test-set floors) over all 248 cases:\n")
        rows = []
        for q in QUANTITIES:
            rows.append([LABEL[q]]+[g(get[(method, s, q)]["rmse"]) for s in SPLITS[:-1]]
                        + [g(get[(method, "all", q)]["max_abs"]), g(get[(method, "all", q)]["median_relative"])])
        out.append(table(["Quantity"]+[SPLIT_LABEL[s] for s in SPLITS[:-1]]+["Largest error", "Median relative"],
                         rows))
        out.append("")
    scores = {}
    for method in sorted({r["method"] for r in metrics}):
        for split in SPLITS:
            scores[(method, split)] = [float(get[(method, split, q)]["rmse"]) for q in QUANTITIES]
    # Score per split = mean over quantities of RMSE / RMS(reference), as on the validation panels.
    references = read_csv(ROOT/"results/test_set_v1/references.csv")
    rms = {}
    for split in SPLITS:
        rows = [r for r in references if split == "all" or r["split"] == split]
        rms[split] = [np.sqrt(np.mean([float(r[q])**2 for r in rows])) for q in QUANTITIES]
    rows = []
    for method in ["pinn_selected", "deeponet_selected", "levy_approximation"]+sorted(
            m for m in {r["method"] for r in metrics} if m[:1] in ("P", "D")):
        rows.append([f"`{method}`"]+[g(np.mean(np.array(scores[(method, s)])/np.array(rms[s]))) for s in SPLITS]
                    + [g(timing["methods"][method]["batch_seconds_per_contract"]*1e6)])
    out.append("Test score (mean over price and the ten Greeks of RMSE / RMS of the reference) of every pricer, "
               "by split; the ladder levels are shown for context only, the selection was frozen before the "
               "test set was opened:\n")
    out.append(table(["Pricer"]+[SPLIT_LABEL[s] for s in SPLITS]+["Microseconds per contract"], rows))
    return "\n".join(out)


def benchmark_section(prefix, panel):
    rows = read_csv(RESULTS/f"{prefix}monte_carlo_benchmark.csv")
    coverage = read_csv(RESULTS/f"{prefix}greek_coverage.csv")
    identities = read_csv(RESULTS/f"{prefix}greek_identities.csv")
    throughput = json.loads((RESULTS/f"{prefix}neural_throughput.json").read_text(encoding="utf-8"))["models"]
    lead = " ".join(f"{FAMILY[family]}: {g(t['seconds_per_contract']*1e6)} microseconds per contract for the price "
                    f"and all ten Greeks in a batch of {n(t['batch'])}, {g(t['single_contract_seconds']*1e3)} ms "
                    "for a single contract." for family, t in throughput.items())
    methods = []
    for r in rows:
        key = (r["method"], int(r["draws"]))
        if key not in methods:
            methods.append(key)
    name = {"pinn_selected": "PINN (selected)", "deeponet_selected": "DeepONet (selected)",
            "plain_monte_carlo": "Plain Monte Carlo", "sobol_rqmc_engine": "Sobol RQMC engine"}
    get = {(r["method"], int(r["draws"]), r["quantity"]): r for r in rows}
    out = [f"Neural evaluation, one thread. {lead}\n"]
    for statistic, title in (("rmse", "RMSE"), ("median_abs", "Median absolute error")):
        body = []
        for method, draws in methods:
            body.append([name[method], n(draws) if draws else "0"]
                        + [g(get[(method, draws, q)][statistic]) for q in QUANTITIES]
                        + ([g(get[(method, draws, "price")]["seconds_per_contract"])] if statistic == "rmse" else []))
        header = ["Method", "Normal draws"]+[LABEL[q] for q in QUANTITIES]+(["Seconds per contract"]
                                                                            if statistic == "rmse" else [])
        out.append(f"{title} against the quadrature reference:\n")
        out.append(table(header, body))
        out.append("")
    if panel == "test":
        out.append(interpolation_table(rows, name))
        out.append("")
    reference_paths = max(int(r["draws"]) for r in rows if r["method"] == "plain_monte_carlo")
    body = []
    for q in QUANTITIES:
        mc = float(get[("plain_monte_carlo", reference_paths, q)]["rmse"])
        cells = [LABEL[q], g(mc)]
        for family in ("pinn", "deeponet"):
            neural = float(get[(f"{family}_selected", 0, q)]["rmse"])
            cells += [g(neural), n(reference_paths*(mc/neural)**2)]
        body.append(cells)
    out.append(f"Plain Monte Carlo paths needed to match each neural RMSE, extrapolated from {n(reference_paths)} "
               "paths with the 1/sqrt(N) law:\n")
    out.append(table(["Quantity", f"MC RMSE at {n(reference_paths)} paths", "PINN RMSE", "Equivalent paths (PINN)",
                      "DeepONet RMSE", "Equivalent paths (DeepONet)"], body))
    out.append("")
    body = []
    for paths in sorted({int(r["paths"]) for r in coverage}):
        for r in coverage:
            if int(r["paths"]) == paths:
                body.append([n(paths), LABEL[r["quantity"]], f"{100*float(r['reference']):.1f}%",
                             f"{100*float(r['pinn']):.1f}%", f"{100*float(r['deeponet']):.1f}%"])
    out.append("Greek validation against Monte Carlo: share of cases whose value lies inside the plain Monte "
               "Carlo 95% interval. The reference column checks the intervals themselves; a neural share near "
               "it means the neural error is small next to the Monte Carlo error at that path count.\n")
    out.append(table(["Paths", "Quantity", "Reference", "PINN", "DeepONet"], body))
    out.append("")
    body = [[r["panel"], r["model"], n(r["contracts"]), g(r["pde_residual_rmse_over_theta_rms"]),
             f"{r['delta_outside_bounds']} ({g(r['delta_worst_excess'])})",
             f"{r['gamma_not_psd']} ({g(r['gamma_worst_negative'])})",
             f"{r['price_below_intrinsic']} ({g(r['price_worst_below'])})", r["price_above_upper_bound"]]
            for r in identities]
    out.append("Identity and no-arbitrage checks of the Greeks. The PDE residual uses each pricer's own Theta, "
               "Deltas and Gammas in the two-asset Black-Scholes equation in spot coordinates. Delta bounds are "
               "0 <= Delta_i <= w_i exp(-q_i tau) for calls (mirrored for puts); the Gamma matrix of a convex "
               "payoff is positive semi-definite; prices must lie between the discounted intrinsic value and "
               "the discounted forward (calls) or strike (puts). Counts use the test set's floors (1e-4 for "
               "prices and Deltas, 1e-7 for Gammas); brackets give the worst size.\n")
    out.append(table(["Panel", "Pricer", "Contracts", "PDE residual RMSE / Theta RMS",
                      "Delta outside bounds (worst)", "Gamma not PSD (most negative eigenvalue)",
                      "Price below intrinsic (worst)", "Price above bound"], body))
    return "\n".join(out)


def interpolation_table(rows, name):
    """RMSE of the neural pricers and plain Monte Carlo on the 128 in-domain test cases only."""
    from experiments.neural.evaluate_test import load_test_set
    test = load_test_set()
    mask = test["split"] == "interpolation"
    values = np.load(RESULTS/"benchmark_values.npz")
    body = []
    for key, label, draws in [("pinn", "pinn_selected", 0), ("deeponet", "deeponet_selected", 0)]+[
            (f"mc{d}", "plain_monte_carlo", d) for d in sorted({int(r["draws"]) for r in rows
                                                                 if r["method"] == "plain_monte_carlo"})]:
        cells = [name[label], n(draws) if draws else "0"]
        for q in QUANTITIES:
            error = values[f"{key}_{q}"][mask]-test["reference"][q][mask]
            cells.append(g(np.sqrt(np.mean(error**2))))
        body.append(cells)
    return ("RMSE on the 128 interpolation cases only (inside the training domain), where the neural pricers "
            "are meant to be used:\n\n"+table(["Method", "Normal draws"]+[LABEL[q] for q in QUANTITIES], body))


def maturity_section(maturity):
    out = []
    for family in ("pinn", "deeponet"):
        body = [[f"{g(r['tau_low'])}-{g(r['tau_high'])}", n(r["contracts"])]+[g(r[q]) for q in SHOWN]
                for r in maturity if r["model"] == family]
        out.append(f"**{FAMILY[family]}**, RMSE / RMS of the reference by years to expiry (validation and "
                   "short-maturity panels; the training domain starts at 0.05):\n")
        out.append(table(["Years to expiry", "Contracts"]+[LABEL[q] for q in SHOWN], body))
        out.append("")
    return "\n".join(out)


def domain_table(settings):
    protocol = json.loads((ROOT/settings["protocol"]).read_text(encoding="utf-8"))
    rows = [[name.replace("_", " "), f"{g(low)} to {g(high)}"] for name, (low, high) in protocol["training_domain"].items()]
    return table(["Parameter", "Training domain"], rows, ["---", "---"])


def glance_section(settings, records, selection, panel, prefix):
    benchmark = read_csv(RESULTS/f"{prefix}monte_carlo_benchmark.csv")
    get = {(r["method"], int(r["draws"]), r["quantity"]): r for r in benchmark}
    rows = []
    for family in ("pinn", "deeponet"):
        level = selection["selected"][family]["level"]
        record = records[level]
        rows.append([f"{FAMILY[family]} `{level}`", g(record["scores"]["validation"]["score"]),
                     g(record["scores"]["short_validation"]["score"]),
                     g(get[(f"{family}_selected", 0, "price")]["rmse"]),
                     g(get[(f"{family}_selected", 0, "gamma_11")]["rmse"]),
                     g(get[(f"{family}_selected", 0, "theta")]["rmse"]),
                     g(float(get[(f"{family}_selected", 0, "price")]["seconds_per_contract"])*1e6),
                     n(chain_seconds(settings, records, level)), n(ladder_seconds(settings, records, family))])
    for method, label in (("plain_monte_carlo", "Plain Monte Carlo, {} paths"),
                          ("sobol_rqmc_engine", "Sobol RQMC engine, {} draws")):
        draws = max(int(r["draws"]) for r in benchmark if r["method"] == method)
        label = label.format(n(draws))
        rows.append([label, "", "", g(get[(method, draws, "price")]["rmse"]), g(get[(method, draws, "gamma_11")]["rmse"]),
                     g(get[(method, draws, "theta")]["rmse"]),
                     g(float(get[(method, draws, "price")]["seconds_per_contract"])*1e6), "0", "0"])
    where = "frozen test set (248 cases)" if panel == "test" else "development panel (256 fresh contracts)"
    return (f"Validation scores are on 4,096 fresh contracts of the training domain and 4,096 short-maturity "
            f"contracts; RMSEs and timings are on the {where}, price and all ten Greeks per contract, one CPU "
            "thread, in a batch of that panel's size. Training seconds are those of the selected pricer, "
            "including the levels it was warm-started from, and of the whole ladder, the full development "
            "cost; prices with Greeks in batches of 4,096 take the time per contract given in the "
            "benchmark section.\n\n"
            + table(["Pricer", "Validation score", "Short-maturity score", "Price RMSE", "Gamma 11 RMSE",
                     "Theta RMSE", "Microseconds per contract", "Training s (selected)", "Training s (ladder)"],
                    rows))


def chain_seconds(settings, records, level):
    """Training seconds of a level including the levels it was warm-started from."""
    seconds = records[level]["total_seconds"]
    start = settings["levels"][level].get("warm_start") or level_warm_start(settings, level)
    return seconds+(chain_seconds(settings, records, start) if start else 0.0)


def level_warm_start(settings, level):
    """The warm start a level inherits through its ``base`` chain (None for fresh runs)."""
    entry = settings["levels"][level]
    if "warm_start" in entry:
        return entry["warm_start"]
    return level_warm_start(settings, entry["base"]) if "base" in entry else None


def ladder_seconds(settings, records, family):
    return sum(r["total_seconds"] for level, r in records.items() if settings["levels"][level]["family"] == family)


METHOD = """\
## Problem

European arithmetic basket calls and puts on two correlated stocks under risk-neutral GBM, with strike
100, constant rate, dividend yields, volatilities and correlation. A contract has ten inputs: two
spots (as moneyness S_i/K), time to expiry, rate, two dividend yields, two volatilities, the
correlation and the first weight (the second is one minus it). The pricers return the price and ten
Greeks: both Deltas, the three spot Gammas, both Vegas, Theta (calendar time, per year), rate rho
and the correlation sensitivity, all per unit decimal change.

The training domain is the one of the frozen test-set protocol (`experiments/config/protocol/test_set_v1.json`):

{domain}

## Exact reduction shared by both families

The call is homogeneous of degree one in spots and strike, the weights only scale the spots, and
rates and dividends only enter through the forwards and the discount factor, so

    C = K exp(-r tau) c(z, s, tau; sigma_1, sigma_2, rho),
    f_i = w_i S_i exp((r - q_i) tau) / K,  z = log(f_1 + f_2),  s = f_1 / (f_1 + f_2),

and in y_i = log f_i the normalized call solves the driftless PDE

    c_tau = 1/2 sigma_1^2 (c_11 - c_1) + rho sigma_1 sigma_2 c_12 + 1/2 sigma_2^2 (c_22 - c_2),
    c(tau = 0) = (exp(z) - 1)^+.

Puts follow from exact put-call parity. Ten inputs become three coordinates and three parameters,
and every Greek is obtained by automatic differentiation of the currency price through this map
(`price_and_greeks` in `src/neural/reference.py`). The PDE is posed on z in [-3, 2.5] and s in
[0.01, 0.99]; the training domain maps into z in [-0.52, 0.46], so the far boundary sits well away
from it, as a grid solver's would. The volatility, correlation and maturity ranges are the
protocol's; the PDE starts at the payoff, so physics collocation covers maturities from 0 to 2
years, while every labelled contract lies inside the training domain.

## The two families

**PINN** (`src/neural/pinn.py`): one tanh MLP trained on the residual of the pricing PDE at
collocation points, with the payoff imposed exactly or softly and the far boundary softly. No
reference labels, except in the final anchored level.

**DeepONet** (`src/neural/deeponet.py`): a branch net on the contract's model parameters and a trunk
net on where the price is evaluated, combined by an inner product (Lu et al., 2021). After the
reduction the branch sees (sigma_1, sigma_2, rho) and the trunk (z, s, tau). It is trained on
reference labels from the batched conditional quadrature (`src/neural/reference.py`, the twin of
the adaptive quadrature in `src/basket`), optionally with their derivatives and the PDE residual.

Both families share the output constructions in `src/neural/networks.py`, the PDE residuals in
`src/neural/pde.py` and the trainer in `src/neural/training.py`: Adam with an exponentially decaying
rate, residual-based adaptive sampling, L-BFGS, and the Nystrom Newton-CG optimizer of the 1D study.

## Data, seeds and selection

| Set | Contracts | Seed | Use |
|---|---:|---:|---|
| DeepONet labels | 32,768 (131,072 for the scale-up) | 7001 | training |
| PINN anchors | 4,096 | 7002 | training of the anchored level only |
| Monitor | 512 | 7201 | checkpoint choice within a run |
| Validation | 4,096 | 7301 | comparing levels; selection |
| Short-maturity validation | 4,096 (tau 0.005-0.05) | 7302 | reported, not used for selection |
| Frozen test set | 248 | 20261010 | final assessment, once |

All but the test set are scrambled Sobol draws from the protocol's training domain, calls and puts
alternating, with references from the batched quadrature. {rule}
"""

LIMITS = """\
## Limits

- The model is the repository's: constant rates, dividends, volatilities and correlation under GBM,
  two stocks. The reduction is exact for this model only.
- Accuracy is measured against conditional quadrature, exact to about 1e-12 for prices here; Monte
  Carlo is a competitor, not the reference.
- Timings depend on the machine, the batch and the thread count. Training and labelling are one-off
  costs that pay off only when many contracts are priced.
- Outside the training domain the pricers extrapolate without guarantees; the test set's
  extrapolation split measures how far that goes.
"""

REPRODUCE = """\
## Reproduce

Run from the `2d/` folder (Windows, Python 3.13.3, NumPy 2.1.0, SciPy 1.15.3, PyTorch 2.8.0 CPU).

    python -m experiments.neural.verify_results     # hashes, then re-evaluates the frozen pricers
    python -m pytest tests -q

To retrain (hours on a laptop CPU; each level resumes if interrupted), in ladder order:

    python -m experiments.neural.train P0_vanilla    # likewise every P* and D* level
    python -m experiments.neural.train D8_envelope --seed 1    # and seeds 2 and 3
    python -m experiments.neural.ensemble D9_ensemble
    python -m experiments.neural.select_models       # applies the rule, freezes the selection
    python -m experiments.neural.evaluate_test       # scores the frozen pricers on results/test_set_v1
    python -m experiments.neural.benchmark_monte_carlo
    python -m experiments.neural.make_report
"""

FINDINGS = """\
## Findings

- **Both ladders improved on the textbook network by more than an order of magnitude** on the
  validation score: PINN 0.287 to 0.0130 (`P0` to `P10`), DeepONet 0.193 to 0.0061 (`D0` to `D9`).
- **The largest gains came from the structure of the problem, not from the optimizer.** The exact
  payoff through the Levy approximation, the boundary-layer coordinate and the enveloped correction
  each changed the score by a third to a factor of five. Among optimizers, L-BFGS was the one that
  mattered (-40% for the PINN); Newton-CG added little.
- **In the training domain both pricers beat plain Monte Carlo with 1,000,000 paths on the price**
  (test RMSE 0.015 for the PINN and 0.0088 for the DeepONet against 0.020) and land between
  100,000 and 1,000,000 paths on the Greeks. They return the price and all ten Greeks in 17 and
  128 microseconds per contract in batches of 4,096, against 0.36 s for that Monte Carlo.
- **The repository's Sobol RQMC engine stays far more accurate** (price RMSE 0.0012 at 65,536
  draws, 0.83 s per contract). The neural pricers win on speed only, and only once their training
  (hours here) is spread over many contracts.
- **Outside the training domain the errors grow mainly for correlation and volatility.** Rate,
  dividend, moneyness and maturity extrapolate well because the exact reduction maps them back into
  the trained coordinates.
- **The DeepONet beats the PINN at every comparable stage.** The best pure-physics PINN (`P9`,
  no reference values at all) scored 0.0173; the selected PINN is the anchored hybrid `P10`.
- **The ensemble improved validation by 12% but not the test set** (test score 0.0215 against 0.0208
  for the single `D8`); the selection was frozen before the test set was opened.
- **No-arbitrage:** after the enveloped correction, no test price falls below its intrinsic value and
  no Delta leaves its bounds; a few Gamma matrices (9 PINN and 15 DeepONet of 248 cases) have a
  slightly negative smaller eigenvalue, since the true matrix is nearly rank one.
- **Against Monte Carlo intervals** the neural Greeks sit inside the 95% interval for 65-89% of
  cases at 100,000 paths and 38-81% at 1,000,000 paths, against 92-96% for the reference: Monte
  Carlo at those path counts can detect the neural errors, which the exact-reference RMSEs quantify.
"""
PINN_NOTES = """\
What each step showed:

- **Reduction alone (P1) did not help**: with a soft payoff the network must learn the kink at expiry
  over a wide spatial domain, and it scored no better than the textbook PINN.
- **The exact payoff (P2) was the decisive step** (0.30 to 0.060): the Levy approximation carries the
  payoff and most of the price, and the network only learns the correction.
- **Loss balancing (P3) and adaptive sampling (P4) moved little.** The residual stayed near the
  Levy approximation's own residual: that residual does not vanish near expiry but settles as a
  function of d = z / (total volatility), a structure an MLP in z cannot follow as tau -> 0.
- **The boundary-layer coordinate (P5)** gave the network d directly and cut the score by a third.
- **L-BFGS (P6) cut it by another 40%**; Nystrom Newton-CG (P7) added 4% at a quarter of the cost.
- **Width 128 (P8)** trained from scratch did worse than P7 within the same budget: a wider net
  needs more steps than the scale-up allowed.
- **The enveloped correction (P9)** scales the network by the Levy call's sensitivity to total
  volatility, so it fades like the time value away from the strike; it beat P7 by 20% and halved
  the short-maturity error.
- **Anchors (P10)**, reference prices at 4,096 contracts of their own seed added to the PDE loss as
  in the 1D study's anchored PINN, cut the score by another quarter. P10 is a hybrid: the only PINN
  level that sees reference values.
"""

DEEPONET_NOTES = """\
What each step showed:

- **Reduction (D1)** helped modestly; **the exact payoff (D2)** was again the largest single step.
- **Derivative labels (D3)** halved the score: the Greeks are what the score measures, and the
  labels supervise them directly.
- **The PDE residual (D4)** added 2%; the labels already pin the solution.
- **The boundary-layer coordinate in the trunk (D5)** helped most at short maturities (-40%).
- **L-BFGS (D6)** cut the score by 15%; **width and rank 128 with four times the data (D7)** matched
  it from scratch.
- **The enveloped correction (D8)** cut the score by 40% and the short-maturity score by two thirds.
- **The four-seed ensemble (D9)** averages independent trainings of D8.
"""


def report(settings, records, selection, maturity, panel, prefix):
    out = ["# Neural pricers for two-stock basket options: PINN and DeepONet", ""]
    pinn, deeponet = selection["selected"]["pinn"]["level"], selection["selected"]["deeponet"]["level"]
    out.append(f"Two neural pricers for European arithmetic basket options on two correlated stocks, each "
               f"built as a ladder in which every level adds one technique and is scored on the same "
               f"validation contracts. The pre-declared rule selected **`{pinn}`** for the PINN and "
               f"**`{deeponet}`** for the DeepONet; both were frozen by hash "
               f"({selection['frozen_at'][:16].replace('T', ' ')} UTC) before being scored once on the "
               "frozen two-stock test set, then benchmarked against Monte Carlo with their Greeks validated.")
    out.append("")
    out.append("## Results at a glance\n")
    out.append(glance_section(settings, records, selection, panel, prefix))
    out.append("")
    out.append(FINDINGS)
    out.append("")
    out.append(METHOD.format(domain=domain_table(settings), rule=settings["selection_rule"]))
    out.append("## PINN ladder\n")
    out.append(ladder_section(settings, records, selection, "pinn"))
    out.append("")
    out.append(PINN_NOTES)
    out.append("")
    out.append("## DeepONet ladder\n")
    out.append(ladder_section(settings, records, selection, "deeponet"))
    out.append("")
    out.append(DEEPONET_NOTES)
    out.append("")
    out.append("![Ladders](ladder.png)\n")
    if panel == "test":
        out.append("## Frozen test set\n")
        checks = json.loads((RESULTS/"test_set_checks.json").read_text(encoding="utf-8"))
        nearest = min(checks["smallest_scaled_distance_to_development"].values())
        worst = max(checks["batched_reference_worst_relative_difference"].values())
        out.append(f"`results/test_set_v1` (248 cases: 128 interpolation, 32 short-maturity, 72 extrapolation, "
                   f"16 anchor) was verified against its seal before scoring. Every development contract differs "
                   f"from every test case by at least {g(nearest)} of some parameter's range, and the batched "
                   f"quadrature that labelled the training data reproduces the sealed references to a worst "
                   f"relative difference of {g(worst)} (in the bumped sensitivities; prices, Deltas and Gammas "
                   "agree to about 1e-12).\n")
        out.append(test_section(selection))
        out.append("")
    out.append("## Benchmark against Monte Carlo\n")
    out.append("Plain Monte Carlo uses pseudo-random exact GBM draws with pathwise first-order Greeks and "
               "likelihood-ratio/pathwise Gammas (`src/neural/monte_carlo.py`); the RQMC engine is "
               "`rqmc_estimate` in `src/pricing` (conditional estimator, geometric control, 16 scrambles, "
               "bumped parameter sensitivities). Seconds per contract include all Greeks, one thread.\n")
    out.append(benchmark_section(prefix, panel))
    out.append("")
    out.append("![Monte Carlo benchmark](monte_carlo_benchmark.png)\n")
    out.append("## Greeks by maturity\n")
    out.append(maturity_section(maturity))
    out.append("![Error by maturity](error_by_maturity.png)\n")
    out.append(LIMITS)
    out.append(REPRODUCE)
    return "\n".join(out)
