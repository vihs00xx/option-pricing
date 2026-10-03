"""Benchmark the selected PINN and DeepONet against Monte Carlo, and validate their Greeks.

Cases: the frozen test set (default, after ``evaluate_test``) or a fresh development
panel (``--cases development``). Competitors, each pricing every case with all ten
Greeks:

* plain Monte Carlo (pseudo-random exact GBM draws; pathwise first-order Greeks and
  likelihood-ratio/pathwise Gammas) at 1,000 to 1,000,000 paths;
* the repository's replicated Sobol RQMC engine (``src.pricing.rqmc_estimate``:
  conditional estimator, geometric control variate, 16 scrambles, bumped parameter
  sensitivities) at 4,096 and 65,536 draws.

Errors are measured against the quadrature reference. Greek validation adds the share
of cases whose value lies inside the Monte Carlo 95% intervals (for the reference and
for each neural pricer) and identity and no-arbitrage checks of the neural Greeks on
the cases and on larger validation panels.

Usage: python -m experiments.neural.benchmark_monte_carlo [--cases test|development]
"""

import argparse
import json
from time import perf_counter

import numpy as np
import torch

from experiments.neural.common import RESULTS, evaluation_set, keep_full_speed, load_settings
from experiments.neural.evaluate_test import load_test_set, verify_selection
from experiments.records import ROOT, write_csv
from src.neural.coordinates import COLUMN, FIELDS, protocol_domain, sample_domain
from src.neural.networks import load_model
from src.neural.monte_carlo import monte_carlo_estimates
from src.neural.reference import GREEKS, QUANTITIES, price_and_greeks, reference_values
from src.pricing import Basket, rqmc_estimate

MC_PATHS = (1_000, 10_000, 100_000, 1_000_000)
RQMC_POINTS = (256, 4096)         # per replicate; 16 replicates
COVERAGE_PATHS = (100_000, 1_000_000)


def development_cases(settings):
    domain, strike = protocol_domain(ROOT/settings["protocol"])
    x, put = sample_domain(domain, strike, 256, seed=settings["seeds"]["validation"]+200)
    return dict(x=x, put=put, split=np.array(["development"]*len(x)), case_id=[f"dev_{i:03d}" for i in range(len(x))],
                reference=reference_values(x, put))


def basket(row, put):
    p = dict(zip(FIELDS, row))
    return Basket(spots=(p["spot_1"], p["spot_2"]), strike=p["strike"], tau=p["tau"], rate=p["rate"],
                  volatilities=(p["vol_1"], p["vol_2"]), dividends=(p["dividend_1"], p["dividend_2"]),
                  weights=(p["weight_1"], 1-p["weight_1"]), correlation=p["correlation"],
                  option="put" if put else "call")


def error_rows(method, draws, values, reference, seconds, panel):
    rows = []
    for q in QUANTITIES:
        error = values[q]-reference[q]
        rows.append(dict(panel=panel, method=method, draws=draws, quantity=q,
                         rmse=float(np.sqrt(np.mean(error**2))), median_abs=float(np.median(np.abs(error))),
                         max_abs=float(np.max(np.abs(error))), seconds_per_contract=seconds))
    return rows


def identity_checks(model_name, values, x, put, panel):
    """PDE residual, Delta bounds, Gamma convexity and price bounds of one pricer's outputs.

    Violations are counted beyond the test set's floors (1e-4 for prices and Deltas,
    1e-7 for Gammas), so round-off-sized departures of deep in- or out-of-the-money
    contracts are not counted; the worst sizes are reported as well.
    """
    c = {name: x[:, i] for i, name in enumerate(FIELDS)}
    s1, s2, v1, v2, rho = c["spot_1"], c["spot_2"], c["vol_1"], c["vol_2"], c["correlation"]
    generator = (0.5*v1*v1*s1*s1*values["gamma_11"]+rho*v1*v2*s1*s2*values["gamma_12"]
                 + 0.5*v2*v2*s2*s2*values["gamma_22"]+(c["rate"]-c["dividend_1"])*s1*values["delta_1"]
                 + (c["rate"]-c["dividend_2"])*s2*values["delta_2"]-c["rate"]*values["price"])
    residual = -values["theta"]-generator
    w = np.stack((c["weight_1"], 1-c["weight_1"]))
    cap = w*np.exp(-np.stack((c["dividend_1"], c["dividend_2"]))*c["tau"])
    delta = np.stack((values["delta_1"], values["delta_2"]))
    low, high = np.where(put, -cap, 0), np.where(put, 0, cap)
    delta_excess = np.maximum(np.maximum(low-delta, delta-high), 0).max(0)
    hessian = np.stack((np.stack((values["gamma_11"], values["gamma_12"]), -1),
                        np.stack((values["gamma_12"], values["gamma_22"]), -1)), -2)
    smallest = np.linalg.eigvalsh(hessian)[:, 0]
    forward = (cap*np.stack((s1, s2))).sum(0)
    discounted_strike = c["strike"]*np.exp(-c["rate"]*c["tau"])
    intrinsic = np.maximum(np.where(put, discounted_strike-forward, forward-discounted_strike), 0)
    upper = np.where(put, discounted_strike, forward)
    below = np.maximum(intrinsic-values["price"], 0)
    above = np.maximum(values["price"]-upper, 0)
    return dict(panel=panel, model=model_name, contracts=len(x),
                pde_residual_rmse_over_theta_rms=float(np.sqrt(np.mean(residual**2))/np.sqrt(np.mean(values["theta"]**2))),
                delta_outside_bounds=int((delta_excess > 1e-4).sum()), delta_worst_excess=float(delta_excess.max()),
                gamma_not_psd=int((smallest < -1e-7).sum()), gamma_worst_negative=float(max(-smallest.min(), 0)),
                price_below_intrinsic=int((below > 1e-4).sum()), price_worst_below=float(below.max()),
                price_above_upper_bound=int((above > 1e-4).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", choices=("test", "development"), default="test")
    args = parser.parse_args()
    keep_full_speed()
    torch.set_num_threads(1)
    settings = load_settings()
    selection = verify_selection()
    if args.cases == "test":
        if not (RESULTS/"test_metrics.csv").exists():
            raise RuntimeError("Run evaluate_test first: the benchmark reuses the frozen test cases")
        cases = load_test_set()
    else:
        cases = development_cases(settings)
    panel = args.cases
    x, put, reference = cases["x"], cases["put"], cases["reference"]
    n = len(x)
    rows, coverage_rows, identity_rows, outputs = [], [], [], {}
    models = {family: load_model(ROOT/entry["checkpoint"]) for family, entry in selection["selected"].items()}
    for family, model in models.items():
        price_and_greeks(model.price, x[:2], put[:2])          # warm-up
        start = perf_counter()
        values = price_and_greeks(model.price, x, put)
        rows += error_rows(f"{family}_selected", 0, values, reference, (perf_counter()-start)/n, panel)
        outputs[family] = values
        identity_rows.append(identity_checks(family, values, x, put, panel))
    identity_rows.append(identity_checks("reference", reference, x, put, panel))
    monte_carlo = {}
    for paths in MC_PATHS:
        start = perf_counter()
        estimates = [monte_carlo_estimates(x[i], bool(put[i]), paths, seed=settings["seeds"]["validation"]+i)
                     for i in range(n)]
        seconds = (perf_counter()-start)/n
        mean = {q: np.array([e[q][0] for e in estimates]) for q in QUANTITIES}
        se = {q: np.array([e[q][1] for e in estimates]) for q in QUANTITIES}
        monte_carlo[paths] = (mean, se)
        rows += error_rows("plain_monte_carlo", paths, mean, reference, seconds, panel)
        print(f"plain MC {paths}: {seconds:.3g} s per contract", flush=True)
    for points in RQMC_POINTS:
        start = perf_counter()
        estimates = [rqmc_estimate(basket(x[i], put[i]), n_points=points, n_replicates=16,
                                   seed=settings["seeds"]["validation"]+i, control=True) for i in range(n)]
        seconds = (perf_counter()-start)/n
        mean = {q: np.array([e.values[q] for e in estimates]) for q in QUANTITIES}
        rows += error_rows("sobol_rqmc_engine", points*16, mean, reference, seconds, panel)
        print(f"RQMC {points}x16: {seconds:.3g} s per contract", flush=True)
    for paths in COVERAGE_PATHS:
        mean, se = monte_carlo[paths]
        for q in QUANTITIES:
            inside = {name: float(np.mean(np.abs(values[q]-mean[q]) <= 1.96*se[q]))
                      for name, values in dict(reference=reference, **outputs).items()}
            coverage_rows.append(dict(panel=panel, paths=paths, quantity=q, **inside))
    throughput = {}
    for kind in ("validation", "short_validation"):
        data = evaluation_set(kind)
        vx, vput = data["inputs"]
        for family, model in models.items():
            start = perf_counter()
            values = price_and_greeks(model.price, vx, vput, chunk=len(vx))
            batch_seconds = perf_counter()-start
            if kind == "validation":
                single = []
                for i in range(32):
                    start = perf_counter()
                    price_and_greeks(model.price, vx[i:i+1], vput[i:i+1])
                    single.append(perf_counter()-start)
                throughput[family] = dict(batch=len(vx), seconds_per_contract=batch_seconds/len(vx),
                                          single_contract_seconds=float(np.median(single)))
            identity_rows.append(identity_checks(family, values, vx, vput, kind))
    prefix = "" if panel == "test" else "development_"
    (RESULTS/f"{prefix}neural_throughput.json").write_text(json.dumps(dict(threads=1, models=throughput), indent=2)+"\n",
                                                           encoding="utf-8")
    write_csv(RESULTS/f"{prefix}monte_carlo_benchmark.csv", rows)
    write_csv(RESULTS/f"{prefix}greek_coverage.csv", coverage_rows)
    write_csv(RESULTS/f"{prefix}greek_identities.csv", identity_rows)
    np.savez_compressed(RESULTS/f"{prefix}benchmark_values.npz", **{
        f"{family}_{q}": values[q] for family, values in outputs.items() for q in QUANTITIES},
        **{f"mc{paths}_{q}": monte_carlo[paths][0][q] for paths in MC_PATHS for q in QUANTITIES},
        **{f"mc{paths}_se_{q}": monte_carlo[paths][1][q] for paths in MC_PATHS for q in QUANTITIES})


if __name__ == "__main__":
    main()
