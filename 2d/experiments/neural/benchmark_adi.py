"""Score the two-stock ADI grid solver on the frozen test set, next to the neural pricers.

ADI (``src.pricing.adi_values``: modified Craig-Sneyd, Richardson extrapolation over the
grid and its nested doubling, PDE Theta, central bumps for Vega, rho and correlation)
needs no training, so its settings are fixed here in advance from the engine-validation
study (``results/engine_validation_v1``) and not tuned on the test cases.

Usage: python -m experiments.neural.benchmark_adi [--workers 12]
"""

import argparse
from multiprocessing import Pool
import os
from time import perf_counter

import numpy as np

from experiments.neural.benchmark_monte_carlo import basket, error_rows, identity_checks
from experiments.neural.common import RESULTS, keep_full_speed
from experiments.neural.evaluate_test import SPLITS, load_test_set, metrics
from experiments.records import write_csv
from src.pricing import adi_values
from src.neural.reference import QUANTITIES

INTERVALS = (40, 80, 160)       # each with Richardson extrapolation and parameter Greeks


def start_worker():
    keep_full_speed()


def price_case(job):
    row, put, intervals = job
    start = perf_counter()
    values = adi_values(basket(row, put), intervals=intervals, richardson=True, parameter_greeks=True)
    return [values[q] for q in QUANTITIES], perf_counter()-start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    test = load_test_set()
    x, put, reference = test["x"], test["put"], test["reference"]
    rows, split_rows, identity_rows, saved = [], [], [], {}
    # One BLAS thread per worker process: the grid solves are small and parallelize across cases.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    with Pool(args.workers, initializer=start_worker) as pool:
        for intervals in INTERVALS:
            results = []
            for k, result in enumerate(pool.imap(price_case, [(x[i], bool(put[i]), intervals) for i in range(len(x))]), 1):
                results.append(result)
                if k % 62 == 0:
                    print(f"ADI {intervals} intervals: {k}/{len(x)} cases", flush=True)
            values = {q: np.array([r[0][k] for r in results]) for k, q in enumerate(QUANTITIES)}
            seconds = float(np.mean([r[1] for r in results]))
            rows += error_rows("adi_richardson", intervals, values, reference, seconds, "test")
            for split in SPLITS+("all",):
                mask = np.ones(len(x), bool) if split == "all" else test["split"] == split
                for q, row in metrics(values, reference, mask).items():
                    split_rows.append(dict(method="adi_richardson", intervals=intervals, split=split,
                                           cases=int(mask.sum()), quantity=q, **row))
            identity_rows.append(identity_checks(f"adi_{intervals}", values, x, put, "test"))
            saved.update({f"adi{intervals}_{q}": values[q] for q in QUANTITIES})
            print(f"ADI {intervals} intervals: {seconds:.3g} s per contract, price RMSE "
                  f"{rows[-len(QUANTITIES)]['rmse']:.3g}", flush=True)
    write_csv(RESULTS/"adi_benchmark.csv", rows)
    write_csv(RESULTS/"adi_test_metrics.csv", split_rows)
    write_csv(RESULTS/"adi_identities.csv", identity_rows)
    np.savez_compressed(RESULTS/"adi_values.npz", **saved)


if __name__ == "__main__":
    main()
