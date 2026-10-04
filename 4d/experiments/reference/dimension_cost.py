"""Cost of the tensor Gauss-Hermite reference against the number of stocks.

Every configuration runs alone in a fresh process, one at a time, so its time
and peak memory are not shared with other work. A configuration whose memory,
extrapolated from the previous level of the same series, would not fit in
the memory free at that moment is skipped and recorded as such.
"""

from . import common  # noqa: F401  (pins BLAS threads before NumPy)

import argparse
from multiprocessing import Pool
from time import perf_counter

import numpy as np

from src.pricing import Basket, gauss_hermite_values

from .common import OUTPUT, append_jsonl, peak_memory_gb, read_jsonl, write_csv
from .dev_contracts import C_STAR

RAW = OUTPUT / "dimension_cost.jsonl"
HARD_SEED = 404201
MODERATE_SEED = 404202
EXTRA_STOCKS = [dict(spot=100, volatility=0.5, dividend=0.02), dict(spot=98, volatility=0.35, dividend=0.01)]
PLAN = ([(d, n, greeks) for d in (3, 4, 5) for greeks in (True, False) for n in (16, 24, 32, 40, 48)]
        + [(6, n, False) for n in (16, 24)] + [(6, 16, True)])


def moderate(d):
    rng = np.random.default_rng(MODERATE_SEED)
    while True:
        R = np.eye(d)
        upper = np.triu_indices(d, 1)
        R[upper] = rng.uniform(0.2, 0.5, len(upper[0]))
        R = R + np.triu(R, 1).T
        if np.linalg.eigvalsh(R)[0] >= 0.05:
            break
    return Basket(spots=(100,) * d, strike=100, tau=1, rate=0.03, volatilities=np.linspace(0.2, 0.4, d),
                  dividends=(0.01,) * d, weights=(1 / d,) * d, correlation=R)


def hard(d):
    """C* at tau 2, cut to three stocks or extended by random but valid correlations."""
    R = np.array(C_STAR["correlation"], dtype=float)
    spots, vols, divs = (list(C_STAR[k]) for k in ("spots", "volatilities", "dividends"))
    weights = np.array(C_STAR["weights"])
    rng = np.random.default_rng(HARD_SEED)
    for extra in EXTRA_STOCKS[:max(0, d - 4)]:
        while True:
            row = rng.uniform(-0.5, 0.9, len(R))
            grown = np.block([[R, row[:, None]], [row[None, :], np.ones((1, 1))]])
            if np.linalg.eigvalsh(grown)[0] >= 0.05:
                break
        R = grown
        spots.append(extra["spot"]), vols.append(extra["volatility"]), divs.append(extra["dividend"])
        weights = np.append(0.9 * weights, 0.1)
    R, weights = R[:d, :d], weights[:d]
    return Basket(spots=spots[:d], strike=100, tau=2, rate=C_STAR["rate"], volatilities=vols[:d],
                  dividends=divs[:d], weights=weights / weights.sum(), correlation=R)


def measure(task):
    family, d, nodes, greeks = task
    c = (moderate if family == "moderate" else hard)(d)
    start = perf_counter()
    values = gauss_hermite_values(c, nodes, greeks=greeks)
    return dict(family=family, d=d, nodes=nodes, greeks=greeks, grid_points=nodes ** (d - 1),
                seconds=perf_counter() - start, peak_gb=peak_memory_gb(), values=np.atleast_1d(values).tolist())


def available_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-margin", type=float, default=0.8, help="usable share of free memory")
    args = parser.parse_args()
    done = {(r["family"], r["d"], r["nodes"], r["greeks"]): r for r in read_jsonl(RAW)}
    for family in ("moderate", "hard"):
        for d, nodes, greeks in PLAN:
            key = (family, d, nodes, greeks)
            if key in done:
                continue
            previous = [r for r in done.values() if r["family"] == family and r["d"] == d
                        and r["greeks"] == greeks and "peak_gb" in r and r["nodes"] < nodes]
            if previous:
                last = max(previous, key=lambda r: r["nodes"])
                need = last["peak_gb"] * nodes ** (d - 1) / last["grid_points"]
                if need > args.memory_margin * available_gb():
                    row = dict(family=family, d=d, nodes=nodes, greeks=greeks, grid_points=nodes ** (d - 1),
                               skipped=f"needs about {need:.1f} GB, {available_gb():.1f} GB free")
                    append_jsonl(RAW, row)
                    done[key] = row
                    print("skip", key, row["skipped"], flush=True)
                    continue
            with Pool(1) as pool:
                row = pool.apply(measure, ((family, d, nodes, greeks),))
            append_jsonl(RAW, row)
            done[key] = row
            print(f"{family} d={d} n={nodes} greeks={greeks}: {row['seconds']:.1f}s {row['peak_gb']:.2f} GB",
                  flush=True)
    rows = []
    for (family, d, nodes, greeks), r in sorted(done.items()):
        lower = [x for x in done.values() if x["family"] == family and x["d"] == d and x["greeks"] == greeks
                 and x["nodes"] < nodes and "values" in x]
        change = ""
        if "values" in r and lower:
            change = float(np.max(np.abs(np.array(r["values"]) - np.array(max(lower, key=lambda x: x["nodes"])["values"]))))
        rows.append(dict(family=family, d=d, nodes=nodes, greeks=greeks, grid_points=r["grid_points"],
                         quantities=len(r.get("values", [])) or "", seconds=r.get("seconds", ""),
                         peak_gb=r.get("peak_gb", ""), max_change_vs_previous=change, skipped=r.get("skipped", "")))
    write_csv(OUTPUT / "dimension_cost.csv", rows)


if __name__ == "__main__":
    main()
