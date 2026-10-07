"""Effect of truncating the variance domain at v_hi, on development contracts.

The networks' computational box stops at v_hi (not the ADI v_max = 15.28). This measures,
with the ADI solver at a fixed fine grid, how much a far-v condition at v_hi moves the
price and the grid Greeks at v0 <= 0.49 against the Lewis reference, next to the same
solver with v_max = 15.28. Two label-free conditions: V_v = 0 ("zero_slope"), and V equal
to the Black-Scholes price at the expected average variance from v_hi ("base_value", the
networks' base).

Run from heston/: python -m experiments.neural.variance_cap_study [--workers 12]
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.records import ROOT, write_csv, write_json
from src.heston.model import Contracts, development_contracts
from src.heston.reference import reference

OUT = ROOT / "results" / "neural" / "variance_cap_study"
CASES = [("zero_slope", v) for v in (0.75, 1.0, 1.5, 2.0, 3.0, 15.28)] + \
        [("base_value", v) for v in (0.75, 1.0, 1.5, 2.0, 3.0)]
NAMES = ("price", "delta", "gamma", "vega", "vanna", "volga", "theta_cal")


def job(args):
    record, v_hi, far_v = args
    from src.heston.adi import GridSize, adi_quantities
    c = Contracts.from_records([record])
    out = adi_quantities(c, GridSize(160, 100, 80), v_hi, parameter_greeks=False, far_v=far_v)
    return {q: float(out[q][0]) for q in NAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    main_c, short_c, _ = development_contracts()
    c = Contracts.concat(main_c.take(np.arange(0, 4096, 64)), short_c.take(np.arange(0, 512, 64)))
    # the v0-extrapolation values of the test set (0.42, 0.49) on a few copies
    extra = main_c.take(np.arange(1, 4096, 512)).with_values(v0=np.tile([0.42, 0.49], 4))
    c = Contracts.concat(c, extra)
    ref, _ = reference(c)
    rows, summary = [], {}
    records = c.records()
    with ProcessPoolExecutor(args.workers) as pool:
        for far_v, v_hi in CASES:
            vals = list(pool.map(job, [(r, v_hi, far_v) for r in records]))
            err = {q: np.array([v[q] for v in vals]) - ref[q] for q in NAMES}
            key = f"{far_v}_{v_hi}"
            summary[key] = {q: dict(max=float(np.max(np.abs(e))), rmse=float(np.sqrt(np.mean(e**2))))
                            for q, e in err.items()}
            for i in range(len(c)):
                rows.append(dict(far_v=far_v, v_hi=v_hi, contract=i, v0=c.v0[i], tau=c.tau[i], xi=c.xi[i],
                                 **{f"err_{q}": err[q][i] for q in NAMES}))
            print(key, {q: f"{summary[key][q]['max']:.1e}" for q in NAMES}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "errors.csv", rows)
    write_json(OUT / "summary.json", dict(grid="ADI 160 x 100 x 80 (S x v x time steps)", contracts=len(c),
                                          note="max |ADI - reference| over the contracts; the 15.28 row is the "
                                               "solver's own error with the full variance domain",
                                          summary=summary))


if __name__ == "__main__":
    main()
