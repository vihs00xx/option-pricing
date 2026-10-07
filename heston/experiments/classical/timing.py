"""Quiet single-thread timings of every classical setting on a fixed subset of test cases.

The benchmark's own seconds were measured in worker processes that shared the CPU with
the neural training; these are the timings the report uses. Run with nothing else busy.

Usage (from heston/): python -m experiments.classical.timing [--cases 12]
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
from time import perf_counter

import numpy as np
import torch

from experiments.classical.benchmark import SETTINGS, V_MAX, setting_name
from experiments.records import ROOT, write_json
from experiments.test_set.load import load_test_set
from src.heston.adi import GridSize, adi_quantities
from src.heston.cos import cos_quantities
from src.heston.monte_carlo import mc_quantities, rqmc_quantities
from src.heston.reference import reference

OUT = ROOT / "results" / "classical_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(1)
    from experiments.neural.common import keep_full_speed
    keep_full_speed()
    test = load_test_set()
    c = test["contracts"]
    order = np.argsort(c.tau)
    idx = order[np.linspace(0, len(order) - 1, args.cases).astype(int)]
    sub = c.take(idx)
    out = {"cases": [str(test["case_id"][i]) for i in idx], "taus": [float(t) for t in sub.tau], "seconds": {}}
    def per_case(fn):
        t = perf_counter()
        for i in range(len(sub)):
            fn(sub.take([i]))
        return (perf_counter() - t) / len(sub)
    out["seconds"]["reference_numpy_12"] = per_case(lambda ci: reference(ci))
    for s in SETTINGS["cos"]:
        name = setting_name("cos", s)
        cos_quantities(sub.take([0]), s["terms"])
        out["seconds"][name] = per_case(lambda ci: cos_quantities(ci, s["terms"]))
        big = test["contracts"]
        t = perf_counter()
        for _ in range(16):
            cos_quantities(big, s["terms"])
        out["seconds"][name + "_batch"] = (perf_counter() - t) / (16 * len(big))
    for s in SETTINGS["adi"]:
        out["seconds"][setting_name("adi", s)] = per_case(
            lambda ci: adi_quantities(ci, GridSize(s["m1"], s["m2"], s["steps"]), V_MAX))
        print(out["seconds"], flush=True)
    for s in SETTINGS["rqmc"]:
        out["seconds"][setting_name("rqmc", s)] = per_case(
            lambda ci: rqmc_quantities(ci, s["log2_points"], s["dt"], 1, s["scrambles"]))
        print(setting_name("rqmc", s), out["seconds"][setting_name("rqmc", s)], flush=True)
    for s in SETTINGS["mc"]:
        out["seconds"][setting_name("mc", s)] = per_case(lambda ci: mc_quantities(ci, s["paths"], s["dt"], 1))
        print(setting_name("mc", s), out["seconds"][setting_name("mc", s)], flush=True)
    out["note"] = ("seconds per contract, all 12 quantities, one thread, nothing else running; the subset spans "
                   "the test set's maturities evenly; COS '_batch' prices the whole test set at once")
    write_json(OUT / "timing.json", out)


if __name__ == "__main__":
    main()
