"""Score the frozen selection once on heston_test_v1, with the Black-Scholes base and every level.

Usage (from heston/): python -m experiments.neural.evaluate_test

Checks selection.json (checkpoint and source hashes) and the test-set seal first. Writes
test_metrics.csv, test_predictions.csv, test_checks.json and neural_timing.json.
"""

import argparse
import json
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import torch

from experiments.neural.common import RESULTS, evaluation_set, keep_full_speed, load_settings
from experiments.records import ROOT, digest, write_csv, write_json
from experiments.test_set.load import load_test_set
from src.heston.model import QUANTITIES
from src.neural.coordinates import base_call, price_from_call, reduced_coordinates, rows_from_contracts
from src.neural.networks import load_model
from src.neural.reference import price_and_greeks

SPLITS = ("interpolation", "short_maturity", "extrapolation", "anchor")


def verify_selection():
    sel = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    for name, h in sel["evaluation_sources_sha256"].items():
        if digest(ROOT / name) != h:
            raise RuntimeError(f"{name} changed after the selection was frozen")
    for fam in sel["selected"].values():
        if digest(ROOT / fam["checkpoint"]) != fam["checkpoint_sha256"]:
            raise RuntimeError(f"{fam['checkpoint']} changed after the selection was frozen")
    return sel


def base_price(x, put):
    features, scale = reduced_coordinates(x)
    return price_from_call(base_call(features), features, scale, put)


def metrics(values, ref, mask):
    out = {}
    for q in QUANTITIES:
        e = np.abs(values[q][mask] - ref[q][mask])
        rms = np.sqrt(np.mean(ref[q][mask] ** 2))
        out[q] = dict(rmse=float(np.sqrt(np.mean(e**2))), median=float(np.median(e)), max=float(e.max()),
                      ratio=float(np.sqrt(np.mean(e**2)) / rms) if rms > 0 else float("nan"))
    return out


def arbitrage_checks(values, x, put, tol=1e-10):
    S, tau, r, q = (x[:, i] for i in (0, 1, 2, 3))
    carry, disc = np.exp(-q * tau), np.exp(-r * tau)
    intrinsic = np.where(put, np.maximum(disc - S * carry, 0), np.maximum(S * carry - disc, 0))
    lo = np.where(put, -carry, 0.0)
    hi = np.where(put, 0.0, carry)
    return dict(price_below_intrinsic=int(np.sum(values["price"] < intrinsic - tol)),
                delta_out_of_bounds=int(np.sum((values["delta"] < lo - tol) | (values["delta"] > hi + tol))),
                negative_gamma=int(np.sum(values["gamma"] < -tol)),
                negative_vega=int(np.sum(values["vega"] < -tol)), cases=int(len(S)))


def timing(price_fn, x, put, single=100):
    xs, ps = x[:4096], put[:4096]
    price_and_greeks(price_fn, xs[:64], ps[:64])
    t = perf_counter()
    price_and_greeks(price_fn, xs, ps, chunk=4096)
    batch = (perf_counter() - t) / len(xs)
    t = perf_counter()
    for i in range(single):
        price_and_greeks(price_fn, xs[i:i + 1], ps[i:i + 1])
    one = (perf_counter() - t) / single
    return dict(batch_seconds=batch, single_seconds=one, batch=len(xs))


def coverage(values, test, saved, setting):
    """Share of cases whose value lies in the Monte Carlo 95% interval of a classical run."""
    out = {}
    for q in QUANTITIES:
        m, se = saved.get(f"{setting}__{q}"), saved.get(f"{setting}__se_{q}")
        if m is None:
            return None
        out[q] = float(np.mean(np.abs(values[q] - m) <= 1.96 * se))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    keep_full_speed()
    torch.set_num_threads(args.threads)
    sel = verify_selection()
    seal = (ROOT / "results/test_set_v1/protocol.sha256").read_text(encoding="utf-8").strip()
    if seal != sel["test_set"]["protocol_sha256"]:
        raise RuntimeError("the test-set seal differs from the one recorded at selection")
    opened_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"test-set seal verified {seal} at {opened_at}", flush=True)
    test = load_test_set()
    if test["protocol"]["cases_sha256"] != digest(ROOT / "results/test_set_v1/cases.json"):
        raise RuntimeError("test set changed")
    x, put = rows_from_contracts(test["contracts"])
    ref = test["reference"]
    settings = load_settings()
    seed = settings["seeds"]["training"]
    pricers = {"bs_base": base_price}
    labels = {"bs_base": "Black-Scholes at the expected average variance"}
    for fam, s in sel["selected"].items():
        model = load_model(ROOT / s["checkpoint"])
        pricers[f"{fam}_selected"] = model.price
        labels[f"{fam}_selected"] = s["level"]
    for level in sel["ladder"]:
        if level not in (s["level"] for s in sel["selected"].values()):
            pricers[level] = load_model(ROOT / sel["ladder"][level]["checkpoint"]).price
            labels[level] = level
    rows, preds, checks = [], [], {}
    values_all = {}
    for name, fn in pricers.items():
        values = price_and_greeks(fn, x, put)
        values_all[name] = values
        for split in SPLITS + ("all",):
            mask = np.ones(len(x), bool) if split == "all" else test["split"] == split
            m = metrics(values, ref, mask)
            rows.append(dict(pricer=name, level=labels[name], split=split, cases=int(mask.sum()),
                             score=float(np.mean([m[q]["ratio"] for q in QUANTITIES])),
                             **{f"{k}_{q}": m[q][k] for q in QUANTITIES for k in ("rmse", "median", "max")}))
        checks[name] = arbitrage_checks(values, x, put)
    for i, cid in enumerate(test["case_id"]):
        for name in ("bs_base", "pinn_selected", "deeponet_selected"):
            preds.append(dict(case_id=cid, split=test["split"][i], pricer=name,
                              **{q: values_all[name][q][i] for q in QUANTITIES}))
    write_csv(RESULTS / "test_metrics.csv", rows)
    write_csv(RESULTS / "test_predictions.csv", preds)
    # Greek validation against the classical Monte Carlo intervals on the same cases
    cls = ROOT / "results/classical_v1/values.npz"
    cov = {}
    if cls.exists():
        saved = dict(np.load(cls, allow_pickle=True))
        for setting in ("mc_100000_dt32", "mc_1000000_dt32"):
            cov[setting] = {name: coverage(values_all[name] if name != "reference" else ref, test, saved, setting)
                            for name in ("reference", "pinn_selected", "deeponet_selected")}
    # no-arbitrage on the validation panels too
    for kind in ("validation", "short_validation"):
        panel = evaluation_set(kind)
        xv, pv = panel["inputs"]
        for name in ("pinn_selected", "deeponet_selected"):
            checks[f"{name}_{kind}"] = arbitrage_checks(price_and_greeks(pricers[name], xv, pv), xv, pv)
    write_json(RESULTS / "test_checks.json", dict(seal_verified=seal, opened_at=opened_at, no_arbitrage=checks,
                                                  monte_carlo_coverage_95=cov))
    # timing, one thread, on validation contracts
    xv, pv = evaluation_set("validation")["inputs"]
    tm = {name: timing(pricers[name], xv, pv) for name in ("bs_base", "pinn_selected", "deeponet_selected")}
    tm["threads"] = args.threads
    write_json(RESULTS / "neural_timing.json", tm)
    print(json.dumps({r["pricer"]: round(r["score"], 5) for r in rows if r["split"] == "all"}, indent=1))


if __name__ == "__main__":
    main()
