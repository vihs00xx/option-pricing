"""Calibration test: fit (v0, kappa, theta, xi, rho) to a synthetic surface of 100 options.

The surface: 10 moneyness levels (S/K 0.8 to 1.2) x 10 maturities (0.1 to 2 years), out-of-the-
money calls and puts, priced by the Lewis reference at fixed true parameters. Each pricer
(COS with the frozen rule at 128 and 256 terms, the selected PINN, the selected DeepONet)
is calibrated from the same 8 random starting points inside the training domain by a
box-constrained Levenberg-Marquardt on the price residuals. COS gets its Jacobian from
its own sensitivities (analytic Vega, bumped parameter sensitivities); the networks get
theirs exactly from autograd. One thread.

Usage (from heston/): python -m experiments.neural.calibration
"""

import json
from time import perf_counter

import numpy as np
import torch

from experiments.neural.common import RESULTS, keep_full_speed, ranges
from experiments.records import ROOT, write_csv, write_json
from src.heston.cos import cos_quantities
from src.heston.model import Contracts
from src.heston.reference import reference
from src.neural.coordinates import COLUMN, rows_from_contracts
from src.neural.networks import load_model

PARAMS = ("v0", "kappa", "theta", "xi", "rho")
TRUE = dict(v0=0.04, kappa=1.5, theta=0.06, xi=0.6, rho=-0.65)
MARKET = dict(r=0.02, q=0.01)
STARTS = 8
SEED = 20261008


def surface():
    m, t = np.meshgrid(np.linspace(0.8, 1.2, 10), np.linspace(0.1, 2.0, 10), indexing="ij")
    m, t = m.ravel(), t.ravel()
    n = len(m)
    return Contracts(S=m, tau=t, r=np.full(n, MARKET["r"]), q=np.full(n, MARKET["q"]),
                     v0=np.full(n, TRUE["v0"]), kappa=np.full(n, TRUE["kappa"]), theta=np.full(n, TRUE["theta"]),
                     xi=np.full(n, TRUE["xi"]), rho=np.full(n, TRUE["rho"]), call=m <= 1.0)


def with_params(c, p):
    return c.with_values(**dict(zip(PARAMS, p)))


def cos_model(terms):
    def f(c, p):
        out = cos_quantities(with_params(c, p), terms)
        J = np.stack([out["vega"], out["d_kappa"], out["d_theta"], out["d_xi"], out["d_rho"]], axis=1)
        return out["price"], J
    return f


def network_model(model):
    cols = [COLUMN[n] for n in PARAMS]

    def f(c, p):
        x, put = rows_from_contracts(with_params(c, p))
        xt = torch.as_tensor(x).requires_grad_(True)
        V = model.price(xt, torch.as_tensor(put))
        g, = torch.autograd.grad(V.sum(), xt)
        return V.detach().numpy(), g[:, cols].numpy()
    return f


def levenberg_marquardt(f, c, target, p0, lo, hi, max_iter=200, tol=1e-14):
    """Box-constrained LM: damped Gauss-Newton steps projected onto [lo, hi]."""
    p = np.clip(np.array(p0, float), lo, hi)
    price, J = f(c, p)
    r = price - target
    cost = r @ r
    lam = 1e-3
    evals = 1
    for it in range(1, max_iter + 1):
        A = J.T @ J
        g = J.T @ r
        accepted = False
        for _ in range(30):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            q = np.clip(p + step, lo, hi)
            price_q, J_q = f(c, q)
            evals += 1
            r_q = price_q - target
            cost_q = r_q @ r_q
            if cost_q < cost:
                accepted = True
                break
            lam *= 4
        if not accepted:
            break
        improvement = (cost - cost_q) / max(cost, 1e-300)
        p, J, r, cost = q, J_q, r_q, cost_q
        lam = max(lam / 3, 1e-12)
        if improvement < tol or cost < 1e-28:
            break
    return p, float(np.sqrt(cost / len(r))), it, evals


def main():
    keep_full_speed()
    torch.set_num_threads(1)
    sel = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    c = surface()
    target = reference(c, full=False)[0]["price"]
    rg = ranges()
    lo = np.array([rg[n][0] for n in PARAMS])
    hi = np.array([rg[n][1] for n in PARAMS])
    rng = np.random.default_rng(SEED)
    starts = lo + (hi - lo) * rng.random((STARTS, len(PARAMS)))
    pricers = {"cos_128": cos_model(128), "cos_256": cos_model(256)}
    for fam, s in sel["selected"].items():
        pricers[f"{fam}_{s['level']}"] = network_model(load_model(ROOT / s["checkpoint"]))
    true = np.array([TRUE[n] for n in PARAMS])
    rows, summary = [], {}
    for name, f in pricers.items():
        f(c, starts[0])     # warm-up
        for k, p0 in enumerate(starts):
            t = perf_counter()
            p, rmse, iters, evals = levenberg_marquardt(f, c, target, p0, lo, hi)
            seconds = perf_counter() - t
            err = np.abs(p - true) / np.abs(true)
            rows.append(dict(pricer=name, start=k, seconds=seconds, iterations=iters, evaluations=evals,
                             fit_rmse=rmse, max_rel_param_error=float(err.max()),
                             **{f"fit_{n}": float(v) for n, v in zip(PARAMS, p)},
                             **{f"start_{n}": float(v) for n, v in zip(PARAMS, p0)}))
        rs = [r for r in rows if r["pricer"] == name]
        summary[name] = dict(seconds_mean=float(np.mean([r["seconds"] for r in rs])),
                             iterations_mean=float(np.mean([r["iterations"] for r in rs])),
                             median_max_rel_param_error=float(np.median([r["max_rel_param_error"] for r in rs])),
                             recovered_within_1pct=int(sum(r["max_rel_param_error"] < 0.01 for r in rs)),
                             recovered_within_10pct=int(sum(r["max_rel_param_error"] < 0.10 for r in rs)),
                             median_fit_rmse=float(np.median([r["fit_rmse"] for r in rs])))
        print(name, summary[name], flush=True)
    write_csv(RESULTS / "calibration.csv", rows)
    write_json(RESULTS / "calibration.json", dict(true=TRUE, market=MARKET, options=len(c), starts=STARTS, seed=SEED,
                                                  summary=summary, target="Lewis reference prices"))


if __name__ == "__main__":
    main()
