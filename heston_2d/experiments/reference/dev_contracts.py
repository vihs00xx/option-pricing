"""Write the Phase 0 development contracts: 40 random domain contracts plus named corners.

Development seeds only (909000-909999); nothing here touches a test set. Random draws use
the protocol's training domain (src/neural/coordinates.py sample_domain with its own seed).
Run from this folder: python -m experiments.reference.dev_contracts
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.neural.coordinates import Stocks, contract_of, protocol_domain, protocol_path, sample_domain, stock_correlation

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT/"experiments/config/reference/dev_contracts_v1.json"
SEED = 909101


def fields(c):
    return dict(spots=list(c.spots), strike=c.strike, tau=c.tau, rate=c.rate, dividends=list(c.dividends),
                weights=list(c.weights), v0=list(c.v0), kappa=list(c.kappa), theta=list(c.theta), xi=list(c.xi),
                leverage=list(c.leverage), correlation=[list(r) for r in c.correlation],
                vol_correlation=[list(r) for r in c.vol_correlation], option=c.option)


def with_leverage(c, value):
    """The contract with every leverage at ``value`` and its C and P kept (R follows)."""
    leverage = np.full(c.dimension, value)
    R = stock_correlation(c.vol_correlation_matrix(), c.residual_correlation_matrix(), leverage)
    return c.changed(leverage=tuple(leverage), correlation=R)


def named(base, d):
    """Corners of the domain: maturities, Feller violation, strong leverage, near Black-Scholes."""
    each = lambda value: (value,)*d
    rows = [(f"base_tau{t:g}", "random00 at another maturity", base.changed(tau=t)) for t in (0.005, 0.02, 0.5, 2.0, 3.0)]
    rows += [
        ("feller_violation", "kappa 0.5, theta 0.02, xi 1, v0 0.02",
         base.changed(kappa=each(0.5), theta=each(0.02), xi=each(1.0), v0=each(0.02))),
        ("strong_leverage", "leverage -0.9 and xi 0.8 for every stock", with_leverage(base, -0.9).changed(xi=each(0.8))),
        ("near_black_scholes", "xi 0.1, leverage 0, v0 = theta",
         with_leverage(base, 0.0).changed(xi=each(0.1), v0=base.theta)),
        ("high_v0_short", "v0 0.36, tau 0.05", base.changed(v0=each(0.36), tau=0.05)),
        ("deep_itm_call", "spots 1.3 K", base.changed(spots=each(130.0), option="call")),
        ("deep_otm_put", "spots 1.3 K", base.changed(spots=each(130.0), option="put")),
    ]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if CONTRACTS.exists() and not args.overwrite:
        raise FileExistsError(f"{CONTRACTS} exists; pass --overwrite to rewrite it")
    d = int(json.loads(protocol_path().read_text(encoding="utf-8"))["stocks"])
    stocks = Stocks(d)
    domain, strike = protocol_domain()
    x, put = sample_domain(stocks, domain, strike, 64, SEED)
    randoms = [contract_of(stocks, x[k], put[k]) for k in range(40)]
    rows = [dict(name=f"random{k:02d}", group="random", description="domain draw", contract=fields(c))
            for k, c in enumerate(randoms)]
    rows += [dict(name=name, group="named", description=text, contract=fields(c)) for name, text, c in named(randoms[0], d)]
    CONTRACTS.parent.mkdir(parents=True, exist_ok=True)
    CONTRACTS.write_text(json.dumps(dict(version=1, seed=SEED, contracts=rows), indent=1), encoding="utf-8")
    print(f"{len(rows)} contracts written to {CONTRACTS}")


if __name__ == "__main__":
    main()
