"""Write the Phase 0 development contracts: 75 random domain contracts plus named corners.

Development seeds only (404000-404999); nothing here touches a test set.
"""

from . import common  # noqa: F401  (pins BLAS threads before NumPy)

import argparse
import json

import numpy as np

from src.pricing import Basket

from .common import CONTRACTS, contract_to_dict, diagnostics

SEED = 404101
STRIKE = 100.0
C_STAR_CORRELATION = [[1, -0.5, 0.3, 0.6], [-0.5, 1, -0.4, -0.3], [0.3, -0.4, 1, 0.2], [0.6, -0.3, 0.2, 1]]
C_STAR = dict(spots=(100, 95, 105, 100), strike=STRIKE, rate=0.03, volatilities=(0.1, 0.6, 0.3, 0.45),
              dividends=(0, 0.02, 0.04, 0.01), weights=(0.4, 0.3, 0.2, 0.1), correlation=C_STAR_CORRELATION)
N1 = dict(spots=(100,) * 4, strike=STRIKE, rate=0.03, volatilities=(0.3, 0.35, 0.4, 0.45),
          dividends=(0, 0.01, 0.02, 0.03), weights=(0.4, 0.3, 0.2, 0.1), correlation=-0.3)


def random_contract(rng, option):
    """One draw from the development domain; weights and correlations are redrawn until valid."""
    spots = STRIKE * rng.uniform(0.7, 1.3, 4)
    tau = rng.uniform(0.05, 2)
    volatilities = rng.uniform(0.1, 0.6, 4)
    rate = rng.uniform(-0.02, 0.10)
    dividends = rng.uniform(0, 0.06, 4)
    while True:
        weights = rng.dirichlet(np.ones(4))
        if weights.min() >= 0.1:
            break
    upper = np.triu_indices(4, 1)
    while True:
        correlation = np.eye(4)
        correlation[upper] = rng.uniform(-0.5, 0.9, 6)
        correlation = correlation + np.triu(correlation, 1).T
        if np.linalg.eigvalsh(correlation)[0] >= 0.05:
            break
    return Basket(spots=spots, strike=STRIKE, tau=tau, rate=rate, volatilities=volatilities,
                  dividends=dividends, weights=weights / weights.sum(), correlation=correlation, option=option)


def named_corners(randoms):
    corners = [(f"cstar_tau{tau:g}", "C* (spec contract)", Basket(tau=tau, **C_STAR))
               for tau in (0.005, 0.02, 0.05, 1, 2, 3)]
    corners.append(("vol_dispersion", "vols 0.05 to 0.75, correlation 0.3, tau 1",
                    Basket(**{**N1, "volatilities": (0.05, 0.75, 0.25, 0.5), "correlation": 0.3}, tau=1)))
    corners += [(f"n1_tau{tau:g}", "N1: all correlations -0.3", Basket(tau=tau, **N1)) for tau in (0.02, 1)]
    block = np.full((4, 4), 0.1)
    block[:2, :2] = block[2:, 2:] = 0.8
    np.fill_diagonal(block, 1)
    corners.append(("two_sector", "0.8 within sectors {1,2},{3,4}, 0.1 across, tau 1",
                    Basket(**{**N1, "correlation": block}, tau=1)))
    corners.append(("positive_min_eig_0.05", "all correlations 0.95 (smallest eigenvalue 0.05), tau 1",
                    Basket(**{**N1, "correlation": 0.95}, tau=1)))
    corners.append(("one_weight_0.7", "weights (0.7, 0.1, 0.1, 0.1), correlation 0.3, tau 1",
                    Basket(**{**N1, "weights": (0.7, 0.1, 0.1, 0.1), "correlation": 0.3}, tau=1)))
    for option in ("call", "put"):
        for moneyness, scale in (("deep_itm", 1.3 if option == "call" else 0.7),
                                 ("deep_otm", 0.7 if option == "call" else 1.3)):
            corners.append((f"{moneyness}_{option}", f"spots {scale:g} K, correlation 0.3, tau 0.5",
                            Basket(**{**N1, "spots": (100 * scale,) * 4, "correlation": 0.3}, tau=0.5,
                                   option=option)))
    for index in range(3):
        for tau in (0.005, 0.01):
            corners.append((f"random{index:02d}_tau{tau:g}", f"random{index:02d} at tau {tau:g}",
                            randoms[index].changed(tau=tau)))
    corners.append(("e4_stress", "N1 with vols 0.3 and correlations -0.33: outside the domain",
                    Basket(**{**N1, "volatilities": (0.3,) * 4, "correlation": -0.33}, tau=1)))
    return corners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if CONTRACTS.exists() and not args.overwrite:
        raise FileExistsError(f"{CONTRACTS} exists; pass --overwrite to rewrite it")
    rng = np.random.default_rng(SEED)
    randoms = [random_contract(rng, "call" if i % 2 == 0 else "put") for i in range(75)]
    rows = [dict(name=f"random{i:02d}", group="random", description="domain draw", contract=contract_to_dict(c),
                 **diagnostics(c)) for i, c in enumerate(randoms)]
    rows += [dict(name=name, group="named", description=text, contract=contract_to_dict(c), **diagnostics(c))
             for name, text, c in named_corners(randoms)]
    CONTRACTS.parent.mkdir(parents=True, exist_ok=True)
    CONTRACTS.write_text(json.dumps(dict(version=1, seed=SEED, contracts=rows), indent=1), encoding="utf-8")
    print(f"{len(rows)} contracts written to {CONTRACTS}")


if __name__ == "__main__":
    main()
