"""Write the Phase 0 development contracts: 60 random domain contracts plus named corners.

Development seeds only (808000-808999); nothing here touches a test set. Random draws use
the protocol's training domain, with correlations from its factor model.
"""

from . import common  # noqa: F401  (pins BLAS threads before NumPy)

import argparse
import json

import numpy as np

from src.neural.coordinates import factor_loadings, factor_matrix, protocol_domain
from src.pricing import Basket

from .common import CONTRACTS, contract_to_dict, diagnostics

SEED = 808101
STRIKE = 100.0
D = 8
C_STAR_LOADINGS = [[0.9, 0.3, 0.0], [0.3, -0.5, 0.4], [0.6, 0.2, -0.5], [0.5, 0.5, 0.5],
                   [0.2, -0.4, -0.3], [0.8, 0.0, 0.4], [0.4, 0.5, -0.2], [0.7, -0.3, 0.1]]
C_STAR = dict(spots=(100, 95, 105, 100, 90, 110, 98, 102), strike=STRIKE, rate=0.03,
              volatilities=(0.1, 0.6, 0.3, 0.45, 0.2, 0.5, 0.35, 0.15),
              dividends=(0, 0.02, 0.04, 0.01, 0.03, 0.0, 0.05, 0.02),
              weights=(0.25, 0.2, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05),
              correlation=factor_matrix(np.array(C_STAR_LOADINGS)))
N1 = dict(spots=(100,) * D, strike=STRIKE, rate=0.03, volatilities=tuple(np.linspace(0.3, 0.45, D)),
          dividends=tuple(np.linspace(0, 0.03, D)), weights=(0.125,) * D, correlation=-0.12)


def random_contract(rng, domain, option):
    """One draw from the training domain (uniform box, Dirichlet weights above the floor)."""
    box = lambda name, size=None: rng.uniform(*domain[name], size)
    spots = STRIKE * box("moneyness_1", D)
    floor = domain["weight_floor"]
    weights = floor + (1 - D * floor) * rng.dirichlet(np.ones(D))
    correlation = factor_matrix(factor_loadings(rng, D, domain["correlation_model"]))
    return Basket(spots=spots, strike=STRIKE, tau=box("tau"), rate=box("rate"), volatilities=box("volatility_1", D),
                  dividends=box("dividend_1", D), weights=weights / weights.sum(), correlation=correlation,
                  option=option)


def named_corners(randoms):
    corners = [(f"cstar_tau{tau:g}", "C* (three-factor loadings, dispersed volatilities)", Basket(tau=tau, **C_STAR))
               for tau in (0.005, 0.02, 0.05, 1, 2, 3)]
    corners.append(("vol_dispersion", "volatilities alternating 0.05 and 0.75, correlation 0.3, tau 1",
                    Basket(**{**N1, "volatilities": (0.05, 0.75) * 4, "correlation": 0.3}, tau=1)))
    corners += [(f"n1_tau{tau:g}", "N1: all correlations -0.12 (smallest eigenvalue 0.16)", Basket(tau=tau, **N1))
                for tau in (0.02, 1)]
    block = np.full((D, D), 0.1)
    block[:4, :4] = block[4:, 4:] = 0.8
    np.fill_diagonal(block, 1)
    corners.append(("two_sector", "0.8 within sectors {1-4},{5-8}, 0.1 across, tau 1",
                    Basket(**{**N1, "correlation": block}, tau=1)))
    corners.append(("positive_min_eig_0.05", "all correlations 0.95 (smallest eigenvalue 0.05), tau 1",
                    Basket(**{**N1, "correlation": 0.95}, tau=1)))
    corners.append(("one_weight_0.72", "weights (0.72, 0.04, ..., 0.04), correlation 0.3, tau 1",
                    Basket(**{**N1, "weights": (0.72,) + (0.04,) * 7, "correlation": 0.3}, tau=1)))
    for option in ("call", "put"):
        for moneyness, scale in (("deep_itm", 1.3 if option == "call" else 0.7),
                                 ("deep_otm", 0.7 if option == "call" else 1.3)):
            corners.append((f"{moneyness}_{option}", f"spots {scale:g} K, correlation 0.3, tau 0.5",
                            Basket(**{**N1, "spots": (100 * scale,) * D, "correlation": 0.3}, tau=0.5,
                                   option=option)))
    for index in range(3):
        for tau in (0.005, 0.01):
            corners.append((f"random{index:02d}_tau{tau:g}", f"random{index:02d} at tau {tau:g}",
                            randoms[index].changed(tau=tau)))
    return corners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if CONTRACTS.exists() and not args.overwrite:
        raise FileExistsError(f"{CONTRACTS} exists; pass --overwrite to rewrite it")
    domain, _ = protocol_domain()
    rng = np.random.default_rng(SEED)
    randoms = [random_contract(rng, domain, "call" if i % 2 == 0 else "put") for i in range(60)]
    rows = [dict(name=f"random{i:02d}", group="random", description="domain draw", contract=contract_to_dict(c),
                 **diagnostics(c)) for i, c in enumerate(randoms)]
    rows += [dict(name=name, group="named", description=text, contract=contract_to_dict(c), **diagnostics(c))
             for name, text, c in named_corners(randoms)]
    CONTRACTS.parent.mkdir(parents=True, exist_ok=True)
    CONTRACTS.write_text(json.dumps(dict(version=1, seed=SEED, contracts=rows), indent=1), encoding="utf-8")
    print(f"{len(rows)} contracts written to {CONTRACTS}")


if __name__ == "__main__":
    main()
