"""For d = 2 the general neural code reproduces the frozen two-stock study (2d/, read only).

The two-stock package is loaded under another name from the sibling folder; its
committed ladder checkpoints load into the general classes and give the same prices,
Greeks, PDE residuals and labels. Skipped when 2d/ is not next to this folder.
"""

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import torch

from src.neural.coordinates import Stocks, levy_call, reduced_coordinates
from src.neural.monte_carlo import monte_carlo_estimates
from src.neural.networks import build_model
from src.neural.pde import raw_residual, reduced_residual
from src.neural.reference import call_function, call_pricer, derivative_labels, price_and_greeks, quantities

TWO = Path(__file__).resolve().parents[2]/"2d"
LADDER = TWO/"results/neural/pinn_deeponet_v1/ladder"
STOCKS = Stocks(2)
NAMES = {"s": "s_1", "rho": "rho_12", "correlation": "correlation_12"}
CHECKPOINTS = ("P0_vanilla_seed0", "P2_hard_payoff_seed0", "P5_similarity_seed0", "P9_envelope_seed0",
               "P10_anchored_seed0", "D0_vanilla_seed0", "D3_differential_seed0", "D5_similarity_seed0",
               "D8_envelope_seed3", "D9_ensemble_seed0")


def two_stock(module):
    """A module of the two-stock package, imported as ``two_stock_src``."""
    if "two_stock_src" not in sys.modules:
        spec = importlib.util.spec_from_file_location("two_stock_src", TWO/"src/__init__.py",
                                                      submodule_search_locations=[str(TWO/"src")])
        package = importlib.util.module_from_spec(spec)
        sys.modules["two_stock_src"] = package
        spec.loader.exec_module(package)
    return importlib.import_module(f"two_stock_src.{module}")


def general_model(record):
    """The general class with a two-stock checkpoint's settings and weights."""
    settings = dict(record["settings"])
    settings["boxes"] = {NAMES.get(k, k): v for k, v in settings["boxes"].items()}
    model = build_model(record["family"], d=2, **settings)
    model.load_state_dict(record["state_dict"])
    return model.eval()


@unittest.skipUnless(LADDER.exists(), "needs the two-stock study in ../2d")
class TwoStockReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        cls.coordinates = two_stock("neural.coordinates")
        cls.reference = two_stock("neural.reference")
        domain, strike = cls.coordinates.protocol_domain()
        cls.x, cls.put = cls.coordinates.sample_domain(domain, strike, 32, seed=17)
        cls.short, cls.short_put = cls.coordinates.sample_domain(domain, strike, 16, seed=18, tau=(0.005, 0.05))

    def test_fields_and_quantities_line_up(self):
        self.assertEqual(STOCKS.fields, tuple(NAMES.get(f, f) for f in self.coordinates.FIELDS))
        self.assertEqual(quantities(STOCKS), self.reference.QUANTITIES)

    def test_reduced_coordinates_and_levy(self):
        x = torch.as_tensor(self.x)
        mine, scale = reduced_coordinates(STOCKS, x)
        theirs, their_scale = self.coordinates.reduced_coordinates(x)
        torch.testing.assert_close(mine, theirs, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(scale, their_scale, rtol=0, atol=0)
        torch.testing.assert_close(levy_call(STOCKS, mine), self.coordinates.levy_call(theirs), rtol=1e-14, atol=1e-15)

    def test_gauss_hermite_twin_matches_the_trapezoid_reference(self):
        for x, put in ((self.x[:8], self.put[:8]), (self.short[:8], self.short_put[:8])):
            mine = price_and_greeks(STOCKS, call_pricer(STOCKS, call_function(STOCKS, 128)), x, put)
            theirs = self.reference.reference_values(x, put)
            for q in quantities(STOCKS):
                assert_allclose(mine[q], theirs[q], rtol=1e-9, atol=1e-10, err_msg=q)

    def test_checkpoints_give_the_same_prices_and_greeks(self):
        networks = two_stock("neural.networks")
        x = np.vstack((self.x, self.short))
        put = np.concatenate((self.put, self.short_put))
        for name in CHECKPOINTS:
            theirs_model = networks.load_model(LADDER/f"{name}.pt")
            record = torch.load(LADDER/f"{name}.pt", map_location="cpu", weights_only=False)
            members = record["ensemble"] if "ensemble" in record else [record]
            mine_models = [general_model(m) for m in members]
            mine_price = lambda xs, ps: torch.stack([m.price(xs, ps) for m in mine_models]).mean(0)
            mine = price_and_greeks(STOCKS, mine_price, x, put)
            theirs = self.reference.price_and_greeks(theirs_model.price, x, put)
            for q in quantities(STOCKS):
                assert_allclose(mine[q], theirs[q], rtol=1e-12, atol=1e-12, err_msg=f"{name} {q}")

    def test_pde_residuals_and_labels(self):
        pde = two_stock("neural.pde")
        record = torch.load(LADDER/"P9_envelope_seed0.pt", map_location="cpu", weights_only=False)
        mine_model = general_model(record)
        theirs_model = two_stock("neural.networks").load_model(LADDER/"P9_envelope_seed0.pt")
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        mine, _ = reduced_residual(STOCKS, mine_model.call, features)
        theirs, _ = pde.reduced_residual(theirs_model.call, features)
        torch.testing.assert_close(mine, theirs, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(derivative_labels(STOCKS, mine_model.call, features),
                                   self.reference.derivative_labels(theirs_model.call, features),
                                   rtol=1e-12, atol=1e-12)
        raw = torch.load(LADDER/"P0_vanilla_seed0.pt", map_location="cpu", weights_only=False)
        mine_raw, theirs_raw = general_model(raw), two_stock("neural.networks").load_model(LADDER/"P0_vanilla_seed0.pt")
        x, put = torch.as_tensor(self.x), torch.as_tensor(self.put)
        torch.testing.assert_close(raw_residual(STOCKS, mine_raw.normalized_price, x, put),
                                   pde.raw_residual(theirs_raw.normalized_price, x, put), rtol=1e-12, atol=1e-12)

    def test_monte_carlo_draws_the_same_paths(self):
        theirs_mc = two_stock("neural.monte_carlo")
        for i in range(3):
            mine = monte_carlo_estimates(STOCKS, self.x[i], self.put[i], 50_000, seed=31+i)
            theirs = theirs_mc.monte_carlo_estimates(self.x[i], self.put[i], 50_000, seed=31+i)
            for q in quantities(STOCKS):
                assert_allclose(mine[q], theirs[q], rtol=1e-10, atol=1e-12, err_msg=q)


if __name__ == "__main__":
    unittest.main()
