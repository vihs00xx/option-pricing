import unittest

import numpy as np

from src.heston.adi import GridSize, adi_quantities
from src.heston.model import QUANTITIES, Contracts, development_contracts, hard_contracts
from src.heston.monte_carlo import brownian_bridge, mc_quantities, rqmc_quantities
from src.heston.reference import reference

V_MAX = 15.28359698190767


class TestClassical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main, _, _ = development_contracts()
        cls.c = Contracts.concat(main.take([0, 1, 2]), hard_contracts().take([4]))
        cls.ref, _ = reference(cls.c)

    def test_adi_converges(self):
        errs = []
        for size in (GridSize(40, 20, 20), GridSize(80, 40, 40)):
            a = adi_quantities(self.c, size, V_MAX, parameter_greeks=False)
            errs.append(np.max(np.abs(a["price"] - self.ref["price"])))
        self.assertLess(errs[1], 5e-4)
        self.assertGreater(errs[0] / errs[1], 2.5)     # second order in the grid

    def test_adi_parameter_greeks(self):
        a = adi_quantities(self.c.take([0]), GridSize(80, 40, 40), V_MAX)
        for q in ("rho_rate", "d_kappa", "d_theta", "d_xi", "d_rho", "theta_cal"):
            scale = max(1.0, abs(self.ref[q][0]))
            self.assertLess(abs(a[q][0] - self.ref[q][0]) / scale, 2e-3, q)

    def test_adi_base_value_far_condition(self):
        a = adi_quantities(self.c.take([1]), GridSize(80, 40, 40), 3.0, parameter_greeks=False, far_v="base_value")
        self.assertLess(abs(a["price"][0] - self.ref["price"][1]), 5e-4)

    def test_monte_carlo_unbiased(self):
        c1 = self.c.take([1])
        v, se = mc_quantities(c1, 40_000, 1 / 32, seed=3)
        for q in ("price", "delta", "vega", "d_theta"):
            self.assertLess(abs(v[q] - self.ref[q][1]), 5 * se[q] + 1e-4, q)

    def test_rqmc_unbiased(self):
        c1 = self.c.take([2])
        v, se = rqmc_quantities(c1, 10, 1 / 32, seed=4, scrambles=8)
        for q in ("price", "delta", "vega"):
            self.assertLess(abs(v[q] - self.ref[q][2]), 6 * se[q] + 2e-4, q)

    def test_brownian_bridge_increments(self):
        z = np.random.default_rng(0).standard_normal((20000, 7))
        inc = brownian_bridge(z)
        cov = np.cov(inc.T)
        np.testing.assert_allclose(cov, np.eye(7), atol=0.05)


if __name__ == "__main__":
    unittest.main()
