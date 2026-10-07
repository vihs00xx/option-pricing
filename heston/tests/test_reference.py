import unittest

import numpy as np
import torch

from src.heston import torch_reference as tr
from src.heston.black_scholes import bs_price
from src.heston.characteristic import cf, cumulants
from src.heston.cos import cos_quantities
from src.heston.model import (QUANTITIES, Contracts, average_variance, development_contracts,
                              hard_contracts)
from src.heston.reference import (gil_pelaez_price, lewis_integrands, price_on_nodes, reference,
                                  richardson)


def sample(n_domain=6, n_short=3):
    main, short, hard = development_contracts()
    return Contracts.concat(main.take(np.arange(n_domain)), short.take(np.arange(n_short)),
                            hard.take([0, 3, 4, 5]))


class TestReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.c = sample()
        cls.ref, cls.info = reference(cls.c, keep_nodes=True)

    def test_converged(self):
        self.assertTrue(self.info["converged"].all())

    def test_put_call_parity(self):
        calls = self.c.with_values(call=True)
        puts = self.c.with_values(call=False)
        pc, _ = reference(calls, full=False)
        pp, _ = reference(puts, full=False)
        forward = self.c.S * np.exp(-self.c.q * self.c.tau) - self.c.discount
        np.testing.assert_allclose(pc["price"] - pp["price"], forward, atol=1e-13)

    def test_lewis_form_reduces_to_black_scholes(self):
        # Lewis integral with the Black-Scholes phi(u - i/2) = exp(-w tau (u^2 + 1/4) / 2)
        from scipy.integrate import quad
        for x in (-0.3, 0.0, 0.25):
            for wt in (0.0004, 0.04, 0.5):
                f = lambda u: np.cos(u * x) * np.exp(-0.5 * wt * (u * u + 0.25)) / (u * u + 0.25)
                val, _ = quad(f, 0, np.inf, epsabs=1e-14, epsrel=1e-13, limit=500)
                c = np.exp(x) - np.exp(x / 2) / np.pi * val
                self.assertAlmostEqual(c, float(bs_price(x, wt, 1.0, 0.0)), delta=1e-11)

    def test_black_scholes_limit(self):
        c = self.c.take(np.arange(6))
        errs = []
        for xi in (1e-2, 1e-3):
            cc = c.with_values(xi=xi, rho=0.0)
            p, _ = reference(cc, full=False)
            bs = bs_price(cc.x, average_variance(cc.v0, cc.theta, cc.kappa, cc.tau), cc.tau, cc.r, cc.call)
            errs.append(np.max(np.abs(p["price"] - bs)))
        self.assertLess(errs[0], 1e-4)
        self.assertGreater(errs[0] / errs[1], 50)  # error falls like xi^2

    def test_lewis_vs_gil_pelaez(self):
        for i in range(len(self.c)):
            gp = gil_pelaez_price(self.c.take([i]))
            self.assertAlmostEqual(gp, self.ref["price"][i], delta=1e-11)

    def test_parameter_derivatives_vs_richardson(self):
        for i in (0, 4, 7):
            c1 = self.c.take([i])
            nd, wt = self.info["nodes"][i]
            for p, h in (("kappa", 1e-3), ("theta", 1e-4), ("xi", 1e-3), ("rho", 1e-3)):
                self.assertAlmostEqual(richardson(c1, p, h, nd, wt), self.ref[f"d_{p}"][i], delta=1e-8)
            theta_fd = -richardson(c1, "tau", min(1e-4, 0.05 * c1.tau[0]), nd, wt)
            self.assertAlmostEqual(theta_fd, self.ref["theta_cal"][i], delta=1e-7)

    def test_fixed_rule_matches_adaptive(self):
        for i in range(len(self.c)):
            nd, wt = self.info["nodes"][i]
            self.assertAlmostEqual(price_on_nodes(self.c.take([i]), nd, wt)[0], self.ref["price"][i], delta=1e-12)

    def test_torch_twin_matches_numpy(self):
        out, info = tr.twin(self.c)
        self.assertTrue(info["converged"].all())
        for q in QUANTITIES:
            np.testing.assert_allclose(out[q], self.ref[q], atol=1e-9, rtol=0, err_msg=q)

    def test_cos_vs_reference(self):
        main, _, _ = development_contracts()
        c = main.take(np.arange(40))
        ref, _ = reference(c)
        co = cos_quantities(c, 512, rule="c2", L=12.0, parameter_greeks=False)
        ok = ~((2 * c.kappa * c.theta < 0.25 * c.xi**2))  # strong Feller violations need wider truncation
        for q, tol in (("price", 1e-6), ("delta", 1e-5), ("gamma", 1e-3)):
            self.assertLess(np.max(np.abs(co[q] - ref[q])[ok]), tol, q)

    def test_cos_all_quantities(self):
        main, _, _ = development_contracts()
        c = main.take(np.arange(3, 23))
        ref, _ = reference(c)
        co = cos_quantities(c, 1024, rule="c4", L=8.0)
        for q in QUANTITIES:
            scale = max(1.0, np.max(np.abs(ref[q])))
            self.assertLess(np.max(np.abs(co[q] - ref[q])) / scale, 1e-5, q)

    def test_cumulants_match_characteristic_function(self):
        c = self.c
        for i in range(len(c)):
            args = (c.tau[i], c.v0[i], c.kappa[i], c.theta[i], c.xi[i], c.rho[i])
            lp = lambda u: np.log(cf(np.array([u + 0j]), *args))[0]
            h = 1e-3
            mean = ((lp(h) - lp(-h)) / (2 * h)).imag
            var = -((lp(h) - 2 * lp(0.0) + lp(-h)) / h**2).real
            m, v = cumulants(*args)
            self.assertAlmostEqual(m, mean, delta=1e-6 * abs(m) + 1e-12)
            self.assertAlmostEqual(v, var, delta=1e-4 * v)

    def test_martingale(self):
        c = hard_contracts()
        for i in range(len(c)):
            phi = cf(np.array([-1j]), c.tau[i], c.v0[i], c.kappa[i], c.theta[i], c.xi[i], c.rho[i])
            self.assertAlmostEqual(abs(phi[0] - 1), 0.0, delta=1e-14)


if __name__ == "__main__":
    unittest.main()
