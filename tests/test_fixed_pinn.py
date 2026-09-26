import unittest

import torch
from numpy.testing import assert_allclose

from src.neural.fixed_pinn import (
    FixedBlackScholesPINN,
    FixedBlackScholesProblem,
    sample_collocation,
)


_MODULE_DTYPE = None


def setUpModule():
    global _MODULE_DTYPE
    _MODULE_DTYPE = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)


def tearDownModule():
    torch.set_default_dtype(_MODULE_DTYPE)


class FixedPINNTests(unittest.TestCase):
    def setUp(self):
        self.problem = FixedBlackScholesProblem(100, 1, 0.05, 0.2, 0, 0.05, 4)
        self.model = FixedBlackScholesPINN(self.problem, hidden_layers=2, hidden_width=8)

    def test_boundary_conditions_are_exact(self):
        tau = torch.linspace(0, self.problem.T, 21).reshape(-1, 1)
        left = self.model(torch.full_like(tau, torch.log(torch.tensor(self.problem.x_min))), tau)
        right = self.model(torch.full_like(tau, torch.log(torch.tensor(self.problem.x_max))), tau)
        expected_right = self.problem.x_max - torch.exp(-self.problem.r * tau)
        assert_allclose(left.detach().numpy(), 0, atol=1e-14)
        assert_allclose(right.detach().numpy(), expected_right.numpy(), atol=1e-14)

    def test_residual_and_greeks_are_finite(self):
        x, tau = sample_collocation(self.problem, 64, 1234)
        residual = self.model.pde_residual(x, tau)
        values = self.model.price_and_greeks(x, tau)
        self.assertEqual(residual.shape, (64, 1))
        self.assertTrue(torch.all(torch.isfinite(residual)))
        self.assertTrue(all(torch.all(torch.isfinite(value)) for value in values.values()))

    def test_sampling_is_reproducible_and_concentrated(self):
        first = sample_collocation(self.problem, 100, 55, near_strike_fraction=0.3)
        second = sample_collocation(self.problem, 100, 55, near_strike_fraction=0.3)
        assert_allclose(first[0].detach().numpy(), second[0].detach().numpy())
        near = torch.exp(first[0]).detach().numpy()[-30:]
        self.assertTrue(((near >= 0.7) & (near <= 1.3)).all())


if __name__ == "__main__":
    unittest.main()
