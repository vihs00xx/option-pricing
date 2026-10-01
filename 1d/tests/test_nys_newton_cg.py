import unittest

import torch
from numpy.testing import assert_allclose

from src.neural.fixed_pinn import FixedBlackScholesPINN, FixedBlackScholesProblem
from src.neural.nys_newton_cg import NysNewtonCG, flatten, unflatten


_MODULE_DTYPE = None


def setUpModule():
    global _MODULE_DTYPE
    _MODULE_DTYPE = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)


def tearDownModule():
    torch.set_default_dtype(_MODULE_DTYPE)


def finite_difference_hessian_vector(loss_fn, params, vector, epsilon=1e-6):
    for parameter, direction in zip(params, vector):
        parameter.data.add_(direction, alpha=epsilon)
    plus = torch.autograd.grad(loss_fn(), params)
    for parameter, direction in zip(params, vector):
        parameter.data.add_(direction, alpha=-2 * epsilon)
    minus = torch.autograd.grad(loss_fn(), params)
    for parameter, direction in zip(params, vector):
        parameter.data.add_(direction, alpha=epsilon)
    return [(a - b) / (2 * epsilon) for a, b in zip(plus, minus)]


class FlattenTests(unittest.TestCase):
    def test_round_trip(self):
        torch.set_default_dtype(torch.float64)
        first = torch.randn(2, 3)
        second = torch.randn(5)
        flat = flatten([first, second])
        restored = unflatten(flat, [first, second])
        assert_allclose(restored[0].numpy(), first.numpy())
        assert_allclose(restored[1].numpy(), second.numpy())
        self.assertEqual(flat.numel(), 11)


class HessianVectorTests(unittest.TestCase):
    def test_hessian_vector_matches_finite_differences(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(3)
        problem = FixedBlackScholesProblem(100, 1, 0.05, 0.2, 0, 0.05, 4)
        model = FixedBlackScholesPINN(problem, hidden_layers=2, hidden_width=8)
        y = torch.tensor([[0.2], [0.9], [1.1]])
        y.requires_grad_(True)
        tau = torch.tensor([[0.2], [0.5], [0.9]])
        tau.requires_grad_(True)

        def loss_fn():
            residual = model.pde_residual(y, tau)
            return torch.mean(residual**2)

        params = list(model.parameters())
        grads = torch.autograd.grad(loss_fn(), params, create_graph=True)
        vector = [torch.randn_like(p) * 0.01 for p in params]
        dot = sum(torch.sum(g * v) for g, v in zip(grads, vector))
        hv = torch.autograd.grad(dot, params)
        reference = finite_difference_hessian_vector(loss_fn, params, vector)
        for estimate, exact in zip(hv, reference):
            assert_allclose(
                estimate.detach().numpy(), exact.detach().numpy(), rtol=1e-4,
                atol=1e-8,
            )


class NysNewtonCGTests(unittest.TestCase):
    def _quadratic_problem(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(5)
        matrix = torch.randn(6, 6)
        matrix = matrix @ matrix.T + 0.5 * torch.eye(6)
        target = torch.randn(6)
        parameter = torch.zeros(6, requires_grad=True)

        def loss_fn():
            difference = matrix @ parameter - target
            return 0.5 * torch.sum(difference**2)

        return loss_fn, [parameter], matrix, target

    def test_minimises_a_quadratic(self):
        loss_fn, params, matrix, target = self._quadratic_problem()
        optimizer = NysNewtonCG(
            params, loss_fn, sketch_rank=4, cg_max_steps=20,
            damping=1e-10, seed=1,
        )
        first_loss = float(loss_fn().detach())
        for _ in range(3):
            result = optimizer.step()
            self.assertTrue(result["accepted"], result)
            self.assertLessEqual(result["loss_after"], result["loss_before"] + 1e-12)
        self.assertLess(float(loss_fn()), first_loss * 1e-6)
        expected = torch.linalg.solve(matrix, target)
        assert_allclose(params[0].detach().numpy(), expected.numpy(), rtol=1e-3)

    def test_line_search_never_increases_loss(self):
        loss_fn, params, _, _ = self._quadratic_problem()
        optimizer = NysNewtonCG(
            params, loss_fn, sketch_rank=3, cg_max_steps=5,
            damping=1e-4, seed=2,
        )
        before = float(loss_fn().detach())
        result = optimizer.step()
        self.assertTrue(result["accepted"])
        self.assertLessEqual(result["loss_after"], before)

    def test_reduces_pinn_loss(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(9)
        problem = FixedBlackScholesProblem(100, 1, 0.05, 0.2, 0, 0.05, 4)
        model = FixedBlackScholesPINN(problem, hidden_layers=2, hidden_width=8)
        y = torch.tensor([[0.3], [0.8], [1.0], [1.4]])
        y.requires_grad_(True)
        tau = torch.tensor([[0.2], [0.4], [0.7], [0.9]])
        tau.requires_grad_(True)

        def loss_fn():
            return torch.mean(model.pde_residual(y, tau)**2)

        params = list(model.parameters())
        optimizer = NysNewtonCG(
            params, loss_fn, sketch_rank=10, cg_max_steps=10,
            damping=1e-8, seed=3,
        )
        before = float(loss_fn().detach())
        for _ in range(2):
            result = optimizer.step()
            self.assertTrue(result["accepted"], result)
        self.assertLess(float(loss_fn().detach()), before)


if __name__ == "__main__":
    unittest.main()
