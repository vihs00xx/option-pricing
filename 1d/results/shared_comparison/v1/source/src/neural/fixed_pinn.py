from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FixedBlackScholesProblem:
    K: float
    T: float
    r: float
    sigma: float
    q: float
    x_min: float
    x_max: float


class FixedBlackScholesPINN(nn.Module):
    def __init__(self, problem, hidden_layers=4, hidden_width=64):
        super().__init__()
        self.problem = problem
        layers = [nn.Linear(2, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_width, hidden_width), nn.Tanh()))
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, y, tau):
        p = self.problem
        y_min = torch.log(torch.as_tensor(p.x_min, dtype=y.dtype, device=y.device))
        y_max = torch.log(torch.as_tensor(p.x_max, dtype=y.dtype, device=y.device))
        coordinates = torch.cat((2 * (y - y_min) / (y_max - y_min) - 1,
                                 2 * tau / p.T - 1), dim=1)
        correction = self.network(coordinates)
        right_boundary = p.x_max * torch.exp(-p.q * tau) - torch.exp(-p.r * tau)
        position = (y - y_min) / (y_max - y_min)
        base = position * right_boundary
        interior_shape = 4 * position * (1 - position)
        return base + interior_shape * correction

    def derivatives(self, y, tau, create_graph=True):
        value = self(y, tau)
        ones = torch.ones_like(value)
        value_y = torch.autograd.grad(value, y, ones, create_graph=True, retain_graph=True)[0]
        value_tau = torch.autograd.grad(
            value, tau, ones, create_graph=create_graph, retain_graph=True
        )[0]
        value_yy = torch.autograd.grad(
            value_y, y, torch.ones_like(value_y),
            create_graph=create_graph, retain_graph=True,
        )[0]
        return value, value_y, value_yy, value_tau

    def pde_residual(self, y, tau):
        p = self.problem
        value, value_y, value_yy, value_tau = self.derivatives(y, tau)
        drift = p.r - p.q - 0.5 * p.sigma**2
        return value_tau - 0.5 * p.sigma**2 * value_yy - drift * value_y + p.r * value

    def price_and_greeks(self, y, tau):
        value, value_y, value_yy, value_tau = self.derivatives(y, tau, create_graph=False)
        x = torch.exp(y)
        return {
            "price": self.problem.K * value,
            "delta": value_y / x,
            "gamma": (value_yy - value_y) / (self.problem.K * x**2),
            "theta": -self.problem.K * value_tau,
        }


def sample_collocation(problem, count, seed, near_strike_fraction=0.25,
                       near_strike_half_width=0.3, minimum_tau=1e-4,
                       dtype=torch.float64):
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    unit = engine.draw(count).to(dtype=dtype)
    y_min = torch.log(torch.tensor(problem.x_min, dtype=dtype))
    y_max = torch.log(torch.tensor(problem.x_max, dtype=dtype))
    y = y_min + (y_max - y_min) * unit[:, :1]
    near_count = round(count * near_strike_fraction)
    if near_count:
        near_x = 1 + near_strike_half_width * (2 * unit[-near_count:, :1] - 1)
        y[-near_count:] = torch.log(near_x.clamp(problem.x_min, problem.x_max))
    tau = minimum_tau + (problem.T - minimum_tau) * unit[:, 1:]**2
    return y.requires_grad_(True), tau.requires_grad_(True)


def sample_terminal(problem, count, seed, near_strike_fraction=0.25,
                    near_strike_half_width=0.3, dtype=torch.float64):
    engine = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed)
    unit = engine.draw(count).to(dtype=dtype)
    y_min = torch.log(torch.tensor(problem.x_min, dtype=dtype))
    y_max = torch.log(torch.tensor(problem.x_max, dtype=dtype))
    y = y_min + (y_max - y_min) * unit
    near_count = round(count * near_strike_fraction)
    if near_count:
        near_x = 1 + near_strike_half_width * (2 * unit[-near_count:] - 1)
        y[-near_count:] = torch.log(near_x.clamp(problem.x_min, problem.x_max))
    tau = torch.zeros_like(y)
    payoff = torch.relu(torch.exp(y) - 1)
    return y, tau, payoff


def load_fixed_pinn(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    problem = FixedBlackScholesProblem(**checkpoint["problem"])
    model = FixedBlackScholesPINN(problem, **checkpoint["network"])
    dtype = next(iter(checkpoint["state_dict"].values())).dtype
    model.to(dtype=dtype)
    model.load_state_dict(checkpoint["state_dict"])
    return model
