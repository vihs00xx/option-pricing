import math

import torch
from torch import nn


def _network(inputs, width, depth, rank):
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(inputs, width, dtype=torch.float64), nn.Tanh()])
        inputs = width
    layers.append(nn.Linear(inputs, rank, dtype=torch.float64))
    network = nn.Sequential(*layers)
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return network


class BlackScholesDeepONet(nn.Module):
    """Parametric branch/trunk surrogate; output is price in currency units."""

    def __init__(self, width=64, depth=3, rank=64):
        super().__init__()
        self.architecture = dict(width=width, depth=depth, rank=rank)
        self.branch = _network(6, width, depth, rank)
        self.trunk = _network(2, width, depth, rank)
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.register_buffer("branch_center", torch.tensor(
            [100., 1.025, .04, .03, .35, 0.], dtype=torch.float64))
        self.register_buffer("branch_scale", torch.tensor(
            [20., .975, .06, .03, .25, 1.], dtype=torch.float64))

    def forward(self, contracts, coordinates):
        branch_input = (contracts - self.branch_center) / self.branch_scale
        strike = contracts[:, 0]
        trunk_input = torch.stack((
            (coordinates[:, 0] / strike - 1.) / .3,
            coordinates[:, 1] - 1.,
        ), dim=1)
        normalized_price = (self.branch(branch_input) * self.trunk(trunk_input)).sum(1)
        normalized_price = normalized_price / math.sqrt(self.architecture["rank"]) + self.bias
        return strike * normalized_price

    def price_and_greeks(self, contracts, coordinates, create_graph=False):
        with torch.enable_grad():
            contracts = contracts.detach().clone().requires_grad_(True)
            coordinates = coordinates.detach().clone().requires_grad_(True)
            price = self(contracts, coordinates)
            branch_grad, trunk_grad = torch.autograd.grad(
                price.sum(), (contracts, coordinates), create_graph=True)
            gamma = torch.autograd.grad(trunk_grad[:, 0].sum(), coordinates,
                                        create_graph=create_graph, retain_graph=True)[0][:, 0]
            values = {
                "price": price, "delta": trunk_grad[:, 0], "gamma": gamma,
                "vega": branch_grad[:, 4], "theta": -trunk_grad[:, 1],
                "rho": branch_grad[:, 2],
            }
            return values if create_graph else {name: value.detach() for name, value in values.items()}


def load_deeponet(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BlackScholesDeepONet(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
