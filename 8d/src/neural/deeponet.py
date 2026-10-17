"""Deep operator network (DeepONet) pricer for basket options on d stocks.

The operator maps the contract's model parameters to its whole pricing function.
The branch net encodes the parameters, the trunk net encodes where the function is
evaluated, and the output is their inner product (Lu et al., 2021):

    N = <branch(parameters), trunk(coordinates)> / sqrt(rank) + bias.

Raw layout: branch on (r, q, sigma, rho, the first d - 1 weights) and the put flag,
trunk on (S_1/K, ..., S_d/K, tau). Reduced layout (see ``networks``): branch on
(sigma, rho), the only parameters left after the exact reduction, trunk on
(z, s_1, ..., s_{d-1}, tau) or with sqrt(tau), optionally with the boundary-layer
coordinate tanh(z / (3 sigma_ref sqrt(tau))) of a fixed volatility so that the trunk
still sees coordinates only. Training uses reference labels, optionally their
derivatives and the PDE residual (``training``).
"""

import math

import torch
from torch import nn

from .networks import NeuralPricer, mlp, raw_inputs


class DeepONet(NeuralPricer):
    family = "deeponet"

    def __init__(self, d, layout, output, boxes, width=64, depth=4, rank=64, trunk_width=None, trunk_depth=None,
                 similarity=None):
        if similarity == "levy":
            raise ValueError("a DeepONet trunk takes coordinates only: give a fixed similarity volatility")
        super().__init__(d, layout, output, boxes, similarity)
        self.settings = dict(d=d, layout=layout, output=output, boxes=boxes, width=width, depth=depth, rank=rank,
                             trunk_width=trunk_width, trunk_depth=trunk_depth, similarity=similarity)
        stocks = self.stocks
        if layout == "raw":
            trunk = tuple(f"spot_{i}" for i in range(1, d+1))+("tau",)
            branch = tuple(n for n in raw_inputs(stocks) if n not in trunk)
        else:
            trunk = stocks.features[:stocks.tau+1]
            branch = stocks.features[stocks.tau+1:]
        if similarity is not None:
            trunk = trunk+("similarity",)
        self.register_buffer("trunk_index", torch.tensor([self.inputs.index(n) for n in trunk]))
        self.register_buffer("branch_index", torch.tensor([self.inputs.index(n) for n in branch]))
        flag = 1 if layout == "raw" else 0      # raw models take the put flag in the branch
        self.branch = mlp(len(branch)+flag, width, depth, rank)
        self.trunk = mlp(len(trunk), trunk_width or width, trunk_depth or depth, rank)
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.rank = rank

    def network(self, inputs, flag):
        branch = inputs[:, self.branch_index]
        if flag is not None:
            branch = torch.cat((branch, flag[:, None]), dim=1)
        product = self.branch(branch)*self.trunk(inputs[:, self.trunk_index])
        return product.sum(1)/math.sqrt(self.rank)+self.bias
