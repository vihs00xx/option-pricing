"""Deep operator network (DeepONet) pricer for two-stock basket options.

The operator maps the contract's model parameters to its whole pricing function.
The branch net encodes the parameters, the trunk net encodes where the function is
evaluated, and the output is their inner product (Lu et al., 2021):

    N = <branch(parameters), trunk(coordinates)> / sqrt(rank) + bias.

Raw layout: branch on (r, q_1, q_2, sigma_1, sigma_2, rho, w_1) and the put flag,
trunk on (S_1/K, S_2/K, tau). Reduced layout (see ``networks``): branch on
(sigma_1, sigma_2, rho), the only parameters left after the exact reduction, trunk on
(z, s, tau) or (z, s, sqrt(tau)), optionally with the boundary-layer coordinate
tanh(z / (3 sigma_ref sqrt(tau))) of a fixed volatility so that the trunk still sees
coordinates only. Training uses reference labels, optionally their derivatives and the
PDE residual (``training``).
"""

import math

import torch
from torch import nn

from .networks import NeuralPricer, mlp

RAW_TRUNK = ("spot_1", "spot_2", "tau")
RAW_BRANCH = ("rate", "dividend_1", "dividend_2", "vol_1", "vol_2", "correlation", "weight_1")
REDUCED_TRUNK = ("z", "s", "tau")
REDUCED_BRANCH = ("sigma_1", "sigma_2", "rho")


class DeepONet(NeuralPricer):
    family = "deeponet"

    def __init__(self, layout, output, boxes, width=64, depth=4, rank=64, trunk_width=None, trunk_depth=None,
                 similarity=None):
        if similarity == "levy":
            raise ValueError("a DeepONet trunk takes coordinates only: give a fixed similarity volatility")
        super().__init__(layout, output, boxes, similarity)
        self.settings = dict(layout=layout, output=output, boxes=boxes, width=width, depth=depth, rank=rank,
                             trunk_width=trunk_width, trunk_depth=trunk_depth, similarity=similarity)
        trunk, branch = (RAW_TRUNK, RAW_BRANCH) if layout == "raw" else (REDUCED_TRUNK, REDUCED_BRANCH)
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

