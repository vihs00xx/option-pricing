"""Deep operator network (DeepONet) pricer for one-stock Heston options.

The branch net encodes the model parameters, the trunk net the point where the pricing
function is evaluated, and the output is their inner product (Lu et al., 2021):

    N = <branch(parameters), trunk(coordinates)> / sqrt(rank) + bias.

Raw layout: branch on (r, q, kappa, theta, xi, rho) and the put flag, trunk on
(S/K, v0, tau). Reduced layout: branch on (kappa, theta, xi, rho), trunk on (x, v, tau)
or (x, v, sqrt(tau)) with the Black-Scholes base output. With ``drift_split`` the output
is <vol_branch(xi, rho), trunk_1> + (tau / tau_max) <branch(kappa, theta, xi, rho), trunk_2>:
kappa and theta act only through the variance drift, so their part vanishes as tau -> 0.
"""

import math

import torch
from torch import nn

from .networks import NeuralPricer, mlp

RAW_TRUNK = ("S", "v0", "tau")
RAW_BRANCH = ("r", "q", "kappa", "theta", "xi", "rho")
REDUCED_TRUNK = ("x", "v", "tau")
REDUCED_BRANCH = ("kappa", "theta", "xi", "rho")


class DeepONet(NeuralPricer):
    family = "deeponet"

    def __init__(self, layout, output, boxes, width=64, depth=4, rank=64, trunk_width=None, trunk_depth=None,
                 v_input="v", drift_split=False):
        super().__init__(layout, output, boxes, v_input)
        self.settings = dict(layout=layout, output=output, boxes=boxes, width=width, depth=depth, rank=rank,
                             trunk_width=trunk_width, trunk_depth=trunk_depth, v_input=v_input,
                             drift_split=drift_split)
        if drift_split and (layout == "raw" or output == "plain"):
            raise ValueError("the drift split needs the reduced layout with a Black-Scholes base")
        self.drift_split = drift_split
        trunk, branch = (RAW_TRUNK, RAW_BRANCH) if layout == "raw" else (REDUCED_TRUNK, REDUCED_BRANCH)
        self.register_buffer("trunk_index", torch.tensor([self.inputs.index(n) for n in trunk]))
        self.register_buffer("branch_index", torch.tensor([self.inputs.index(n) for n in branch]))
        flag = 1 if layout == "raw" else 0
        self.branch = mlp(len(branch) + flag, width, depth, rank)
        self.trunk = mlp(len(trunk), trunk_width or width, trunk_depth or depth, 2 * rank if drift_split else rank)
        if drift_split:
            # branch on (xi, rho) only; the full branch's terms are multiplied by tau / tau_max
            self.register_buffer("vol_index", torch.tensor([self.inputs.index(n) for n in ("xi", "rho")]))
            self.vol_branch = mlp(2, width, depth, rank)
            self.tau_max = float(boxes["tau"][1])
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.rank = rank

    def network(self, inputs, flag):
        branch = inputs[:, self.branch_index]
        if flag is not None:
            branch = torch.cat((branch, flag[:, None]), dim=1)
        trunk = self.trunk(inputs[:, self.trunk_index])
        if not self.drift_split:
            return (self.branch(branch) * trunk).sum(1) / math.sqrt(self.rank) + self.bias
        # undo the scaling of sqrt(tau) (index 2) to get tau
        root = self.scaler.center[2] + self.scaler.radius[2] * inputs[:, 2]
        tau = root * root / self.tau_max
        r = self.rank
        vol = (self.vol_branch(inputs[:, self.vol_index]) * trunk[:, :r]).sum(1)
        full = (self.branch(branch) * trunk[:, r:]).sum(1)
        return (vol + tau * full) / math.sqrt(r) + self.bias
