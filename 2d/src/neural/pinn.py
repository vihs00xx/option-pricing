"""Physics-informed neural network (PINN) pricer for two-stock basket options.

One tanh MLP maps the scaled inputs to the price (raw layout) or to the normalized
call or its correction to the Levy approximation (reduced layout; see
``networks``). Training minimizes the residual of the pricing PDE (``pde``) at
collocation points, with the payoff imposed softly or exactly by the output and the
far-boundary condition imposed softly (``training``).
"""

import torch

from .networks import NeuralPricer, mlp


class PINN(NeuralPricer):
    family = "pinn"

    def __init__(self, layout, output, boxes, width=64, depth=4, similarity=None):
        super().__init__(layout, output, boxes, similarity)
        self.settings = dict(layout=layout, output=output, boxes=boxes, width=width, depth=depth,
                             similarity=similarity)
        flag = 1 if layout == "raw" else 0      # raw models take the put flag as an extra input
        self.mlp = mlp(len(self.inputs)+flag, width, depth, 1)

    def network(self, inputs, flag):
        if flag is not None:
            inputs = torch.cat((inputs, flag[:, None]), dim=1)
        return self.mlp(inputs).squeeze(-1)
