"""Physics-informed neural network (PINN) pricer for one-stock Heston options.

One tanh MLP maps the scaled inputs to the price (raw layout) or to the normalized call
or its correction to the Black-Scholes base (reduced layout; see ``networks``). Training
minimizes the Heston PDE residual (``pde``) at collocation points (``training``).
"""

import torch

from .networks import NeuralPricer, mlp


class PINN(NeuralPricer):
    family = "pinn"

    def __init__(self, layout, output, boxes, width=64, depth=4, v_input="v", drift_inputs="plain"):
        super().__init__(layout, output, boxes, v_input, drift_inputs)
        self.settings = dict(layout=layout, output=output, boxes=boxes, width=width, depth=depth, v_input=v_input,
                             drift_inputs=drift_inputs)
        flag = 1 if layout == "raw" else 0
        self.mlp = mlp(len(self.inputs) + flag, width, depth, 1)

    def network(self, inputs, flag):
        if flag is not None:
            inputs = torch.cat((inputs, flag[:, None]), dim=1)
        return self.mlp(inputs).squeeze(-1)
