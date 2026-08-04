import torch
import torch.nn as nn

from snn_space.neurons.surrogate import surrogate_spike


class PLIFNeuron(nn.Module):

    def __init__(self, threshold=1.0, beta_init=0.9):
        super().__init__()

        self.beta = nn.Parameter(torch.tensor(beta_init))

        self.register_buffer(
            "threshold",
            torch.tensor(float(threshold))
        )

    def forward(
        self,
        input_current,
        membrane,
        previous_spike,
    ):

        beta = torch.sigmoid(self.beta)

        membrane = (
            beta * membrane
            + input_current
            - self.threshold * previous_spike
        )

        spike = surrogate_spike(
            membrane,
            self.threshold,
        )

        return spike, membrane