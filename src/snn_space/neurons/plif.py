import torch
import torch.nn as nn

from snn_space.neurons.surrogate import spike_fn


class PLIFNeuron(nn.Module):

    def __init__(
        self,
        threshold=1.0,
        init_tau=2.0,
    ):
        super().__init__()

        self.threshold = threshold

        self.w = nn.Parameter(
            torch.tensor(init_tau)
        )

    def forward(
        self,
        input_current,
        membrane,
    ):

        beta = torch.sigmoid(self.w)

        membrane = beta * membrane + input_current

        spikes = spike_fn(
            membrane - self.threshold
        )

        membrane = membrane - spikes * self.threshold

        return spikes, membrane