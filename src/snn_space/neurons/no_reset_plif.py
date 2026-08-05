import torch
import torch.nn as nn


class NoResetPLIFNeuron(nn.Module):

    def __init__(
        self,
        init_tau=2.0,
    ):
        super().__init__()

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

        return membrane