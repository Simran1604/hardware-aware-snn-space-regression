import torch
import torch.nn as nn


class DirectEncoder(nn.Module):

    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps

    def forward(self, x):
        """
        x : [B,3,H,W]

        returns

        [B,T,3,H,W]
        """

        return x.unsqueeze(1).repeat(
            1,
            self.num_steps,
            1,
            1,
            1,
        )