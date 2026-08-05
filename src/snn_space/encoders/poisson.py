import torch
import torch.nn as nn


class PoissonEncoder(nn.Module):

    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps

    def forward(self, x):

        B, C, H, W = x.shape

        spikes = torch.rand(
            B,
            self.num_steps,
            C,
            H,
            W,
            device=x.device,
        )

        return (spikes < x.unsqueeze(1)).float()