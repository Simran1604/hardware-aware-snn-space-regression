import torch
import torch.nn as nn


class LatencyEncoder(nn.Module):

    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps

    def forward(self, x):

        B, C, H, W = x.shape

        spikes = torch.zeros(
            B,
            self.num_steps,
            C,
            H,
            W,
            device=x.device,
        )

        fire_time = (
            (self.num_steps - 1) * (1 - x)
        ).long()

        for t in range(self.num_steps):
            spikes[:, t] = (fire_time == t).float()

        return spikes