import torch
import torch.nn as nn


class DirectEncoder(nn.Module):
    """
    Direct encoding from the paper.

    The same image is presented at every timestep.
    """

    def __init__(self, simulation_steps: int):
        super().__init__()
        self.T = simulation_steps

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Shape: (B, C, H, W)

        Returns
        -------
        Tensor
            Shape: (T, B, C, H, W)
        """

        return x.unsqueeze(0).repeat(
            self.T,
            1,
            1,
            1,
            1,
        )