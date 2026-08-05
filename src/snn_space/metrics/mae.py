import torch


def mean_absolute_error(predictions, targets):
    """
    Mean Absolute Error over x, y, z coordinates.

    Parameters
    ----------
    predictions : Tensor (N, 3)
    targets : Tensor (N, 3)
    """

    return torch.mean(torch.abs(predictions - targets))