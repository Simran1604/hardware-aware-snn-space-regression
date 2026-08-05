import torch


def euclidean_position_error(predictions, targets):
    """
    Euclidean distance between predicted and
    ground-truth spacecraft position.

    Parameters
    ----------
    predictions : Tensor (N,3)
    targets : Tensor (N,3)

    Returns
    -------
    Mean Euclidean distance.
    """

    distance = torch.norm(
        predictions - targets,
        dim=1,
    )

    return distance.mean()