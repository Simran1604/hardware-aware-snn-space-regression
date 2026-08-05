import torch.optim as optim


def get_optimizer(model, lr):

    return optim.Adam(
        model.parameters(),
        lr=lr,
    )