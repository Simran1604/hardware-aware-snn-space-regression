from torch.utils.data import random_split


def split_dataset(
    dataset,
    train_ratio=0.8,
):

    train_size = int(
        train_ratio * len(dataset)
    )

    val_size = len(dataset) - train_size

    return random_split(
        dataset,
        [train_size, val_size],
    )