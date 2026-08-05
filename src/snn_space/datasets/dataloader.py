from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )