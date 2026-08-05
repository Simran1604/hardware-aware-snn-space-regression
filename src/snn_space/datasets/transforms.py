from torchvision import transforms


def get_transforms(train=True):
    """
    Transforms used for the published experiments.
    The paper specifies normalized RGB images resized to 256×256.
    No augmentation is mentioned.
    """

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])