import pandas as pd
from pathlib import Path

import torch

from PIL import Image

from torch.utils.data import Dataset

class PositionEstimationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        csv_path,
        transform=None,
    ):