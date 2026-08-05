from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PositionEstimationDataset(Dataset):
    """
    Dataset for satellite position estimation.
    """

    def __init__(self, image_dir, csv_file, transform=None):

        self.image_dir = Path(image_dir)

        # FIRST load the CSV
        self.data = pd.read_csv(csv_file, sep=";")

        if "Unnamed: 0.1" in self.data.columns:
            self.data = self.data.drop(columns=["Unnamed: 0.1"])

        valid_rows = []

        for _, row in self.data.iterrows():

            image_id = int(row["Unnamed: 0"])
            image_path = self.image_dir / f"image{image_id}.png"

            if not image_path.exists():
                continue

            try:
                with Image.open(image_path) as img:
                    img.verify()      # Verify that the PNG is not corrupted

                valid_rows.append(row)

            except Exception:
                continue

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

        print(f"Using {len(self.data)} valid images.")

        print(f"Using {len(self.data)} valid samples.")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # self.image_dir = Path(image_dir)

        # self.data = pd.read_csv(csv_file, sep=";")

        # # Remove accidental index column if present
        # if "Unnamed: 0.1" in self.data.columns:
        #     self.data = self.data.drop(columns=["Unnamed: 0.1"])

        # if transform is None:
        #     self.transform = transforms.Compose([
        #         transforms.Resize((256, 256)),
        #         transforms.ToTensor(),
        #     ])
        # else:
        #     self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_id = int(row["Unnamed: 0"])

        image_path = self.image_dir / f"image{image_id}.png"

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        target = torch.tensor(
            [
                row["x"],
                row["y"],
                row["z"]
            ],
            dtype=torch.float32,
        )

        return image, target