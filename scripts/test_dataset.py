import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from snn_space.datasets.position_dataset import PositionEstimationDataset

dataset = PositionEstimationDataset(
    image_dir="data/raw/train_val",
    csv_file="data/raw/train_val.csv"
)

print("Dataset size:", len(dataset))

image, target = dataset[0]

print("Image shape:", image.shape)
print("Target:", target)