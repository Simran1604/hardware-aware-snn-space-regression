import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from snn_space.datasets.position_dataset import PositionEstimationDataset

dataset = PositionEstimationDataset(
    image_dir="data/raw/train_val",
    csv_file="data/raw/train_val.csv",
)

image, target = dataset[0]

plt.figure(figsize=(6, 6))
plt.imshow(image.permute(1, 2, 0))
plt.title(
    f"x={target[0]:.4f}\n"
    f"y={target[1]:.4f}\n"
    f"z={target[2]:.4f}"
)
plt.axis("off")
plt.show()