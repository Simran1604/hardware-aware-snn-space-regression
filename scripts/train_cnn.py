import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from snn_space.datasets.position_dataset import PositionEstimationDataset
from snn_space.models.cnn import CNNBaseline
from snn_space.training.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PositionEstimationDataset(
    image_dir="data/raw/train_val",
    csv_file="data/raw/train_val.csv"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
)

model = CNNBaseline().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

criterion = nn.MSELoss()

trainer = Trainer(
    model,
    loader,
    optimizer,
    criterion,
    device,
)

loss = trainer.train_epoch()

print(loss)