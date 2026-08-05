from pathlib import Path

import torch
from torch.utils.data import random_split, DataLoader

from snn_space.datasets.position_dataset import PositionEstimationDataset
from snn_space.encoders.direct import DirectEncoder
from snn_space.models.snn import SNN

from snn_space.training.config import TrainingConfig
from snn_space.training.loss import get_loss
from snn_space.training.optimizer import get_optimizer
from snn_space.training.early_stopping import EarlyStopping
from snn_space.training.trainer import Trainer


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

IMAGE_DIR = "data/images"
CSV_FILE = "data/position_estimation_dataset.csv"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

config = TrainingConfig()


# ----------------------------------------------------
# Dataset
# ----------------------------------------------------

dataset = PositionEstimationDataset(
    image_dir=IMAGE_DIR,
    csv_file=CSV_FILE,
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# ----------------------------------------------------
# Model
# ----------------------------------------------------

encoder = DirectEncoder(
    num_steps=8,
)

model = SNN(
    encoder=encoder,
    simulation_steps=8,
).to(device)

print(model)


# ----------------------------------------------------
# Training
# ----------------------------------------------------

criterion = get_loss()

optimizer = get_optimizer(
    model,
    config.learning_rate,
)

early_stopping = EarlyStopping(
    patience=config.patience,
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    early_stopping=early_stopping,
)


# ----------------------------------------------------
# Train
# ----------------------------------------------------

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=config.epochs,
)

print("Training finished.")