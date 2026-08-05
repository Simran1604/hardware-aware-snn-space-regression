import argparse
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from snn_space.datasets.position_dataset import PositionEstimationDataset

from snn_space.encoders.direct import DirectEncoder
from snn_space.encoders.latency import LatencyEncoder
from snn_space.encoders.poisson import PoissonEncoder

from snn_space.models.snn import SNN

from snn_space.training.config import TrainingConfig
from snn_space.training.loss import get_loss
from snn_space.training.optimizer import get_optimizer
from snn_space.training.early_stopping import EarlyStopping
from snn_space.training.trainer import Trainer


# --------------------------------------------------
# Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--encoder",
    default="direct",
    choices=["direct", "latency", "poisson"],
)

parser.add_argument(
    "--timesteps",
    type=int,
    default=8,
)

parser.add_argument(
    "--epochs",
    type=int,
    default=200,
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
)

args = parser.parse_args()

# --------------------------------------------------

IMAGE_DIR = "data/raw/train_val"
CSV_FILE = "data/raw/train_val.csv"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

config = TrainingConfig()

config.learning_rate = args.lr
config.epochs = args.epochs

print("Loading dataset...")

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
)

print("Creating model...")

if args.encoder == "direct":
    encoder = DirectEncoder(args.timesteps)

elif args.encoder == "latency":
    encoder = LatencyEncoder(args.timesteps)

else:
    encoder = PoissonEncoder(args.timesteps)

experiment_name = f"{args.encoder}_t{args.timesteps}"

model = SNN(
    encoder=encoder,
    simulation_steps=args.timesteps,
).to(device)

criterion = get_loss()

optimizer = get_optimizer(
    model,
    config.learning_rate,
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    experiment_name=experiment_name,
    early_stopping=EarlyStopping(config.patience),
)

print("Starting training...")

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=config.epochs,
)

os.makedirs("results", exist_ok=True)

results_path = f"results/{experiment_name}.csv"

pd.DataFrame(history).to_csv(
    results_path,
    index=False,
)

print(f"Saved history -> {results_path}")

summary_path = "results/summary.csv"

best_epoch = min(
    range(len(history["val_loss"])),
    key=lambda i: history["val_loss"][i],
)

summary = pd.DataFrame([{
    "encoder": args.encoder,
    "timesteps": args.timesteps,
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "best_epoch": history["epoch"][best_epoch],
    "best_train_loss": history["train_loss"][best_epoch],
    "best_val_loss": history["val_loss"][best_epoch],
    "best_mae": history["mae"][best_epoch],
    "best_distance": history["distance"][best_epoch],
}])

if os.path.exists(summary_path):

    previous = pd.read_csv(summary_path)

    summary = pd.concat(
        [previous, summary],
        ignore_index=True,
    )

summary.to_csv(
    summary_path,
    index=False,
)

print(summary.tail())

print("Finished.")