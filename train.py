import torch

from torch.utils.data import DataLoader

from snn_space.training.config import TrainingConfig
from snn_space.training.loss import get_loss
from snn_space.training.optimizer import get_optimizer
from snn_space.training.early_stopping import EarlyStopping
from snn_space.training.trainer import Trainer

# TODO:
# import PositionDataset
# import SNN

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

config = TrainingConfig()

# train_dataset = ...
# val_dataset = ...

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
)

model = SNN().to(device)

optimizer = get_optimizer(
    model,
    config.learning_rate,
)

criterion = get_loss()

trainer = Trainer(
    model,
    optimizer,
    criterion,
    device,
    EarlyStopping(
        config.patience,
    ),
)

trainer.fit(
    train_loader,
    val_loader,
    config.epochs,
)