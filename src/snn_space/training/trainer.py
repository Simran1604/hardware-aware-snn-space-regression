import torch
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        criterion,
        device,
    ):

        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self):

        self.model.train()

        total_loss = 0

        for images, targets in tqdm(self.train_loader):

            images = images.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(images)

            loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)