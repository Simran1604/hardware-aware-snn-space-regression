import copy
import os
from torch.utils.tensorboard import SummaryWriter
import torch

from snn_space.training.evaluator import Evaluator


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        experiment_name="experiment",
        early_stopping=None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.experiment_name = experiment_name

        self.early_stopping = early_stopping

        self.best_model = None
        self.best_loss = float("inf")

        self.evaluator = Evaluator(
            model=model,
            device=device,
        )

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=f"runs/{self.experiment_name}"
        )

    def train_epoch(
        self,
        loader,
    ):

        self.model.train()

        running_loss = 0.0

        for images, targets in loader:

            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(images)

            loss = self.criterion(
                predictions,
                targets,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)

    @torch.no_grad()
    def validate(
        self,
        loader,
    ):

        self.model.eval()

        running_loss = 0.0

        for images, targets in loader:

            images = images.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(images)

            loss = self.criterion(
                predictions,
                targets,
            )

            running_loss += loss.item()

        return running_loss / len(loader)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
    ):

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "mae": [],
            "distance": [],
        }

        for epoch in range(epochs):

            train_loss = self.train_epoch(
                train_loader
            )

            metrics = self.evaluator.evaluate(
                val_loader,
                self.criterion,
            )

            val_loss = metrics["loss"]
            val_mae = metrics["mae"]
            val_distance = metrics["distance"]

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["mae"].append(val_mae)
            history["distance"].append(val_distance)

            self.writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch + 1,
        )

            self.writer.add_scalar(
                "Loss/Validation",
                val_loss,
                epoch + 1,
            )

            self.writer.add_scalar(
                "Metrics/MAE",
                val_mae,
                epoch + 1,
            )

            self.writer.add_scalar(
                "Metrics/Distance",
                val_distance,
                epoch + 1,
            )

            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train {train_loss:.6f} | "
                f"Val {val_loss:.6f} | "
                f"MAE {val_mae:.6f} | "
                f"Dist {val_distance:.6f}"
            )

            if val_loss < self.best_loss:

                self.best_loss = val_loss

                self.best_model = copy.deepcopy(
                    self.model.state_dict()
                )

                torch.save(
                    self.best_model,
                    f"checkpoints/{self.experiment_name}_best.pt",
                )

            if (
                self.early_stopping is not None
                and self.early_stopping.step(val_loss)
            ):

                print("Early stopping triggered.")
                break

        if self.best_model is not None:

            self.model.load_state_dict(
                self.best_model
            )

        self.writer.close()
        return history