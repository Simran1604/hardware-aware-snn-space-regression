import torch

from snn_space.metrics.mae import mean_absolute_error
from snn_space.metrics.euclidean import euclidean_position_error


class Evaluator:

    def __init__(self, model, device):

        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader, criterion):

        self.model.eval()

        total_loss = 0.0
        total_mae = 0.0
        total_distance = 0.0

        num_batches = 0

        for images, targets in loader:

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)

            loss = criterion(outputs, targets)

            mae = mean_absolute_error(
                outputs,
                targets,
            )

            distance = euclidean_position_error(
                outputs,
                targets,
            )

            total_loss += loss.item()
            total_mae += mae.item()
            total_distance += distance.item()

            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "mae": total_mae / num_batches,
            "distance": total_distance / num_batches,
        }