import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch

from snn_space.metrics.mae import mean_absolute_error
from snn_space.metrics.euclidean import euclidean_position_error

pred = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 2.0, 2.0]
])

gt = torch.tensor([
    [1.0, 2.0, 4.0],
    [1.0, 2.0, 3.0]
])

print("MAE:", mean_absolute_error(pred, gt))
print("Distance:", euclidean_position_error(pred, gt))