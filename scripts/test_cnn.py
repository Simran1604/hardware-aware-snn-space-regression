import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch

from snn_space.models.cnn import CNNBaseline

model = CNNBaseline()

print(model)

dummy = torch.randn(1, 3, 256, 256)

output = model(dummy)

print("Output shape:", output.shape)

print(output)

total_params = sum(p.numel() for p in model.parameters())

print(f"Parameters: {total_params:,}")