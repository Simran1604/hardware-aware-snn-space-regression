import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch

from snn_space.models.snn import SimpleSNN

model = SimpleSNN(
    simulation_steps=10
)

dummy = torch.rand(
    1,
    3,
    256,
    256,
)

output = model(dummy)

print(output.shape)

print(output.min())

print(output.max())

print(output.mean())