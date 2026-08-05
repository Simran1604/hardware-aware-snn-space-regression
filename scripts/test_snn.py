import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch

from snn_space.models.snn import SNN
from snn_space.encoders.direct import DirectEncoder


encoder = DirectEncoder(
    num_steps=10,
)

model = SNN(
    encoder=encoder,
    simulation_steps=10,
)

dummy = torch.rand(
    1,
    3,
    256,
    256,
)

output = model(dummy)

print(output.shape)

print(output)