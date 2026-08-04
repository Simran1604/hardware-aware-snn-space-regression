import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch

from snn_space.encoders.direct import DirectEncoder

encoder = DirectEncoder(simulation_steps=6)

image = torch.randn(
    2,
    3,
    256,
    256,
)

encoded = encoder(image)

print(encoded.shape)

print(torch.equal(encoded[0], encoded[5]))