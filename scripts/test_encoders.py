import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

import torch

from snn_space.encoders.direct import DirectEncoder
from snn_space.encoders.latency import LatencyEncoder
from snn_space.encoders.poisson import PoissonEncoder

x = torch.rand(2,3,256,256)

for Encoder in [
    DirectEncoder,
    LatencyEncoder,
    PoissonEncoder,
]:

    encoder = Encoder(8)

    y = encoder(x)

    print(
        Encoder.__name__,
        y.shape,
    )