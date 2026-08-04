import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch

from snn_space.neurons.surrogate import surrogate_spike


v = torch.tensor(
    [-1.0, 0.5, 1.0, 1.5],
    requires_grad=True,
)

threshold = torch.tensor(1.0)

spikes = surrogate_spike(v, threshold)

print(spikes)

loss = spikes.sum()

loss.backward()

print(v.grad)