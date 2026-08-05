import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch

from snn_space.neurons.plif import PLIFNeuron

torch.manual_seed(0)

neuron = PLIFNeuron()

membrane = torch.zeros(1)

inputs = [0.3, 0.4, 0.5, 0.2, 0.6]

for t, value in enumerate(inputs):

    current = torch.tensor([value])

    spike, membrane = neuron(current, membrane)

    print(
        f"{t:2d} | "
        f"input={value:.2f} | "
        f"mem={membrane.item():.4f} | "
        f"spike={spike.item()}"
    )