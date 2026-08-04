import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch

from snn_space.neurons.plif import PLIFNeuron

neuron = PLIFNeuron()

membrane = torch.zeros(1)

inputs = [
    0.3,
    0.4,
    0.5,
    0.2,
    0.6,
]

for t, value in enumerate(inputs):

    current = torch.tensor([value])

    previous_spike = torch.zeros(1)

for t, value in enumerate(inputs):

    current = torch.tensor([value])

    spike, membrane = neuron(
        current,
        membrane,
        previous_spike,
    )

    previous_spike = spike

    print(
        t,
        membrane.item(),
        spike.item()
    )

    print(
        f"t={t} "
        f"input={value:.2f} "
        f"membrane={membrane.item():.3f} "
        f"spike={spike.item()}"
    )