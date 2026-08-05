import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch

from snn_space.neurons.no_reset_plif import NoResetPLIFNeuron

neuron = NoResetPLIFNeuron()

membrane = torch.zeros(1)

inputs = [0.3, 0.4, 0.5, 0.2, 0.6]

for t, value in enumerate(inputs):

    membrane = neuron(
        torch.tensor([value]),
        membrane
    )

    print(
        t,
        membrane.item()
    )