import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from snn_space.energy.nda import (
    NDAEnergyEstimator,
    LayerStatistics,
)

layer = LayerStatistics(

    fan_in=144,

    neurons=1024,

    simulation_steps=8,

    input_sparsity=0.82,

    output_sparsity=0.91,
)

estimator = NDAEnergyEstimator()

print("Neuron Energy (pJ)")
print(estimator.neuron_energy(layer))

print()

print("Layer Energy (pJ)")
print(estimator.layer_energy(layer))