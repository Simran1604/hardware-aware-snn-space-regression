import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from snn_space.energy.hardware_agnostic import HardwareAgnosticEnergy

energy = HardwareAgnosticEnergy(
    synaptic_operations=20_000_000,
    neurons=250_000,
    simulation_steps=8,
)

print("CNN:", energy.cnn_energy())
print("SNN:", energy.snn_energy())
print("Ratio:", energy.relative_energy())