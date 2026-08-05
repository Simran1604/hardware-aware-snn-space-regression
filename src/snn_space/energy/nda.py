from dataclasses import dataclass

from .constants import NDAConstants


@dataclass
class LayerStatistics:
    """
    Statistics required by the NDA model.
    """

    fan_in: int

    neurons: int

    simulation_steps: int

    input_sparsity: float

    output_sparsity: float

    hops: int = NDAConstants.DEFAULT_HOPS


class NDAEnergyEstimator:

    def __init__(self):

        self.c = NDAConstants()

    def neuron_energy(
        self,
        stats: LayerStatistics,
    ):

        rin = 1.0 - stats.input_sparsity

        rout = 1.0 - stats.output_sparsity

        return stats.simulation_steps * (

            stats.fan_in * rin *
            (
                self.c.E_R_WEIGHT +
                self.c.E_ADD
            )

            +

            self.c.E_R_STATE

            +

            self.c.E_ADD

            +

            self.c.E_CMP

            +

            self.c.E_R_LEAK

            +

            self.c.E_MUL

            +

            rout * self.c.E_SUB

            +

            self.c.E_W_STATE

            +

            stats.fan_in *
            rout *
            stats.hops *
            self.c.E_TP_HOP
        )

    def layer_energy(
        self,
        stats: LayerStatistics,
    ):

        return (
            stats.neurons *
            self.neuron_energy(stats)
        )

    def network_energy(
        self,
        layers,
    ):

        total = 0.0

        for layer in layers:

            total += self.layer_energy(layer)

        return total