from dataclasses import dataclass


@dataclass
class EMACConstants:
    """
    Section 3.4.5
    """

    SYNAPTIC_COST = 0.667
    UPDATE_COST = 3.333

    CNN_MAC_COST = 1.0


class HardwareAgnosticEnergy:

    def __init__(
        self,
        synaptic_operations,
        neurons,
        simulation_steps,
    ):

        self.s = synaptic_operations
        self.n = neurons
        self.T = simulation_steps

    def snn_energy(self):

        return (
            self.s * EMACConstants.SYNAPTIC_COST
            + self.n
            * self.T
            * EMACConstants.UPDATE_COST
        )

    def cnn_energy(self):

        return (
            self.s
            * EMACConstants.CNN_MAC_COST
        )

    def relative_energy(self):

        return (
            self.snn_energy()
            / self.cnn_energy()
        )