import torch
import torch.nn as nn

from snn_space.neurons.plif import PLIFNeuron
from snn_space.neurons.no_reset_plif import NoResetPLIFNeuron


class SNN(nn.Module):
    """
    Spiking Neural Network for spacecraft position estimation.

    Architecture:

    Input
        ↓
    Encoder
        ↓
    Conv → PLIF → MaxPool
        ↓
    Conv → PLIF → MaxPool
        ↓
    Conv → PLIF → MaxPool
        ↓
    Conv → PLIF → MaxPool
        ↓
    Flatten
        ↓
    FC → PLIF
        ↓
    FC → PLIF
        ↓
    FC → NoResetPLIF
    """

    def __init__(self, encoder, simulation_steps):

        super().__init__()

        self.encoder = encoder
        self.T = simulation_steps

        # ---------------------------------------------------
        # Convolutional Feature Extractor
        # ---------------------------------------------------

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.plif1 = PLIFNeuron()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.plif2 = PLIFNeuron()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.plif3 = PLIFNeuron()
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.plif4 = PLIFNeuron()
        self.pool4 = nn.MaxPool2d(2)

        # ---------------------------------------------------
        # Regression Head
        # ---------------------------------------------------

        self.fc1 = nn.Linear(
            64 * 16 * 16,
            128,
        )

        self.plif5 = PLIFNeuron()

        self.fc2 = nn.Linear(
            128,
            64,
        )

        self.plif6 = PLIFNeuron()

        self.fc3 = nn.Linear(
            64,
            3,
        )

        self.output = NoResetPLIFNeuron()

    @staticmethod
    def _init_membrane(membrane, current):
        """
        Initialize membrane if it does not exist.
        """
        if membrane is None:
            return torch.zeros_like(current)
        return membrane

    def forward(self, images):

        encoded = self.encoder(images)

        # ---------------------------------------------------
        # Safety checks
        # ---------------------------------------------------

        assert encoded.shape[1] == self.T, (
            f"Encoder produced {encoded.shape[1]} timesteps "
            f"but model expects {self.T}."
        )

        # ---------------------------------------------------
        # Membrane States
        # ---------------------------------------------------

        mem1 = None
        mem2 = None
        mem3 = None
        mem4 = None
        mem5 = None
        mem6 = None
        mem_out = None

        # Optional: collect spike statistics
        total_spikes = 0

        # ---------------------------------------------------
        # Time Simulation
        # ---------------------------------------------------

        for t in range(self.T):

            x = encoded[:, t]

            # ==========================
            # Conv Block 1
            # ==========================

            current = self.conv1(x)

            mem1 = self._init_membrane(mem1, current)

            x, mem1 = self.plif1(
                current,
                mem1,
            )

            total_spikes += x.sum()

            x = self.pool1(x)

            # ==========================
            # Conv Block 2
            # ==========================

            current = self.conv2(x)

            mem2 = self._init_membrane(mem2, current)

            x, mem2 = self.plif2(
                current,
                mem2,
            )

            total_spikes += x.sum()

            x = self.pool2(x)

            # ==========================
            # Conv Block 3
            # ==========================

            current = self.conv3(x)

            mem3 = self._init_membrane(mem3, current)

            x, mem3 = self.plif3(
                current,
                mem3,
            )

            total_spikes += x.sum()

            x = self.pool3(x)

            # ==========================
            # Conv Block 4
            # ==========================

            current = self.conv4(x)

            mem4 = self._init_membrane(mem4, current)

            x, mem4 = self.plif4(
                current,
                mem4,
            )

            total_spikes += x.sum()

            x = self.pool4(x)

            # ==========================
            # Flatten
            # ==========================

            x = x.flatten(1)

            expected = self.fc1.in_features

            if x.shape[1] != expected:
                raise RuntimeError(
                    f"Flatten dimension {x.shape[1]} "
                    f"does not match fc1 input "
                    f"{expected}. "
                    f"Check your input image size."
                )

            # ==========================
            # FC1
            # ==========================

            current = self.fc1(x)

            mem5 = self._init_membrane(mem5, current)

            x, mem5 = self.plif5(
                current,
                mem5,
            )

            total_spikes += x.sum()

            # ==========================
            # FC2
            # ==========================

            current = self.fc2(x)

            mem6 = self._init_membrane(mem6, current)

            x, mem6 = self.plif6(
                current,
                mem6,
            )

            total_spikes += x.sum()

            # ==========================
            # Output Layer
            # ==========================

            current = self.fc3(x)

            mem_out = self._init_membrane(
                mem_out,
                current,
            )

            mem_out = self.output(
                current,
                mem_out,
            )

        # ---------------------------------------------------
        # Return final membrane potential
        # ---------------------------------------------------

        return mem_out