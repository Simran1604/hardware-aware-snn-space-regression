from dataclasses import dataclass


@dataclass
class ExperimentResult:

    encoder: str

    timesteps: int

    run: int

    mae: float

    euclidean_error: float

    emac_energy: float

    nda_energy: float