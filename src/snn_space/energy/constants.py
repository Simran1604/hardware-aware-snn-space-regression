from dataclasses import dataclass


@dataclass(frozen=True)
class NDAConstants:
    """
    Table A.6 of the paper.
    Energy values are in picojoules (pJ).
    """

    E_ADD = 0.9
    E_MUL = 3.7
    E_SUB = 0.9
    E_CMP = 0.9

    E_R_WEIGHT = 20.0
    E_R_STATE = 20.0
    E_R_LEAK = 20.0
    E_W_STATE = 20.0

    E_TP_HOP = 10.0

    DEFAULT_HOPS = 8