from dataclasses import dataclass


TIMESTEPS = [2, 4, 6, 8, 10]

ENCODINGS = [
    "direct",
    "latency",
    "poisson",
]


@dataclass(frozen=True)
class Experiment:

    encoder: str

    timesteps: int


def all_experiments():

    return [
        Experiment(e, t)
        for e in ENCODINGS
        for t in TIMESTEPS
    ]