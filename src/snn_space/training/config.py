from dataclasses import dataclass


@dataclass
class TrainingConfig:

    learning_rate = 1e-3

    batch_size = 1

    epochs = 10

    patience = 20

    optimizer = "Adam"

    loss = "MSE"

    simulation_steps = [2, 4, 6, 8, 10]