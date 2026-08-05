from snn_space.experiments.config import all_experiments


def main():

    experiments = all_experiments()

    print(f"{len(experiments)} experiments")

    for exp in experiments:

        print(
            exp.encoder,
            exp.timesteps,
        )


if __name__ == "__main__":

    main()