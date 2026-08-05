class EarlyStopping:

    def __init__(self, patience=20):

        self.patience = patience

        self.best_loss = float("inf")

        self.counter = 0

    def step(self, loss):

        if loss < self.best_loss:

            self.best_loss = loss

            self.counter = 0

            return False

        self.counter += 1

        return self.counter >= self.patience