import os
import pickle


class LRScheduler:

    def __init__(self, optimizer):

        self.optimizer = optimizer

        # Number of scheduler steps already performed.
        self.last_epoch = -1

    def get_lr(self):

        raise NotImplementedError

    def step(self):

        self.last_epoch += 1

        self.optimizer.lr = self.get_lr()

        return self.optimizer.lr

    def state_dict(self):
        """
        Return the scheduler state required to resume training.
        """

        return {
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict):
        """
        Restore the scheduler state.
        """

        self.last_epoch = state_dict["last_epoch"]

        # Recalculate and restore the LR corresponding
        # to the loaded scheduler position.
        self.optimizer.lr = self.get_lr()

    def save(self, path):
        """
        Save scheduler state to disk.
        """

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(path, "wb") as f:

            pickle.dump(
                self.state_dict(),
                f,
            )

    def load(self, path):
        """
        Load scheduler state from disk.
        """

        with open(path, "rb") as f:

            state = pickle.load(f)

        self.load_state_dict(state)

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"last_epoch={self.last_epoch}, "
            f"lr={self.optimizer.lr}"
            f")"
        )