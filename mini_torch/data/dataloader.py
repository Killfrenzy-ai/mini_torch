import numpy as np

from mini_torch.tensors import tensor
from mini_torch.backend import xp


class DataLoader:
    """
    Iterate over a Dataset in mini-batches.

    Example:
        dataset = TensorDataset(x, y)

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
        )

        for batch_x, batch_y in loader:
            ...
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._indices = np.arange(len(dataset))
        self._position = 0

    def __len__(self):
        """
        Return the number of batches.
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        Start a new epoch.
        """
        self._position = 0

        if self.shuffle:
            np.random.shuffle(self._indices)

        return self

    def __next__(self):
        """
        Return the next mini-batch.
        """
        if self._position >= len(self.dataset):
            raise StopIteration

        start = self._position
        end = min(
            start + self.batch_size,
            len(self.dataset),
        )

        batch_indices = self._indices[start:end]

        self._position = end

        samples = [
            self.dataset[index]
            for index in batch_indices
        ]

        batched = []

        for column in zip(*samples):

            array = xp().stack(
                [item.data for item in column]
            )

            requires_grad = any(
                item.requires_grad
                for item in column
            )

            batched.append(
                tensor(
                    array,
                    requires_grad=requires_grad,
                )
            )

        return tuple(batched)