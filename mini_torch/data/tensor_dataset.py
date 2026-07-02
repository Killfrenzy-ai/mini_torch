from mini_torch.data.dataset import Dataset


class TensorDataset(Dataset):
    """
    Dataset wrapping one or more tensors.

    Every tensor must have the same number of samples
    along the first dimension.

    Example:
        dataset = TensorDataset(x, y)

        sample = dataset[5]

        x_i, y_i = sample
    """

    def __init__(self, *tensors):

        if len(tensors) == 0:
            raise ValueError(
                "TensorDataset requires at least one tensor."
            )

        length = len(tensors[0].data)

        for tensor in tensors:

            if len(tensor.data) != length:
                raise ValueError(
                    "All tensors must have the same first dimension."
                )

        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0].data)

    def __getitem__(self, index):
        return tuple(
            tensor[index]
            for tensor in self.tensors
        )