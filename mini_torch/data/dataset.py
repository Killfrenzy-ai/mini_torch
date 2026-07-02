from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Base class for all datasets.
    """

    @abstractmethod
    def __len__(self):
        """
        Return the number of samples.
        """
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        Return one sample.
        """
        pass