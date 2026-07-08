from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """
    Base tokenizer interface.
    """

    @abstractmethod
    def fit(self, text):
        """
        Build the vocabulary.
        """
        pass

    @abstractmethod
    def encode(self, text):
        """
        Convert text into token IDs.
        """
        pass

    @abstractmethod
    def decode(self, ids):
        """
        Convert token IDs back to text.
        """
        pass