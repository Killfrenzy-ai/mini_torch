from mini_torch.data.dataset import Dataset
from mini_torch.tensors import tensor
from numbers import Integral

import numpy as np


class TextDataset(Dataset):
    """
    Dataset for next-token prediction.

    Each sample consists of an input sequence and the
    corresponding target sequence shifted by one token.
    """

    def __init__(self,token_ids,context_length,):

        self.tokens = token_ids
        self.context_length = context_length

        if context_length <= 0:
            raise ValueError(
                "context_length must be positive."
            )

        if len(token_ids) <= context_length:
            raise ValueError(
                "token_ids must contain more tokens than context_length."
            )
        
        self.tokens = np.asarray(token_ids , dtype = np.int64)

    def __len__(self):
        return len(self.tokens) - self.context_length
    
    def __getitem__(self,index):

        if not isinstance(index, Integral):
            raise TypeError("Index must be an integer.")

        if index < 0 or index >= len(self):
            raise IndexError("TextDataset index out of range.")

        x = self.tokens[index : index + self.context_length]

        y = self.tokens[index + 1: index + self.context_length + 1]

        return (tensor(x),tensor(y))