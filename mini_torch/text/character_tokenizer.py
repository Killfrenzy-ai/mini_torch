from mini_torch.text.tokenizer import Tokenizer

import os
import pickle


DEFAULT_CHECKPOINT_PATH = r"checkpoints\tokenizer.pkl"

class CharacterTokenizer(Tokenizer):
    """
    Character-level tokenizer.
    """

    def __init__(self):

        self.stoi = {}

        self.itos = {}

        self.vocab = []

        self.vocab_size = 0

    def fit(self, text):

        unique_chars = sorted(set(text))

        self.vocab = unique_chars

        self.vocab_size = len(unique_chars)

        self.stoi = {char:idx for idx,char in enumerate(unique_chars)}
        self.itos = {idx:char for char,idx in self.stoi.items()}

    def encode(self, text):
        """
        Convert text into token IDs.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        list[int]
            Token IDs.
        """

        if not self.stoi:
            raise RuntimeError("Tokenizer has not been fitted.")

        ids = []

        for char in text:

            if char not in self.stoi:
                raise ValueError(
                    f"Unknown character: {repr(char)}"
                )

            ids.append(self.stoi[char])

        return ids
    
    def decode(self, ids):
        """
        Convert token IDs back into text.

        Parameters
        ----------
        ids : sequence of int
            Token IDs.

        Returns
        -------
        str
            Decoded text.
        """

        if not self.itos:
            raise RuntimeError(
                "Tokenizer has not been fitted."
            )

        chars = []

        for idx in ids:

            if idx not in self.itos:
                raise ValueError(
                    f"Unknown token ID: {idx}"
                )

            chars.append(
                self.itos[idx]
            )

        return "".join(chars)
    
    def state_dict(self):

        return {
            "stoi" : self.stoi.copy(),
            "itos": self.itos.copy(),
            "vocab": self.vocab.copy(),
            "vocab_size": self.vocab_size
        }
    
    def load_state_dict(self,state):

        self.stoi = state["stoi"].copy()
        self.itos = state["itos"].copy()
        self.vocab = state["vocab"].copy()
        self.vocab_size = state["vocab_size"]

    def save(self, path=DEFAULT_CHECKPOINT_PATH):
        """
        Save Token Vocabulary
        """

        os.makedirs(os.path.dirname(path), exist_ok = True)

        with open(path,"wb") as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path=DEFAULT_CHECKPOINT_PATH):
        """
        Load tokenizer vocabulary
        """

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.load_state_dict(state)