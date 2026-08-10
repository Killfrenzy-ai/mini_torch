import os
import pickle

class Vocabulary:

    DEFAULT_CHECKPOINT_PATH = r"checkpoints\bpe_tokenizer.pkl"

    def __init__(self):

        self.tokens = []

        self.stoi = {}

        self.itos = {}

        self.special_tokens = (
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
            "<mask>",
        )

    def build(self, symbols):

        symbols = sorted(set(symbols) - set(self.special_tokens))

        tokens = []

        tokens.extend(self.special_tokens)

        tokens.extend(symbols)

        self.tokens = tokens
        self.stoi = {token: idx for idx, token in enumerate(self.tokens)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
    @property
    def vocab_size(self):

        return len(self.tokens)

    @property
    def unk_id(self):

        return self.stoi["<unk>"]

    @property
    def pad_id(self):

        return self.stoi["<pad>"]

    def state_dict(self):

        return {

            "tokens": self.tokens,
        }

    def load_state_dict(self, state,):

        self.tokens = state["tokens"]

        self.stoi = {token: idx for idx, token in enumerate(self.tokens)}

        self.itos = {idx: token for token, idx in self.stoi.items()}

    def save(self, path=DEFAULT_CHECKPOINT_PATH):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path=DEFAULT_CHECKPOINT_PATH):

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.load_state_dict(state)

    def __contains__(self, token,):
        """
        Check whether a token exists in the vocabulary.
        """

        return token in self.stoi

    def token_to_id(self, token,):
        """
        Convert a token into its integer id.

        Unknown tokens are mapped to <unk>.
        """

        return self.stoi.get(
            token,
            self.unk_id,
        )

    def id_to_token(self, idx,):
        """
        Convert an integer id back into its token.
        """

        return self.itos[idx]

    @property
    def bos_id(self):
        """
        Beginning-of-sequence token id.
        """

        return self.stoi["<bos>"]

    @property
    def eos_id(self):
        """
        End-of-sequence token id.
        """

        return self.stoi["<eos>"]

    @property
    def mask_id(self):
        """
        Mask token id.
        """

        return self.stoi["<mask>"]
