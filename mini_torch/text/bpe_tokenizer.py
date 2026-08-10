import pickle
import re
import os

from mini_torch.text.bpe.trainer import BPETrainer
from mini_torch.text.bpe.vocabulary import Vocabulary
from mini_torch.text.bpe.encoder import BPEEncoder


class BPETokenizer:
    """
    High-level Byte Pair Encoding tokenizer.

    Responsibilities
    ----------------
    - Train a BPE vocabulary
    - Encode text
    - Decode token ids
    - Save / load tokenizer state
    """

    DEFAULT_CHECKPOINT_PATH = (
        r"checkpoints/bpe_tokenizer.pkl"
    )

    def __init__(self):

        self.trainer = None

        self.vocab = Vocabulary()

        self.encoder = None

        self.vocab_size = 0

        self.is_trained = False

    # ---------------------------------------------------------
    # Internal pre-tokenizer
    # ---------------------------------------------------------

    @staticmethod
    def _split_text(text):
        """
        Temporary pre-tokenizer.

        Will later become a standalone
        BPENormalizer + PreTokenizer.
        """

        return re.findall(

            r"\s+|\w+|[^\w\s]",

            text,

        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train(
        self,
        text,
        vocab_size=16000,
    ):

        tokens = self._split_text(text)

        self.trainer = BPETrainer()

        self.trainer.fit(

            tokens,

            vocab_size=vocab_size,

        )

        self.vocab.build(

            self.trainer.symbols,

        )

        self.encoder = BPEEncoder(

            self.vocab,

            self.trainer.merges,

        )

        self.vocab_size = self.vocab.vocab_size

        self.is_trained = True

        return self

    def encode(self,text,):
        """
        Encode raw text into token ids.
        """

        if not self.is_trained:

            raise RuntimeError(
                "Tokenizer has not been trained."
            )

        tokens = self._split_text(text)

        return self.encoder.encode(tokens)

    def decode(self, ids,):
        """
        Decode token ids back into text.
        """

        if not self.is_trained:

            raise RuntimeError(
                "Tokenizer has not been trained."
            )

        symbols = [

            self.vocab.id_to_token(idx)

            for idx in ids

        ]

        text = []

        for symbol in symbols:

            if symbol == "</w>":

                continue

            if symbol.endswith("</w>"):

                text.append(

                    symbol[:-4]

                )

            else:

                text.append(symbol)

        return "".join(text)

    def state_dict(self):
        """
        Return the complete tokenizer state.
        """

        if not self.is_trained:

            raise RuntimeError(
                "Tokenizer has not been trained."
            )

        return {

            "merges": self.trainer.merges,

            "symbols": list(self.trainer.symbols),

            "vocabulary": self.vocab.state_dict(),

        }

    def load_state_dict(self, state,):
        """
        Restore tokenizer from a saved state.
        """

        self.trainer = BPETrainer()

        self.trainer.merges = state["merges"]

        self.trainer.symbols = set(

            state["symbols"]

        )

        self.vocab.load_state_dict(

            state["vocabulary"]

        )

        self.encoder = BPEEncoderV3(

            self.vocab,

            self.trainer.merges,

        )

        self.vocab_size = self.vocab.vocab_size

        self.is_trained = True

    def save(self, path=DEFAULT_CHECKPOINT_PATH,):
        """
        Save tokenizer to disk.
        """

        if not self.is_trained:

            raise RuntimeError(
                "Tokenizer has not been trained."
            )

        os.makedirs(

            os.path.dirname(path),

            exist_ok=True,

        )

        with open(
            path,
            "wb",
        ) as f:

            pickle.dump(

                self.state_dict(),

                f,

            )

    def load(self, path=DEFAULT_CHECKPOINT_PATH,):
        """
        Load tokenizer from disk.
        """

        with open(
            path,
            "rb",
        ) as f:

            state = pickle.load(f)

        self.load_state_dict(state)

        return self

    @property
    def pad_id(self):

        return self.vocab.pad_id


    @property
    def unk_id(self):

        return self.vocab.unk_id


    @property
    def vocabulary(self):

        return self.vocab