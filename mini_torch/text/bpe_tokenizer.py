from mini_torch.text.tokenizer import Tokenizer

import os
import pickle
import re


DEFAULT_CHECKPOINT_PATH = r"checkpoints\bpe_tokenizer.pkl"


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer.
    """

    def __init__(self):

        self.vocab = []

        self.stoi = {}

        self.itos = {}

        self.merges = []

        self.vocab_size = 0

    def _split_text(self, text):
        """
        Split text while preserving whitespace and punctuation.
        """

        return re.findall(
            r"\s+|\w+|[^\w\s]",
            text,
        )

    def _build_corpus(self, text):

        corpus = []

        for token in self._split_text(text):

            symbols = list(token)

            symbols.append("</w>")

            corpus.append(symbols)

        return corpus

    def _get_pair_counts(self, corpus):

        pair_counts = {}

        for word in corpus:

            for i in range(len(word) - 1):

                pair = (
                    word[i],
                    word[i + 1],
                )

                pair_counts[pair] = (
                    pair_counts.get(pair, 0) + 1
                )

        return pair_counts

    def _merge_pair(self, corpus, pair,):
        """
        Merge every occurrence of `pair`
        into a single symbol.

        Example

        ("l","o") -> "lo"
        """

        merged_symbol = pair[0] + pair[1]

        new_corpus = [self._apply_merge(word,pair) for word in corpus]

        return new_corpus

    def _apply_merge(self, symbols, pair,):
        """
        Apply a single learned merge
        to one tokenized word.
        """

        merged = pair[0] + pair[1]

        new_symbols = []

        i = 0

        while i < len(symbols):

            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):

                new_symbols.append(merged)

                i += 2

            else:

                new_symbols.append(symbols[i])

                i += 1

        return new_symbols

    def fit(self, text, vocab_size=512):
        """
        Learn BPE merges from text.
        """

        corpus = self._build_corpus(text)

        # ------------------------------------
        # Initial vocabulary (characters only)
        # ------------------------------------

        vocabulary = set()

        for word in corpus:
            vocabulary.update(word)

        # ------------------------------------
        # Learn merges
        # ------------------------------------

        while len(vocabulary) < vocab_size:

            pair_counts = self._get_pair_counts(corpus)

            if not pair_counts:
                break

            best_pair = max(
                pair_counts,
                key=pair_counts.get,
            )

            self.merges.append(best_pair)

            corpus = self._merge_pair(
                corpus,
                best_pair,
            )

            vocabulary.add(
                best_pair[0] + best_pair[1]
            )

            if len(self.merges) % 100 == 0:

                print(
                    f"Learned {len(self.merges)} merges..."
                )

        # ------------------------------------
        # Build vocabulary
        # ------------------------------------

        self.vocab = sorted(vocabulary)

        self.vocab_size = len(self.vocab)

        self.stoi = {
            token: i
            for i, token in enumerate(self.vocab)
        }

        self.itos = {
            i: token
            for token, i in self.stoi.items()
        }

    def encode(self, text):
        """
        Convert text into token ids.
        """

        ids = []

        for token in self._split_text(text):

            symbols = list(token)

            symbols.append("</w>")

            # Replay every learned merge
            for pair in self.merges:

                symbols = self._apply_merge(
                    symbols,
                    pair,
                )

            # Convert symbols to ids
            for symbol in symbols:

                if symbol not in self.stoi:

                    raise ValueError(
                        f"Unknown token: {symbol}"
                    )

                ids.append(
                    self.stoi[symbol]
                )

        return ids

    def decode(self, ids):

        text = ""

        for idx in ids:

            token = self.itos[idx]

            text += token.replace(
                "</w>",
                " "
            )

        # Collapse repeated whitespace
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Remove spaces before punctuation
        text = re.sub(
            r"\s+([,.;:!?])",
            r"\1",
            text,
        )

        # Remove spaces around apostrophes
        text = re.sub(
            r"\s*'\s*",
            "'",
            text,
        )

        # Preserve newlines while removing
        # surrounding spaces
        text = re.sub(
            r" *\n *",
            "\n",
            text,
        )

        return text.strip()

    def state_dict(self):

        return {
            "vocab": self.vocab.copy(),
            "stoi": self.stoi.copy(),
            "itos": self.itos.copy(),
            "merges": self.merges.copy(),
            "vocab_size": self.vocab_size,
        }

    def load_state_dict(self, state):

        self.vocab = state["vocab"].copy()
        self.stoi = state["stoi"].copy()
        self.itos = state["itos"].copy()
        self.merges = state["merges"].copy()
        self.vocab_size = state["vocab_size"]

    def save(self, path=DEFAULT_CHECKPOINT_PATH):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path=DEFAULT_CHECKPOINT_PATH):

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.load_state_dict(state)