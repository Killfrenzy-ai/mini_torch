from pathlib import Path
import re

from mini_torch.text.bpe.trainer import BPETrainer
from mini_torch.text.bpe.vocabulary import Vocabulary
from mini_torch.text.bpe.encoder import BPEEncoder


def split_text(text):

    return re.findall(
        r"\s+|\w+|[^\w\s]",
        text,
    )


text = Path(
    r"mini_torch/examples/data/full_shakespeare.txt"
).read_text(
    encoding="utf-8"
)

tokens = split_text(text)

trainer = BPETrainer()

trainer.fit(
    tokens,
    vocab_size=512,
)

vocab = Vocabulary()

vocab.build(
    trainer.symbols,
)

encoder = BPEEncoder(
    vocab,
    trainer.merges,
)

symbols, prev, nxt, alive, head = encoder._build_arrays(
    "Express"
)

heap = encoder._build_heap(
    symbols,
    nxt,
    head,
)

encoder._debug_heap(heap)

symbols, prev, nxt, alive, head = encoder._build_arrays(
    "    "
)

heap = encoder._build_heap(
    symbols,
    nxt,
    head,
)

encoder._debug_heap(heap)