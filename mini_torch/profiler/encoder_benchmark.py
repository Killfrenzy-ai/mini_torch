from pathlib import Path
from time import perf_counter

from mini_torch.text.bpe.trainer import BPETrainer
from mini_torch.text.bpe.vocabulary import Vocabulary

from mini_torch.text.bpe.encoder import (
    BPEEncoder,
    profiler as profiler_v1,
)

from mini_torch.text.bpe.encoder_v2 import (
    BPEEncoderV2,
    profiler as profiler_v2,
)


def split_text(text):
    """
    Temporary tokenizer until BPETokenizer exists.
    """
    import re

    return re.findall(
        r"\s+|\w+|[^\w\s]",
        text,
    )


# =====================================================
# Load dataset
# =====================================================

print("=" * 80)
print("Loading dataset...")
print("=" * 80)

text = Path(
    r"mini_torch/examples/data/full_shakespeare.txt"
).read_text(
    encoding="utf-8"
)

tokens = split_text(text)

print(f"Words : {len(tokens):,}")
print()


# =====================================================
# Train BPE
# =====================================================

print("=" * 80)
print("Training...")
print("=" * 80)

trainer = BPETrainer()

trainer.fit(
    tokens,
    vocab_size=512,
)

print("Trainer finished.")
print()


# =====================================================
# Build vocabulary
# =====================================================

print("=" * 80)
print("Building vocabulary...")
print("=" * 80)

vocab = Vocabulary()

vocab.build(
    trainer.symbols,
)

print(f"Vocabulary size : {vocab.vocab_size}")
print()


# =====================================================
# Encoder V1
# =====================================================

print("=" * 80)
print("Benchmarking Encoder V1...")
print("=" * 80)

encoder_v1 = BPEEncoder(
    vocab,
    trainer.merges,
)

profiler_v1.reset()

profiler_v1.start()

t0 = perf_counter()

ids_v1 = encoder_v1.encode(tokens)

v1_time = perf_counter() - t0

profiler_v1.stop()

print(f"Generated ids : {len(ids_v1):,}")

profiler_v1.report()

print()


# =====================================================
# Encoder V2
# =====================================================

print("=" * 80)
print("Benchmarking Encoder V2...")
print("=" * 80)

encoder_v2 = BPEEncoderV2(
    vocab,
    trainer.merges,
)

profiler_v2.reset()

profiler_v2.start()

t0 = perf_counter()

ids_v2 = encoder_v2.encode(tokens)

v2_time = perf_counter() - t0

profiler_v2.stop()

print(f"Generated ids : {len(ids_v2):,}")

profiler_v2.report()

print()


# =====================================================
# Correctness
# =====================================================

print("=" * 80)
print("Verifying correctness...")
print("=" * 80)

for word in tokens:

    if word.isspace():
        continue

    a = encoder_v1._encode_word(word)
    b = encoder_v2._encode_word(word)

    if a != b:

        print(word)
        print(a)
        print(b)
        break

else:

    print("✓ Every word matches.")

# =====================================================
# Summary
# =====================================================

print("=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)

print(f"Encoder V1 : {v1_time:.3f} sec")
print(f"Encoder V2 : {v2_time:.3f} sec")

if v2_time > 0:

    print(
        f"Speedup    : "
        f"{v1_time / v2_time:.2f}x"
    )

print("=" * 80)