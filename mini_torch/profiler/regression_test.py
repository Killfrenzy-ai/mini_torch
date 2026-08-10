from pathlib import Path
import re

from mini_torch.text.bpe.trainer import BPETrainer
from mini_torch.text.bpe.vocabulary import Vocabulary

from mini_torch.text.bpe.encoder import BPEEncoder
from mini_torch.text.bpe.encoder_v2 import BPEEncoderV2


# ==========================================================
# Helper
# ==========================================================

def split_text(text):

    return re.findall(
        r"\s+|\w+|[^\w\s]",
        text,
    )


# ==========================================================
# Load corpus
# ==========================================================

print("=" * 80)
print("Loading corpus...")
print("=" * 80)

text = Path(
    r"mini_torch/examples/data/full_shakespeare.txt"
).read_text(
    encoding="utf-8"
)

tokens = split_text(text)

print(f"Corpus Tokens : {len(tokens):,}")
print()


# ==========================================================
# Train tokenizer
# ==========================================================

print("=" * 80)
print("Training...")
print("=" * 80)

trainer = BPETrainer()

trainer.fit(
    tokens,
    vocab_size=512,
)

print()


# ==========================================================
# Vocabulary
# ==========================================================

vocab = Vocabulary()

vocab.build(
    trainer.symbols,
)

encoder_v1 = BPEEncoder(
    vocab,
    trainer.merges,
)

encoder_v2 = BPEEncoderV2(
    vocab,
    trainer.merges,
)


# ==========================================================
# Regression Tests
# ==========================================================

tests = [

    # Empty
    "",

    # Single characters
    "a",
    "A",
    "1",
    ".",
    "!",

    # Normal words
    "hello",
    "world",
    "Express",
    "Shakespeare",
    "therefore",
    "queen",
    "kingdom",
    "unhappy",

    # Mixed case
    "HELLO",
    "Hello",
    "HeLLo",

    # Numbers
    "123",
    "123456789",

    # Punctuation
    "...",
    "!!!",
    "---",
    "()",
    "{}",
    "[]",

    # Repeated letters
    "aa",
    "aaa",
    "aaaa",
    "aaaaaaaa",
    "ssssssss",
    "llllllll",

    # Spaces
    " ",
    "  ",
    "   ",
    "    ",
    "     ",
    "      ",
    "       ",
    "        ",

    # Tabs
    "\t",
    "\t\t",
    "\t\t\t",

    # Newlines
    "\n",
    "\n\n",
    "\n\n\n",

    # Mixed whitespace
    "\n\n      ",
    "      \n\n",
    "\t\t      ",

    # Mixed text
    "hello world",
    "hello\nworld",
    "hello\tworld",
]

print("=" * 80)
print("Running handcrafted regression tests...")
print("=" * 80)

failures = []

for test in tests:

    ids1 = encoder_v1.encode([test])
    ids2 = encoder_v2.encode([test])

    if ids1 != ids2:

        failures.append(
            (
                test,
                ids1,
                ids2,
            )
        )

if failures:

    print(f"\nFAILED : {len(failures)} handcrafted tests\n")

    for token, ids1, ids2 in failures[:20]:

        print("-" * 80)
        print(repr(token))
        print("V1 :", ids1)
        print("V2 :", ids2)

else:

    print("✓ All handcrafted tests passed.")

print()


# ==========================================================
# Full vocabulary regression
# ==========================================================

print("=" * 80)
print("Running unique vocabulary regression...")
print("=" * 80)

unique_tokens = sorted(
    set(tokens)
)

print(
    f"Unique Tokens : {len(unique_tokens):,}"
)

failures = []

for token in unique_tokens:

    ids1 = encoder_v1.encode([token])
    ids2 = encoder_v2.encode([token])

    if ids1 != ids2:

        failures.append(
            (
                token,
                ids1,
                ids2,
            )
        )

print()

if failures:

    print(
        f"FAILED : {len(failures)} unique tokens\n"
    )

    for token, ids1, ids2 in failures[:20]:

        print("-" * 80)
        print(repr(token))
        print("V1 :", ids1)
        print("V2 :", ids2)

else:

    print("✓ Entire vocabulary matches.")

print()


# ==========================================================
# Full corpus regression
# ==========================================================

print("=" * 80)
print("Running corpus regression...")
print("=" * 80)

failures = []

for index, token in enumerate(tokens):

    ids1 = encoder_v1.encode([token])
    ids2 = encoder_v2.encode([token])

    if ids1 != ids2:

        failures.append(
            (
                index,
                token,
                ids1,
                ids2,
            )
        )

print()

if failures:

    print(
        f"FAILED : {len(failures)} corpus mismatches\n"
    )

    for index, token, ids1, ids2 in failures[:20]:

        print("-" * 80)
        print(f"Index : {index}")
        print(repr(token))
        print("V1 :", ids1)
        print("V2 :", ids2)

else:

    print("✓ Entire corpus matches.")

print()


# ==========================================================
# Final Summary
# ==========================================================

print("=" * 80)
print("REGRESSION SUMMARY")
print("=" * 80)

print(f"Corpus Tokens      : {len(tokens):,}")
print(f"Unique Tokens      : {len(unique_tokens):,}")
print(f"Handcrafted Tests  : {len(tests)}")

print()

if failures:

    print("❌ Regression FAILED")

else:

    print("✅ Regression PASSED")

print("=" * 80)