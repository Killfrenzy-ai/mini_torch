from pathlib import Path

from mini_torch.text.bpe_tokenizer import (
    BPETokenizer,
    profiler,
)


def main():

    DATASET_PATH = (
        "mini_torch/examples/data/full_shakespeare.txt"
    )

    VOCAB_SIZE = 512

    print("=" * 80)
    print("Loading dataset...")
    print("=" * 80)

    text = Path(
        DATASET_PATH
    ).read_text(
        encoding="utf-8"
    )

    print(f"Characters : {len(text):,}")
    print()

    tokenizer = BPETokenizer()

    profiler.start()

    print("=" * 80)
    print("Training tokenizer...")
    print("=" * 80)

    tokenizer.fit(
        text=text,
        vocab_size=VOCAB_SIZE,
    )
    print("Total words :", profiler.counters["words"])
    print("Unique words:", profiler.counters["unique_words"])

    print()

    print("=" * 80)
    print("Encoding corpus...")
    print("=" * 80)

    encoded = tokenizer.encode(text)

    print()

    print("=" * 80)
    print("Decoding corpus...")
    print("=" * 80)

    decoded = tokenizer.decode(encoded)

    profiler.stop()

    print()

    profiler.report()

    print()

    print("=" * 80)
    print("Verification")
    print("=" * 80)

    print(
        f"Vocabulary Size : {tokenizer.vocab_size:,}"
    )

    print(
        f"Merges Learned  : {len(tokenizer.merges):,}"
    )

    print(
        f"Encoded Tokens  : {len(encoded):,}"
    )

    print(
        f"Round-trip Match: {decoded == text.strip()}"
    )


if __name__ == "__main__":
    main()