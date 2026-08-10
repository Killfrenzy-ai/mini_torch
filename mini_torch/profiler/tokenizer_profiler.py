"""
Advanced tokenizer profiler.

Profiles:
- Overall fit()/encode()/decode()
- Internal BPE operations
- Memory usage
- Throughput
- Hotspots

Can be attached to any tokenizer.
"""

from __future__ import annotations

import functools
import time
import tracemalloc
from collections import defaultdict


class TokenizerProfiler:

    def __init__(self):

        self.stats = defaultdict(float)

        self.calls = defaultdict(int)

        self.counters = defaultdict(int)

        self.enabled = False

    # -----------------------------------------------------
    # Timing decorator
    # -----------------------------------------------------

    def profile(self, name):

        def decorator(fn):

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):

                if not self.enabled:
                    return fn(*args, **kwargs)

                self.calls[name] += 1

                start = time.perf_counter()

                out = fn(*args, **kwargs)

                elapsed = time.perf_counter() - start

                self.stats[name] += elapsed

                return out

            return wrapper

        return decorator

    # -----------------------------------------------------

    def increment(self, name, amount=1):

        self.counters[name] += amount

    # -----------------------------------------------------

    def start(self):

        self.enabled = True

        self.stats.clear()

        self.calls.clear()

        self.counters.clear()

        tracemalloc.start()

        self.total_start = time.perf_counter()

    # -----------------------------------------------------

    def stop(self):

        self.total_time = time.perf_counter() - self.total_start

        _, peak = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        self.peak_memory = peak / (1024 * 1024)

        self.enabled = False

    # -----------------------------------------------------

    def report(self):

        print()

        print("=" * 80)
        print("TOKENIZER PROFILE")
        print("=" * 80)

        # =====================================================
        # Corpus Statistics
        # =====================================================

        print("\nCorpus Statistics")
        print("-" * 80)

        words = self.counters.get("words", 0)
        input_symbols = self.counters.get("input_symbols", 0)

        print(f"{'Words':35}{words:,}")
        print(f"{'Unique Words':35}{self.counters.get('unique_words',0):,}")
        print(f"{'Longest Word':35}{self.counters.get('longest_word',0)}")
        print(f"{'Shortest Word':35}{self.counters.get('shortest_word',0)}")
        print(f"{'Input Symbols':35}{input_symbols:,}")

        if words:
            print(
                f"{'Average Symbols / Word':35}"
                f"{input_symbols / words:.2f}"
            )

        # =====================================================
        # Vocabulary Statistics
        # =====================================================

        print("\nVocabulary Statistics")
        print("-" * 80)

        print(
            f"{'Initial Vocabulary':35}"
            f"{self.counters.get('initial_vocab',0):,}"
        )

        print(
            f"{'Target Vocabulary':35}"
            f"{self.counters.get('target_vocab',0):,}"
        )

        print(
            f"{'Final Vocabulary':35}"
            f"{self.counters.get('final_vocab',0):,}"
        )

        print(
            f"{'Vocabulary Growth':35}"
            f"{self.counters.get('vocabulary_growth',0):,}"
        )

        print(
            f"{'Learned Merges':35}"
            f"{self.counters.get('learned_merges',0):,}"
        )

        # =====================================================
        # Merge Statistics
        # =====================================================

        print("\nMerge Statistics")
        print("-" * 80)

        merge_iterations = self.counters.get(
            "merge_iterations",
            0,
        )

        merge_attempts = self.counters.get(
            "merge_attempts",
            0,
        )

        merge_replacements = self.counters.get(
            "merge_replacements",
            0,
        )

        merge_replays = self.counters.get(
            "merge_replays",
            0,
        )

        pair_comparisons = self.counters.get(
            "pair_comparisons",
            0,
        )

        pair_words = self.counters.get(
            "pair_count_words",
            0,
        )

        symbols_processed = self.counters.get(
            "symbols_processed",
            0,
        )

        print(f"{'Merge Iterations':35}{merge_iterations:,}")
        print(f"{'Merge Replays':35}{merge_replays:,}")
        print(f"{'Merge Attempts':35}{merge_attempts:,}")
        print(f"{'Merge Replacements':35}{merge_replacements:,}")
        print(f"{'Pair Count Words':35}{pair_words:,}")
        print(f"{'Pair Comparisons':35}{pair_comparisons:,}")
        print(f"{'Symbols Processed':35}{symbols_processed:,}")

        if merge_attempts:

            efficiency = (
                merge_replacements
                / merge_attempts
            ) * 100

            print(
                f"{'Replacement Efficiency':35}"
                f"{efficiency:.2f}%"
            )

        if merge_iterations:

            print(
                f"{'Pair Comparisons / Merge':35}"
                f"{pair_comparisons / merge_iterations:,.2f}"
            )

            print(
                f"{'Symbols / Merge':35}"
                f"{symbols_processed / merge_iterations:,.2f}"
            )

        # =====================================================
        # Encoding Statistics
        # =====================================================

        print("\nEncoding Statistics")
        print("-" * 80)

        words_encoded = self.counters.get(
            "words_encoded",
            0,
        )

        tokens_encoded = self.counters.get(
            "tokens_encoded",
            0,
        )

        chars_encoded = self.counters.get(
            "characters_encoded",
            0,
        )

        input_encode = self.counters.get(
            "encode_input_symbols",
            0,
        )

        output_encode = self.counters.get(
            "encode_output_symbols",
            0,
        )

        print(f"{'Words Encoded':35}{words_encoded:,}")
        print(f"{'Characters Encoded':35}{chars_encoded:,}")
        print(f"{'Tokens Produced':35}{tokens_encoded:,}")
        print(f"{'Input Symbols':35}{input_encode:,}")
        print(f"{'Output Symbols':35}{output_encode:,}")

        if output_encode:

            compression = (
                input_encode
                / output_encode
            )

            print(
                f"{'Compression Ratio':35}"
                f"{compression:.2f}x"
            )

        if words_encoded:

            print(
                f"{'Tokens / Word':35}"
                f"{tokens_encoded / words_encoded:.2f}"
            )

        # =====================================================
        # Throughput
        # =====================================================

        print("\nThroughput")
        print("-" * 80)

        fit_time = self.stats.get("fit", 0)
        encode_time = self.stats.get("encode", 0)
        decode_time = self.stats.get("decode", 0)

        if fit_time:

            print(
                f"{'Training Words/sec':35}"
                f"{words / fit_time:,.2f}"
            )

            print(
                f"{'Merge Iterations/sec':35}"
                f"{merge_iterations / fit_time:,.2f}"
            )

        if encode_time:

            print(
                f"{'Encoding Words/sec':35}"
                f"{words_encoded / encode_time:,.2f}"
            )

            print(
                f"{'Encoding Tokens/sec':35}"
                f"{tokens_encoded / encode_time:,.2f}"
            )

            print(
                f"{'Encoding Chars/sec':35}"
                f"{chars_encoded / encode_time:,.2f}"
            )

        if decode_time:

            decoded_tokens = self.counters.get(
                "decoded_tokens",
                0,
            )

            decoded_chars = self.counters.get(
                "decoded_characters",
                0,
            )

            print(
                f"{'Decode Tokens/sec':35}"
                f"{decoded_tokens / decode_time:,.2f}"
            )

            print(
                f"{'Decode Chars/sec':35}"
                f"{decoded_chars / decode_time:,.2f}"
            )

        # =====================================================
        # Function Timings
        # =====================================================

        print("\nFunction Timings")
        print("-" * 80)

        total_profiled = sum(self.stats.values())

        rows = sorted(
            self.stats.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        print(
            f"{'Function':30}"
            f"{'Time':>12}"
            f"{'Calls':>12}"
            f"{'% Total':>12}"
        )

        print("-" * 80)

        for name, t in rows:

            pct = (
                (t / total_profiled) * 100
                if total_profiled
                else 0
            )

            print(
                f"{name:<30}"
                f"{t:>12.3f}s"
                f"{self.calls[name]:>12}"
                f"{pct:>11.1f}%"
            )

        # =====================================================
        # Memory
        # =====================================================

        print("\nMemory")
        print("-" * 80)

        print(
            f"{'Peak Memory':35}"
            f"{self.peak_memory:.2f} MB"
        )

        # =====================================================
        # Summary
        # =====================================================

        print("\nSummary")
        print("-" * 80)

        print(
            f"{'Total Runtime':35}"
            f"{self.total_time:.3f} s"
        )

        print(
            f"{'Profiled Runtime':35}"
            f"{total_profiled:.3f} s"
        )

        print("=" * 80)