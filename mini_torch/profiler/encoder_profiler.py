import time
from collections import defaultdict
from functools import wraps


class EncoderProfiler:
    """
    Profiler for the BPE encoder.
    """

    def __init__(self):

        self.timings = defaultdict(float)

        self.calls = defaultdict(int)

        self.counters = defaultdict(int)

        self.start_time = None

        self.end_time = None

    # -------------------------------------------------

    def profile(self, name):

        def decorator(func):

            @wraps(func)
            def wrapper(*args, **kwargs):

                start = time.perf_counter()

                result = func(*args, **kwargs)

                elapsed = time.perf_counter() - start

                self.timings[name] += elapsed

                self.calls[name] += 1

                return result

            return wrapper

        return decorator

    # -------------------------------------------------

    def increment(
        self,
        key,
        amount=1,
    ):

        self.counters[key] += amount

    # -------------------------------------------------

    def start(self):

        self.start_time = time.perf_counter()

    # -------------------------------------------------

    def stop(self):

        self.end_time = time.perf_counter()

    # -------------------------------------------------

    def reset(self):
        """
        Reset profiler state.
        """

        self.timings.clear()

        self.calls.clear()

        self.counters.clear()

        self.start_time = None

        self.end_time = None

    # -------------------------------------------------

    @property
    def total_time(self):

        if self.start_time is None:
            return 0.0

        if self.end_time is None:
            return time.perf_counter() - self.start_time

        return self.end_time - self.start_time

    # -------------------------------------------------

    def report(self):

        print()
        print("=" * 80)
        print("ENCODER PROFILER")
        print("=" * 80)
        print()

        print(f"Total Time : {self.total_time:.3f} sec")

        print()
        print("Function Timings")
        print("-" * 80)

        for name in sorted(self.timings):

            print(
                f"{name:<25}"
                f"{self.timings[name]:>10.3f}s"
                f"   Calls: {self.calls[name]:,}"
            )

        print()
        print("Counters")
        print("-" * 80)

        for key in sorted(self.counters):

            print(
                f"{key:<30}"
                f"{self.counters[key]:,}"
            )

        print()

        # -------------------------------------------------
        # Common statistics
        # -------------------------------------------------

        hits = self.counters["cache_hits"]
        misses = self.counters["cache_misses"]

        words = self.counters["words_encoded"]

        tokens = self.counters["tokens_generated"]

        chars = self.counters["characters_encoded"]

        merge_iterations = self.counters["merge_iterations"]

        successful_merges = self.counters["successful_merges"]

        total_cache = hits + misses

        if total_cache:

            print(
                f"Cache Hit Rate           : "
                f"{100 * hits / total_cache:.2f}%"
            )

        if self.total_time > 0:

            print(
                f"Words/sec                : "
                f"{words / self.total_time:,.2f}"
            )

            print(
                f"Tokens/sec               : "
                f"{tokens / self.total_time:,.2f}"
            )

        print()
        print("Derived Metrics")
        print("-" * 80)

        if words:

            print(
                f"Average Tokens / Word    : "
                f"{tokens / words:.3f}"
            )

            print(
                f"Average Chars / Word     : "
                f"{chars / words:.3f}"
            )

            print(
                f"Merge Iterations / Word  : "
                f"{merge_iterations / words:.3f}"
            )

            print(
                f"Successful Merges / Word : "
                f"{successful_merges / words:.3f}"
            )

        # -------------------------------------------------
        # Encoder V2 metrics
        # -------------------------------------------------

        heap_pushes = self.counters["heap_pushes"]
        heap_pops = self.counters["heap_pops"]
        stale_entries = self.counters["stale_heap_entries"]
        node_merges = self.counters["node_merges"]
        neighbor_updates = self.counters["neighbor_updates"]

        if (
            heap_pushes
            or heap_pops
            or node_merges
        ):

            print()
            print("Heap Statistics")
            print("-" * 80)

            print(
                f"Heap Pushes              : "
                f"{heap_pushes:,}"
            )

            print(
                f"Heap Pops                : "
                f"{heap_pops:,}"
            )

            if heap_pushes:

                print(
                    f"Heap Pop/Push Ratio      : "
                    f"{heap_pops / heap_pushes:.3f}"
                )

            if heap_pops:

                print(
                    f"Stale Heap Entries       : "
                    f"{100 * stale_entries / heap_pops:.2f}%"
                )

            print(
                f"Node Merges              : "
                f"{node_merges:,}"
            )

            print(
                f"Neighbor Updates         : "
                f"{neighbor_updates:,}"
            )

        print("=" * 80)