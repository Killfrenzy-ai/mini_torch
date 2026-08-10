import heapq

from mini_torch.profiler.encoder_profiler import EncoderProfiler


profiler = EncoderProfiler()


class BPEEncoder:
    """
    Production-style BPE encoder.

    Uses contiguous arrays instead of Python objects.
    """

    def __init__(
        self,
        vocabulary,
        merges,
    ):

        self.vocab = vocabulary

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

        self.word_cache = {}

        # ---------------------------------------------
        # Precomputed merge information
        # ---------------------------------------------

        self.merge_info = {}

        self._build_merge_table()

    def _build_merge_table(self):
        """
        Precompute everything needed for each merge.

        pair -> (rank, merged_symbol)
        """

        self.merge_info.clear()

        for pair, rank in self.merge_ranks.items():

            left, right = pair

            self.merge_info[pair] = (

                rank,

                left + right,

            )

    # =====================================================
    # Build array representation
    # =====================================================

    @profiler.profile("build_arrays")
    def _build_arrays(
        self,
        word,
    ):
        """
        Build the linked list representation using arrays.

        Each symbol occupies one index.

        Example
        -------

        symbols:
            ['E','x','p','r','e','s','s','</w>']

        prev:
            [-1,0,1,2,3,4,5,6]

        next:
            [1,2,3,4,5,6,7,-1]

        alive:
            [T,T,T,T,T,T,T,T]
        """

        symbols = list(word)
        symbols.append("</w>")

        n = len(symbols)

        profiler.increment(
            "node_allocations",
            n,
        )

        prev = [-1] * n

        nxt = [-1] * n

        alive = [True] * n

        for i in range(n):

            if i > 0:
                prev[i] = i - 1

            if i < n - 1:
                nxt[i] = i + 1

        head = 0

        return (
            symbols,
            prev,
            nxt,
            alive,
            head,
        )

    # =====================================================
    # Iterate over live symbols
    # =====================================================

    def _iter_live(
        self,
        symbols,
        nxt,
        alive,
        head,
    ):
        """
        Iterate through the current linked list.
        """

        current = head

        while current != -1:

            if alive[current]:

                yield current

            current = nxt[current]

    # =====================================================
    # Collect symbols
    # =====================================================

    @profiler.profile("collect_symbols")
    def _collect_symbols(
        self,
        symbols,
        nxt,
        alive,
        head,
    ):
        """
        Convert the linked list back into
        a list of symbols.
        """

        result = []

        current = head

        while current != -1:

            if alive[current]:

                result.append(symbols[current])

            current = nxt[current]


        return result

    # =====================================================
    # Debug
    # =====================================================

    def _debug_arrays(
        self,
        symbols,
        prev,
        nxt,
        alive,
    ):

        print()

        print("Symbols")

        print(symbols)

        print()

        print("Prev")

        print(prev)

        print()

        print("Next")

        print(nxt)

        print()

        print("Alive")

        print(alive)

    def _push_candidate(self, heap, symbols, left, right,):
        """
        Push one merge candidate onto the heap.
        """

        pair = (
            symbols[left],
            symbols[right],
        )

        info = self.merge_info.get(pair)

        if info is None:
            return

        rank, merged_symbol = info

        heapq.heappush(

            heap,

            (
                rank,
                left,               # stable left-to-right ordering
                left,
                right,
                pair[0],
                pair[1],
                merged_symbol
            ),

        )

        profiler.increment("heap_pushes")

    @profiler.profile("build_heap")
    def _build_heap(self, symbols, nxt, head,):
        """
        Build the initial heap.
        """

        heap = []

        current = head

        while current != -1:

            right = nxt[current]

            if right != -1:

                self._push_candidate(

                    heap,

                    symbols,

                    current,

                    right,

                )

            current = right

        return heap

    def _pop_candidate(self, heap,):
        """
        Pop the next merge candidate.
        """

        if not heap:

            return None

        profiler.increment("heap_pops")

        return heapq.heappop(heap)

    def _debug_heap(self, heap,):
        """
        Pretty-print the heap.
        """

        print()

        print("Heap")

        print("-" * 60)

        for candidate in sorted(heap):

            rank, pos, left, right, a, b = candidate

            print(

                f"rank={rank:>4}"

                f"   pos={pos:>2}"

                f"   ({a!r}, {b!r})"

                f"   [{left}->{right}]"

            )

    def _valid_candidate(self, candidate, symbols, nxt, alive,):
        """
        Check whether a heap entry still represents
        the same adjacent pair.
        """

        (
            rank,
            position,
            left,
            right,
            left_symbol,
            right_symbol,
            merged_symbol
        ) = candidate

        if not alive[left]:
            return False

        if not alive[right]:
            return False

        if nxt[left] != right:
            return False

        if symbols[left] != left_symbol:
            return False

        if symbols[right] != right_symbol:
            return False

        return True

    @profiler.profile("merge")
    def _merge(self, left, right, merged_symbol, symbols, prev, nxt, alive,):
        """
        Merge two adjacent array nodes.

        Returns
        -------
        int
            Index of the merged node.
        """

        profiler.increment("node_merges")

        symbols[left] = merged_symbol

        nxt[left] = nxt[right]

        if nxt[right] != -1:

            prev[nxt[right]] = left

        alive[right] = False

        prev[right] = -1
        nxt[right] = -1

        return left

    def _update_neighbors(self, heap, node, symbols, prev, nxt, alive,):
        """
        Push newly-created neighboring pairs.
        """

        left_neighbor = prev[node]

        if (
            left_neighbor != -1
            and alive[left_neighbor]
        ):

            self._push_candidate(

                heap,

                symbols,

                left_neighbor,

                node,

            )

            profiler.increment(
                "neighbor_updates"
            )

        right_neighbor = nxt[node]

        if (
            right_neighbor != -1
            and alive[right_neighbor]
        ):

            self._push_candidate(

                heap,

                symbols,

                node,

                right_neighbor,

            )

            profiler.increment(
                "neighbor_updates"
            )

    def _merge_once(self, heap, symbols, prev, nxt, alive,):
        """
        Perform one valid merge.

        Returns
        -------
        bool
        """

        while heap:

            candidate = self._pop_candidate(
                heap
            )

            if candidate is None:
                return False

            if not self._valid_candidate(
                candidate,
                symbols,
                nxt,
                alive,
            ):

                profiler.increment(
                    "stale_heap_entries"
                )

                continue

            profiler.increment(
                "merge_iterations"
            )

            (
                _,
                _,
                left,
                right,
                _,
                _,
                merged_symbol,
            ) = candidate

            merged = self._merge(

                left,
                right,
                merged_symbol,

                symbols,
                prev,
                nxt,
                alive,

            )

            profiler.increment(
                "successful_merges"
            )

            self._update_neighbors(

                heap,

                merged,

                symbols,
                prev,
                nxt,
                alive,

            )

            return True

        return False

    def _debug_merge_word(self, word,):
        """
        Debug every merge performed.
        """

        (
            symbols,
            prev,
            nxt,
            alive,
            head,
        ) = self._build_arrays(
            word
        )

        heap = self._build_heap(
            symbols,
            nxt,
            head,
        )

        step = 1

        while self._merge_once(

            heap,

            symbols,
            prev,
            nxt,
            alive,

        ):

            print(f"Step {step:02d}")

            print(

                self._collect_symbols(

                    symbols,
                    nxt,
                    alive,
                    head,

                )

            )

            step += 1

        print()

        print("Final")

        print(

            self._collect_symbols(

                symbols,
                nxt,
                alive,
                head,

            )

        )

    @profiler.profile("encode_word")
    def _encode_word(self, word,):
        """
        Encode a single pre-tokenized word.
        """

        # -------------------------------------------------
        # Cache lookup
        # -------------------------------------------------

        profiler.increment("cache_lookups")

        cached = self.word_cache.get(word)

        if cached is not None:

            profiler.increment("cache_hits")

            return cached

        profiler.increment("cache_misses")
        profiler.increment("cache_entries")
        profiler.increment("words_encoded")
        profiler.increment(
            "characters_encoded",
            len(word),
        )

        profiler.counters["max_word_length"] = max(

            profiler.counters["max_word_length"],

            len(word),

        )

        # -------------------------------------------------
        # Build array representation
        # -------------------------------------------------

        (
            symbols,
            prev,
            nxt,
            alive,
            head,
        ) = self._build_arrays(word)

        # -------------------------------------------------
        # Initial heap
        # -------------------------------------------------

        heap = self._build_heap(

            symbols,
            nxt,
            head,

        )

        # -------------------------------------------------
        # Merge loop
        # -------------------------------------------------

        while self._merge_once(

            heap,

            symbols,
            prev,
            nxt,
            alive,

        ):
            pass

        # -------------------------------------------------
        # Collect final symbols
        # -------------------------------------------------

        symbols = self._collect_symbols(

            symbols,
            nxt,
            alive,
            head,

        )

        # -------------------------------------------------
        # Convert to ids
        # -------------------------------------------------

        encoded = tuple(

            self.vocab.token_to_id(symbol)

            for symbol in symbols

        )

        profiler.increment(

            "tokens_generated",

            len(encoded),

        )

        self.word_cache[word] = encoded

        return encoded

    @profiler.profile("encode")
    def encode(self, tokens,):
        """
        Encode a sequence of pre-tokenized words.
        """

        ids = []

        for token in tokens:

            ids.extend(

                self._encode_word(token)

            )

        return ids

    def clear_cache(self):
        """
        Clear the word cache.
        """

        self.word_cache.clear()