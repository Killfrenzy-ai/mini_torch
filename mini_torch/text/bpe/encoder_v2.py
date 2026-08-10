import heapq

from mini_torch.profiler.encoder_profiler import EncoderProfiler


profiler = EncoderProfiler()


# ============================================================
# Symbol Node
# ============================================================

class SymbolNode:
    """
    Doubly-linked list node representing one symbol.

    The node's position is assigned once when the linked list
    is built and never changes. This allows deterministic
    left-to-right tie breaking.
    """

    __slots__ = (
        "symbol",
        "position",
        "prev",
        "next",
        "alive",
    )

    def __init__(
        self,
        symbol,
        position,
    ):

        self.symbol = symbol

        self.position = position

        self.prev = None

        self.next = None

        self.alive = True

    def __repr__(self):

        return (
            f"SymbolNode("
            f"symbol={self.symbol!r}, "
            f"position={self.position}"
            f")"
        )


# ============================================================
# Pair Candidate
# ============================================================

class PairCandidate:
    """
    Immutable heap entry.

    Every candidate stores a snapshot of the pair at the time
    it was inserted into the heap.

    If either node changes later, this candidate simply becomes
    stale and is ignored when popped.
    """

    __slots__ = (
        "rank",
        "position",

        "left",
        "right",

        "left_symbol",
        "right_symbol",
    )

    def __init__(
        self,
        rank,
        left,
        right,
    ):

        self.rank = rank

        # left-most node position
        self.position = left.position

        self.left = left
        self.right = right

        # immutable snapshot
        self.left_symbol = left.symbol
        self.right_symbol = right.symbol

    def __lt__(self, other):
        """
        Heap ordering.

        Lower merge rank wins.

        If two pairs have identical rank,
        the left-most pair wins.
        """

        return (
            self.rank,
            self.position,
        ) < (
            other.rank,
            other.position,
        )

    def pair(self):

        return (
            self.left_symbol,
            self.right_symbol,
        )

    def __repr__(self):

        return (
            "PairCandidate("
            f"rank={self.rank}, "
            f"position={self.position}, "
            f"pair={self.pair()}"
            ")"
        )


# ============================================================
# Encoder Skeleton
# ============================================================

class BPEEncoderV2:
    """
    Production-quality heap-based BPE encoder.

    Commit 1 only initializes the encoder.
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

    @profiler.profile("build_nodes")
    def _build_nodes(self, word):
        """
        Convert a word into a doubly-linked list.

        Each character becomes one SymbolNode.

        An </w> marker is appended automatically.
        """

        symbols = list(word)
        symbols.append("</w>")

        head = None
        previous = None

        for position, symbol in enumerate(symbols):

            node = SymbolNode(
                symbol=symbol,
                position=position,
            )

            profiler.increment("node_allocations")

            if head is None:
                head = node

            if previous is not None:
                previous.next = node
                node.prev = previous

            previous = node

        return head
    def _iter_nodes(self, head):
        """
        Iterate over every live node.

        Mainly used for debugging and symbol collection.
        """

        current = head

        while current is not None:

            if current.alive:
                yield current

            current = current.next

    @profiler.profile("collect_symbols")
    def _collect_symbols(self, head):
        """
        Convert the linked list back into a list of symbols.
        """

        return [
            node.symbol
            for node in self._iter_nodes(head)
        ]

    def _debug_print_list(self, head):
        """
        Pretty-print the current linked list.
        """

        print(
            self._collect_symbols(head)
        )

    def _push_candidate(self, heap, left, right,):
        """
        Push a merge candidate onto the heap.

        Nothing is pushed if the pair is not
        part of the learned merge table.
        """

        pair = (
            left.symbol,
            right.symbol,
        )

        rank = self.merge_ranks.get(pair)

        if rank is None:
            return

        candidate = PairCandidate(
            rank=rank,
            left=left,
            right=right,
        )

        heapq.heappush(
            heap,
            candidate,
        )

        profiler.increment("heap_pushes")

    @profiler.profile("build_heap")
    def _build_heap(self, head,):
        """
        Build the initial merge heap.

        Every adjacent pair is examined exactly once.
        """

        heap = []

        node = head

        while (
            node is not None
            and node.next is not None
        ):

            self._push_candidate(
                heap,
                node,
                node.next,
            )

            node = node.next

        return heap

    def _pop_candidate(self, heap,):
        """
        Pop the next candidate.

        Returns None if the heap is empty.
        """

        if not heap:
            return None

        profiler.increment("heap_pops")

        return heapq.heappop(heap)

    def _debug_print_heap(self, heap,):
        """
        Pretty-print the heap without
        modifying it.
        """

        print()

        print("Heap")

        print("-" * 50)

        for candidate in sorted(heap):

            print(

                f"rank={candidate.rank:>4}   "

                f"pos={candidate.position:>2}   "

                f"{candidate.pair()}"

            )
    
    def _valid_candidate(self, candidate,):
        """
        Returns True only if this heap entry still
        represents the same adjacent pair.
        """

        left = candidate.left
        right = candidate.right

        if not left.alive:
            return False

        if not right.alive:
            return False

        if left.next is not right:
            return False

        if left.symbol != candidate.left_symbol:
            return False

        if right.symbol != candidate.right_symbol:
            return False

        return True

    @profiler.profile("merge_nodes")
    def _merge_nodes(
        self,
        left,
        right,
    ):
        """
        Merge two adjacent nodes.

        Returns
        -------
        SymbolNode
            The merged (left) node.
        """

        profiler.increment("node_merges")

        left.symbol += right.symbol

        left.next = right.next

        if right.next is not None:
            right.next.prev = left

        right.alive = False

        right.prev = None
        right.next = None

        return left

    def _update_neighbors(self, heap, node,):
        """
        Only the pairs touching the merged node
        can become newly valid.
        """

        if (
            node.prev is not None
            and node.prev.alive
        ):

            self._push_candidate(
                heap,
                node.prev,
                node,
            )

            profiler.increment(
                "neighbor_updates"
            )

        if (
            node.next is not None
            and node.next.alive
        ):

            self._push_candidate(
                heap,
                node,
                node.next,
            )

            profiler.increment(
                "neighbor_updates"
            )

    def _merge_once(self, heap,):
        """
        Perform exactly one valid merge.

        Returns
        -------
        bool
            True if a merge happened.
        """

        while heap:

            candidate = self._pop_candidate(
                heap
            )

            if candidate is None:
                return False

            if not self._valid_candidate(
                candidate
            ):

                profiler.increment(
                    "stale_heap_entries"
                )

                continue

            profiler.increment(
                "merge_iterations"
            )

            merged = self._merge_nodes(
                candidate.left,
                candidate.right,
            )

            profiler.increment(
                "successful_merges"
            )

            self._update_neighbors(
                heap,
                merged,
            )

            return True

        return False

    def _debug_merge_word(self, word,):
        """
        Debug every merge performed on one word.
        """

        head = self._build_nodes(word)

        heap = self._build_heap(head)

        step = 1

        while self._merge_once(heap):

            print(
                f"Step {step:02d}"
            )

            print(
                self._collect_symbols(head)
            )

            step += 1

        print()

        print("Final")

        print(
            self._collect_symbols(head)
        )

    @profiler.profile("encode_word")
    def _encode_word(self, word):
        """
        Encode one pre-tokenized word.
        """

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

        # ------------------------------------
        # Build linked list
        # ------------------------------------

        head = self._build_nodes(word)

        # ------------------------------------
        # Build initial heap
        # ------------------------------------

        heap = self._build_heap(head)

        # ------------------------------------
        # Merge loop
        # ------------------------------------

        while self._merge_once(heap):
            pass

        # ------------------------------------
        # Collect symbols
        # ------------------------------------

        symbols = self._collect_symbols(head)

        # ------------------------------------
        # Convert to ids
        # ------------------------------------

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
        Encode an iterable of pre-tokenized words.
        """

        ids = []

        for token in tokens:

            ids.extend(

                self._encode_word(token)

            )

        return ids

    def clear_cache(self):
        """
        Clear cached encodings.
        """

        self.word_cache.clear()