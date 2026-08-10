from collections import defaultdict

import heapq

class BPETrainer:
    """
    Production-style Byte Pair Encoding trainer.

    Responsibilities
    ----------------
    - Build unique-word corpus
    - Learn BPE merges
    - Maintain pair statistics
    - Build vocabulary
    """

    def __init__(self):

        # ----------------------------------
        # Corpus
        # word_id -> tuple(symbols)
        # ----------------------------------

        self.corpus = {}

        # ----------------------------------
        # Word frequencies
        # word_id -> frequency
        # ----------------------------------

        self.word_freq = {}
        self.next_word_id = 0
        self.word_lookup = {}

        # ----------------------------------
        # Pair statistics
        # pair -> frequency
        # ----------------------------------

        self.pair_stats = defaultdict(int)

        # ----------------------------------
        # Reverse index
        # pair -> set(word_ids)
        # ----------------------------------

        self.pair_index = defaultdict(set)

        # ----------------------------------
        # Learned merges
        # ----------------------------------

        self.merges = []

        # ----------------------------------
        # Final vocabulary
        # ----------------------------------

        self.symbols = set()

        # ----------------------------------
        # Priority queue
        #
        # (-frequency, pair)
        # ----------------------------------

        self.heap = []

    def _reset(self):

        self.corpus.clear()

        self.word_freq.clear()

        self.pair_stats.clear()

        self.pair_index.clear()

        self.merges.clear()

        self.symbols.clear()

        self.next_word_id = 0

        self.word_lookup.clear()

        self.heap.clear()

    def _build_corpus(self,tokens):
        """
        Build unique-word corpus and word frequencies.
        """

        for token in tokens:

            symbols = tuple(
                list(token) + ["</w>"]
            )

            if symbols not in self.word_lookup:

                wid = self.next_word_id

                self.next_word_id += 1

                self.word_lookup[symbols] = wid

                self.corpus[wid] = symbols

                self.word_freq[wid] = 1

            else:

                wid = self.word_lookup[symbols]

                self.word_freq[wid] += 1
        return self.corpus_statistics()

    def corpus_statistics(self):

        return {

            "words":
                sum(self.word_freq.values()),

            "unique_words":
                len(self.corpus),

            "characters":
                sum(

                    len(word)

                    * self.word_freq[wid]

                    for wid, word

                    in self.corpus.items()

                ),
        }
    
    def _build_pair_statistics(self):
        """
        Build initial pair frequencies and
        reverse pair index.
        """

        self.pair_stats.clear()
        self.pair_index.clear()

        for word_id, symbols in self.corpus.items():

            freq = self.word_freq[word_id]

            for i in range(len(symbols) - 1):

                pair = (
                    symbols[i],
                    symbols[i + 1],
                )

                self.pair_stats[pair] += freq

                self.pair_index[pair].add(word_id)
                
        self._build_heap()


    def pair_statistics(self):

        return {

            "unique_pairs": len(self.pair_stats),

            "indexed_pairs": len(self.pair_index),

            "total_pair_occurrences": sum(
                self.pair_stats.values()
            ),
        }

    def words_containing_pair(self, pair,):

        return self.pair_index.get(pair, set(),)

    def _merge_word(self, symbols, pair,):
        """
        Merge one occurrence of a pair inside a word.
        """

        merged = []

        i = 0

        merged_symbol = pair[0] + pair[1]

        changed = False

        while i < len(symbols):

            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):

                merged.append(
                    merged_symbol
                )

                i += 2

                changed = True

            else:

                merged.append(
                    symbols[i]
                )

                i += 1

        return tuple(merged), changed

    def _remove_word_pairs(self, word_id,):
        """
        Remove a word's contribution from
        pair statistics.
        """

        symbols = self.corpus[word_id]

        freq = self.word_freq[word_id]

        for i in range(len(symbols) - 1):

            pair = (
                symbols[i],
                symbols[i + 1],
            )

            self.pair_stats[pair] -= freq

            if self.pair_stats[pair] <= 0:

                del self.pair_stats[pair]
            else:
                self._push_pair(pair)

            self.pair_index[pair].discard(
                word_id
            )

            if not self.pair_index[pair]:

                del self.pair_index[pair]

    def _add_word_pairs(self,word_id,):
        """
        Add a word's contribution to
        pair statistics.
        """

        symbols = self.corpus[word_id]

        freq = self.word_freq[word_id]

        for i in range(len(symbols) - 1):

            pair = (
                symbols[i],
                symbols[i + 1],
            )

            self.pair_stats[pair] += freq
            self._push_pair(pair)

            self.pair_index[pair].add(
                word_id
            )

    def _merge_pair(self, pair,):
        """
        Merge one BPE pair across the corpus.
        """

        affected = list(
            self.words_containing_pair(
                pair
            )
        )

        merged_symbol = (
            pair[0] + pair[1]
        )

        self.symbols.add(merged_symbol)

        for word_id in affected:

            self._remove_word_pairs(
                word_id
            )

            symbols = self.corpus[word_id]

            merged, changed = self._merge_word(
                symbols,
                pair,
            )

            if changed:

                self.corpus[word_id] = merged

            self._add_word_pairs(word_id)

        self.merges.append(pair)


    def _learn_next_merge(self,):
        """
        Learn the next BPE merge.
        """

        pair = self._pop_best_pair()

        if pair is None:
            return False

        self._merge_pair(
            pair
        )

        return True

    def fit(self, tokens, vocab_size=512,):
        """
        Train a BPE vocabulary.

        Parameters
        ----------
        tokens : Iterable[str]
            Pretokenized corpus.

        vocab_size : int
            Target vocabulary size.
        """

        self._reset()

        # -----------------------------
        # Build corpus
        # -----------------------------

        self._build_corpus(tokens)

        # -----------------------------
        # Initial vocabulary
        # -----------------------------

        for symbols in self.corpus.values():

            self.symbols.update(symbols)

        # -----------------------------
        # Build initial pair statistics
        # -----------------------------

        self._build_pair_statistics()
        # -----------------------------
        # Learn merges
        # -----------------------------

        print(f"Initial symbols : {len(self.symbols)}")

        while len(self.symbols) < vocab_size:

            success = self._learn_next_merge()

            if not success:
                print("No more merges available.")
                break
        #self.vocab = sorted(self.symbols)
        print(f"Final symbols : {len(self.symbols)}")
        print(f"Learned merges : {len(self.merges)}")

        return self
    def _build_heap(self):
        """
        Build the initial max heap.
        """

        self.heap.clear()

        for pair, freq in self.pair_stats.items():

            heapq.heappush(

                self.heap,

                (-freq, pair),

            )

    def _push_pair(self, pair,):
        """
        Push updated pair statistics
        onto the heap.
        """

        if pair not in self.pair_stats:
            return

        freq = self.pair_stats[pair]

        heapq.heappush(

            self.heap,

            (-freq, pair),

        )

    def _pop_best_pair(self):
        """
        Return the highest-frequency
        valid pair.

        Uses lazy invalidation.
        """

        while self.heap:

            neg_freq, pair = heapq.heappop(
                self.heap
            )

            freq = -neg_freq

            current = self.pair_stats.get(
                pair
            )

            if current is None:
                continue

            if current != freq:
                continue

            return pair

        return None