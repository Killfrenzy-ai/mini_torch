import pytest

from mini_torch.text.character_tokenizer import CharacterTokenizer


# ==========================================================
# Constructor
# ==========================================================

def test_constructor():

    tokenizer = CharacterTokenizer()

    assert tokenizer.stoi == {}
    assert tokenizer.itos == {}
    assert tokenizer.vocab_size == 0


# ==========================================================
# fit()
# ==========================================================

def test_fit_builds_vocab():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("hello")

    assert tokenizer.vocab_size == 4


def test_fit_builds_stoi():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("cabba")

    assert tokenizer.stoi == {
        "a": 0,
        "b": 1,
        "c": 2,
    }


def test_fit_builds_itos():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("cabba")

    assert tokenizer.itos == {
        0: "a",
        1: "b",
        2: "c",
    }


def test_sorted_vocab():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("dbca")

    assert list(tokenizer.stoi.keys()) == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_vocab_size():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("abcdef")

    assert tokenizer.vocab_size == 6


def test_duplicate_characters():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("aaaaaaaa")

    assert tokenizer.vocab_size == 1
    assert tokenizer.stoi == {
        "a": 0,
    }


def test_whitespace():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("a b")

    assert " " in tokenizer.stoi
    assert tokenizer.vocab_size == 3


def test_newline_character():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("a\nb")

    assert "\n" in tokenizer.stoi
    assert tokenizer.vocab_size == 3


def test_empty_string():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("")

    assert tokenizer.vocab_size == 0
    assert tokenizer.stoi == {}
    assert tokenizer.itos == {}


# ==========================================================
# Consistency
# ==========================================================

def test_stoi_itos_consistency():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("hello world")

    for char, idx in tokenizer.stoi.items():
        assert tokenizer.itos[idx] == char


def test_deterministic():

    t1 = CharacterTokenizer()
    t2 = CharacterTokenizer()

    text = "The quick brown fox"

    t1.fit(text)
    t2.fit(text)

    assert t1.stoi == t2.stoi
    assert t1.itos == t2.itos


def test_refit_replaces_vocab():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("abc")

    tokenizer.fit("xyz")

    assert tokenizer.vocab_size == 3

    assert tokenizer.stoi == {
        "x": 0,
        "y": 1,
        "z": 2,
    }


# ==========================================================
# Regression
# ==========================================================

def test_integer_ids_unique():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("abcdef")

    ids = list(tokenizer.stoi.values())

    assert len(ids) == len(set(ids))


def test_character_keys_unique():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("abcdef")

    chars = list(tokenizer.stoi.keys())

    assert len(chars) == len(set(chars))