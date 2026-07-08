import os
import tempfile

import pytest

from mini_torch.text.character_tokenizer import CharacterTokenizer


# ==========================================================
# state_dict()
# ==========================================================

def test_state_dict_keys():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("hello world")

    state = tokenizer.state_dict()

    assert set(state.keys()) == {
        "stoi",
        "itos",
        "vocab",
        "vocab_size",
    }


def test_state_dict_values():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("abc")

    state = tokenizer.state_dict()

    assert state["stoi"] == tokenizer.stoi
    assert state["itos"] == tokenizer.itos
    assert state["vocab"] == tokenizer.vocab
    assert state["vocab_size"] == tokenizer.vocab_size


def test_state_dict_returns_copy():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("abc")

    state = tokenizer.state_dict()

    state["stoi"]["x"] = 100
    state["itos"][100] = "x"
    state["vocab"].append("x")

    assert "x" not in tokenizer.stoi
    assert 100 not in tokenizer.itos
    assert tokenizer.vocab == ["a", "b", "c"]


# ==========================================================
# load_state_dict()
# ==========================================================

def test_load_state_dict():

    tokenizer1 = CharacterTokenizer()

    tokenizer1.fit("hello world")

    state = tokenizer1.state_dict()

    tokenizer2 = CharacterTokenizer()

    tokenizer2.load_state_dict(state)

    assert tokenizer2.stoi == tokenizer1.stoi
    assert tokenizer2.itos == tokenizer1.itos
    assert tokenizer2.vocab == tokenizer1.vocab
    assert tokenizer2.vocab_size == tokenizer1.vocab_size


def test_load_state_dict_overwrites():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("abc")

    tokenizer.load_state_dict(
        {
            "stoi": {"x": 0},
            "itos": {0: "x"},
            "vocab": ["x"],
            "vocab_size": 1,
        }
    )

    assert tokenizer.stoi == {"x": 0}
    assert tokenizer.itos == {0: "x"}
    assert tokenizer.vocab == ["x"]
    assert tokenizer.vocab_size == 1


def test_missing_state_key():

    tokenizer = CharacterTokenizer()

    with pytest.raises(KeyError):

        tokenizer.load_state_dict(
            {
                "stoi": {},
                "itos": {},
                "vocab": [],
            }
        )


# ==========================================================
# save() / load()
# ==========================================================

def test_save_and_load():

    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(
            tmpdir,
            "tokenizer.pkl",
        )

        tokenizer1 = CharacterTokenizer()

        tokenizer1.fit(
            "the quick brown fox"
        )

        tokenizer1.save(path)

        tokenizer2 = CharacterTokenizer()

        tokenizer2.load(path)

        assert tokenizer2.stoi == tokenizer1.stoi
        assert tokenizer2.itos == tokenizer1.itos
        assert tokenizer2.vocab == tokenizer1.vocab
        assert tokenizer2.vocab_size == tokenizer1.vocab_size


def test_save_creates_file():

    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(
            tmpdir,
            "tokenizer.pkl",
        )

        tokenizer = CharacterTokenizer()

        tokenizer.fit("abc")

        tokenizer.save(path)

        assert os.path.exists(path)


def test_load_missing_file():

    tokenizer = CharacterTokenizer()

    with pytest.raises(FileNotFoundError):

        tokenizer.load(
            "missing_tokenizer.pkl"
        )


# ==========================================================
# Round-trip after serialization
# ==========================================================

def test_encode_decode_after_load():

    text = "mini_torch"

    tokenizer = CharacterTokenizer()

    tokenizer.fit(text)

    encoded = tokenizer.encode(text)

    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(
            tmpdir,
            "tokenizer.pkl",
        )

        tokenizer.save(path)

        loaded = CharacterTokenizer()

        loaded.load(path)

        assert loaded.decode(encoded) == text


def test_loaded_tokenizer_matches_original():

    text = "hello world"

    tokenizer = CharacterTokenizer()

    tokenizer.fit(text)

    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(
            tmpdir,
            "tokenizer.pkl",
        )

        tokenizer.save(path)

        loaded = CharacterTokenizer()

        loaded.load(path)

        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.decode(
            tokenizer.encode(text)
        ) == text


# ==========================================================
# Regression
# ==========================================================

def test_multiple_saves():

    with tempfile.TemporaryDirectory() as tmpdir:

        path = os.path.join(
            tmpdir,
            "tokenizer.pkl",
        )

        tokenizer = CharacterTokenizer()

        tokenizer.fit("abc")

        tokenizer.save(path)

        tokenizer.fit("xyz")

        tokenizer.save(path)

        loaded = CharacterTokenizer()

        loaded.load(path)

        assert loaded.stoi == tokenizer.stoi
        assert loaded.itos == tokenizer.itos
        assert loaded.vocab == tokenizer.vocab


def test_state_dict_independent_after_load():

    tokenizer = CharacterTokenizer()

    tokenizer.fit("abc")

    state = tokenizer.state_dict()

    loaded = CharacterTokenizer()

    loaded.load_state_dict(state)

    loaded.stoi["z"] = 99

    assert "z" not in tokenizer.stoi