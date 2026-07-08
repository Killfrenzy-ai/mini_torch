import numpy as np
import pytest

from mini_torch.text.text_dataset import TextDataset
from mini_torch.tensors import tensor


# ==========================================================
# Constructor
# ==========================================================

def test_constructor():

    tokens = np.arange(20)

    dataset = TextDataset(
        tokens,
        context_length=5,
    )

    assert dataset.context_length == 5
    assert np.array_equal(dataset.tokens, tokens)


def test_invalid_context_length_zero():

    with pytest.raises(ValueError):
        TextDataset(
            np.arange(10),
            context_length=0,
        )


def test_invalid_context_length_negative():

    with pytest.raises(ValueError):
        TextDataset(
            np.arange(10),
            context_length=-5,
        )


def test_context_longer_than_tokens():

    with pytest.raises(ValueError):
        TextDataset(
            np.arange(5),
            context_length=10,
        )


def test_context_equal_tokens():

    with pytest.raises(ValueError):
        TextDataset(
            np.arange(5),
            context_length=5,
        )


# ==========================================================
# Length
# ==========================================================

def test_length():

    dataset = TextDataset(
        np.arange(100),
        context_length=10,
    )

    assert len(dataset) == 90


def test_length_small():

    dataset = TextDataset(
        np.arange(6),
        context_length=5,
    )

    assert len(dataset) == 1


# ==========================================================
# __getitem__
# ==========================================================

def test_first_sample():

    dataset = TextDataset(
        np.arange(10),
        context_length=4,
    )

    x, y = dataset[0]

    assert np.array_equal(
        x.data,
        np.array([0, 1, 2, 3]),
    )

    assert np.array_equal(
        y.data,
        np.array([1, 2, 3, 4]),
    )


def test_middle_sample():

    dataset = TextDataset(
        np.arange(10),
        context_length=4,
    )

    x, y = dataset[3]

    assert np.array_equal(
        x.data,
        np.array([3, 4, 5, 6]),
    )

    assert np.array_equal(
        y.data,
        np.array([4, 5, 6, 7]),
    )


def test_last_sample():

    dataset = TextDataset(
        np.arange(10),
        context_length=4,
    )

    x, y = dataset[len(dataset) - 1]

    assert np.array_equal(
        x.data,
        np.array([5, 6, 7, 8]),
    )

    assert np.array_equal(
        y.data,
        np.array([6, 7, 8, 9]),
    )


# ==========================================================
# Tensor Properties
# ==========================================================

def test_returns_tensor():

    dataset = TextDataset(
        np.arange(20),
        context_length=5,
    )

    x, y = dataset[0]

    assert isinstance(x, tensor)
    assert isinstance(y, tensor)


def test_shapes():

    dataset = TextDataset(
        np.arange(20),
        context_length=8,
    )

    x, y = dataset[0]

    assert x.shape == (8,)
    assert y.shape == (8,)


def test_dtype():

    dataset = TextDataset(
        np.arange(20),
        context_length=6,
    )

    x, y = dataset[0]

    assert x.dtype == dataset.tokens.dtype
    assert y.dtype == dataset.tokens.dtype


# ==========================================================
# Boundary Conditions
# ==========================================================

def test_index_out_of_range_positive():

    dataset = TextDataset(
        np.arange(20),
        context_length=5,
    )

    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_index_out_of_range_large():

    dataset = TextDataset(
        np.arange(20),
        context_length=5,
    )

    with pytest.raises(IndexError):
        dataset[100]


def test_negative_index():

    dataset = TextDataset(
        np.arange(20),
        context_length=5,
    )

    with pytest.raises(IndexError):
        dataset[-1]


# ==========================================================
# Sliding Window
# ==========================================================

def test_shift_by_one():

    dataset = TextDataset(
        np.arange(30),
        context_length=7,
    )

    x, y = dataset[5]

    assert np.array_equal(
        y.data[:-1],
        x.data[1:],
    )


def test_multiple_context_lengths():

    for context in [2, 4, 8, 16]:

        tokens = np.arange(50)

        dataset = TextDataset(
            tokens,
            context_length=context,
        )

        x, y = dataset[0]

        assert len(x.data) == context
        assert len(y.data) == context


# ==========================================================
# Regression
# ==========================================================

def test_dataset_does_not_modify_tokens():

    tokens = np.arange(20)

    original = tokens.copy()

    dataset = TextDataset(
        tokens,
        context_length=5,
    )

    dataset[3]

    assert np.array_equal(
        tokens,
        original,
    )


def test_all_samples_valid():

    dataset = TextDataset(
        np.arange(25),
        context_length=6,
    )

    for i in range(len(dataset)):

        x, y = dataset[i]

        assert len(x.data) == 6
        assert len(y.data) == 6

        assert np.array_equal(
            y.data[:-1],
            x.data[1:],
        )