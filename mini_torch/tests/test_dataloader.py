import numpy as np
import pytest

from mini_torch.tensors import tensor
from mini_torch.data import TensorDataset, DataLoader


# ==========================================================
# CONSTRUCTION
# ==========================================================

def test_dataset_length():
    x = tensor(np.random.randn(10, 2))
    y = tensor(np.random.randn(10, 1))

    dataset = TensorDataset(x, y)

    assert len(dataset) == 10


def test_loader_length():
    x = tensor(np.random.randn(10, 2))
    y = tensor(np.random.randn(10, 1))

    dataset = TensorDataset(x, y)

    loader = DataLoader(
        dataset,
        batch_size=4,
    )

    assert len(loader) == 3


def test_invalid_batch_size():
    x = tensor(np.random.randn(5, 2))
    y = tensor(np.random.randn(5, 1))

    dataset = TensorDataset(x, y)

    with pytest.raises(ValueError):
        DataLoader(
            dataset,
            batch_size=0,
        )


# ==========================================================
# TENSORDATASET
# ==========================================================

def test_dataset_getitem():
    x = tensor(np.arange(10).reshape(5, 2))
    y = tensor(np.arange(5).reshape(5, 1))

    dataset = TensorDataset(x, y)

    sample_x, sample_y = dataset[2]

    assert np.allclose(sample_x.data, x.data[2])
    assert np.allclose(sample_y.data, y.data[2])


def test_dataset_multiple_tensors():
    x = tensor(np.random.randn(6, 2))
    y = tensor(np.random.randn(6, 1))
    z = tensor(np.random.randn(6, 4))

    dataset = TensorDataset(x, y, z)

    sample = dataset[3]

    assert len(sample) == 3


# ==========================================================
# BATCHING
# ==========================================================

def test_single_batch():
    x = tensor(np.random.randn(4, 2))
    y = tensor(np.random.randn(4, 1))

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=4,
    )

    batch_x, batch_y = next(iter(loader))

    assert batch_x.shape == (4, 2)
    assert batch_y.shape == (4, 1)


def test_multiple_batches():
    x = tensor(np.random.randn(10, 2))
    y = tensor(np.random.randn(10, 1))

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=3,
    )

    batches = list(loader)

    assert len(batches) == 4


def test_last_partial_batch():
    x = tensor(np.random.randn(10, 2))
    y = tensor(np.random.randn(10, 1))

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=4,
    )

    batches = list(loader)

    batch_x, batch_y = batches[-1]

    assert batch_x.shape == (2, 2)
    assert batch_y.shape == (2, 1)


# ==========================================================
# ITERATION
# ==========================================================

def test_multiple_epochs():
    x = tensor(np.random.randn(8, 2))
    y = tensor(np.random.randn(8, 1))

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    epoch1 = list(loader)
    epoch2 = list(loader)

    assert len(epoch1) == len(epoch2) == 4


def test_stop_iteration():
    x = tensor(np.random.randn(4, 2))
    y = tensor(np.random.randn(4, 1))

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    iterator = iter(loader)

    next(iterator)
    next(iterator)

    with pytest.raises(StopIteration):
        next(iterator)


# ==========================================================
# SHUFFLING
# ==========================================================

def test_shuffle_preserves_all_samples():
    np.random.seed(42)

    x = tensor(np.arange(20).reshape(10, 2))
    y = tensor(np.arange(10).reshape(10, 1))

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
        shuffle=True,
    )

    collected = []

    for batch_x, _ in loader:
        collected.extend(batch_x.data.tolist())

    collected = np.array(collected)

    expected = x.data

    assert sorted(map(tuple, collected)) == sorted(map(tuple, expected))


# ==========================================================
# REQUIRES_GRAD
# ==========================================================

def test_requires_grad_preserved():
    x = tensor(
        np.random.randn(5, 2),
        requires_grad=True,
    )

    y = tensor(
        np.random.randn(5, 1),
    )

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    batch_x, batch_y = next(iter(loader))

    assert batch_x.requires_grad is True
    assert batch_y.requires_grad is False


# ==========================================================
# EDGE CASES
# ==========================================================

def test_batch_size_larger_than_dataset():
    x = tensor(np.random.randn(5, 2))
    y = tensor(np.random.randn(5, 1))

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=32,
    )

    batches = list(loader)

    assert len(batches) == 1

    batch_x, batch_y = batches[0]

    assert batch_x.shape == (5, 2)
    assert batch_y.shape == (5, 1)


def test_batch_size_one():
    x = tensor(np.random.randn(6, 2))
    y = tensor(np.random.randn(6, 1))

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=1,
    )

    batches = list(loader)

    assert len(batches) == 6


# ==========================================================
# REGRESSION TESTS
# ==========================================================

def test_batch_shapes_consistent():
    x = tensor(np.random.randn(15, 3))
    y = tensor(np.random.randn(15, 2))

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=5,
    )

    for batch_x, batch_y in loader:

        assert batch_x.shape[1] == 3
        assert batch_y.shape[1] == 2