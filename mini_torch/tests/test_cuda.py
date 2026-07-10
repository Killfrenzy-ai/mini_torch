import cupy as cp

from mini_torch.backend import use_gpu
from mini_torch.tensors import tensor
from mini_torch.autograd.engine import backward
from mini_torch.nn.linear import Linear
from mini_torch.optim.adam import Adam


def test_cuda_linear():

    use_gpu()

    model = Linear(16, 8)

    model.cuda()

    x = tensor(
        cp.random.randn(4, 16)
    )

    y = model(x)

    loss = y.sum()

    backward(loss)

    assert isinstance(
        y.data,
        cp.ndarray,
    )

    assert isinstance(
        model.weight.grad,
        cp.ndarray,
    )

def test_cuda_adam():

    use_gpu()

    model = Linear(16, 8)

    model.cuda()

    optimizer = Adam(
        model.parameters(),
        lr=1e-3,
    )

    x = tensor(cp.random.randn(4, 16))

    y = model(x)

    loss = y.sum()

    loss.backward()

    before = model.weight.data.copy()

    optimizer.step()

    assert cp.any(before != model.weight.data)