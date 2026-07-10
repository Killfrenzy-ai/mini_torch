import pytest

from mini_torch.backend import use_cpu, use_gpu


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
    )


@pytest.fixture(autouse=True)
def backend(request):

    device = request.config.getoption("--device")

    if device == "cuda":
        use_gpu()
    else:
        use_cpu()