import numpy as np

from mini_torch.tensors import tensor

from mini_torch.nn.linear import Linear
from mini_torch.nn.activations import ReLU
from mini_torch.nn.sequential import Sequential
from mini_torch.nn.losses import MSELoss

from mini_torch.optim.sgd import SGD
from mini_torch.autograd.engine import backward

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

np.random.seed(42)

# ---------------------------------------------------------
# XOR Dataset
# ---------------------------------------------------------

x = tensor(
    np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])
)

y = tensor(
    np.array([
        [0.],
        [1.],
        [1.],
        [0.]
    ])
)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = Sequential(
    Linear(2, 4),
    ReLU(),
    Linear(4, 1),
)

criterion = MSELoss()

optimizer = SGD(
    model.parameters(),
    lr=0.1,
)

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

epochs = 5000

for epoch in range(epochs):

    prediction = model(x)

    loss = criterion(prediction, y)

    optimizer.zero_grad()

    backward(loss)

    optimizer.step()

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:5d} | Loss = {loss.data:.6f}"
        )

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\nPredictions")

pred = model(x)

print(pred.data)

print("\nRounded Predictions")

print(np.round(pred.data))

print("\nTargets")

print(y.data)