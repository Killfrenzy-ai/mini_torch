import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mini_torch.tensors import tensor

from mini_torch.data import (
    TensorDataset,
    DataLoader,
)

from mini_torch.nn.linear import Linear
from mini_torch.nn.activations import ReLU, Softmax
from mini_torch.nn.sequential import Sequential
from mini_torch.nn.losses import CrossEntropyLoss

from mini_torch.optim.sgd import SGD


# ==========================================================
# Reproducibility
# ==========================================================

np.random.seed(42)


# ==========================================================
# Load Iris Dataset
# ==========================================================

iris = load_iris()

x = iris.data.astype(np.float64)

labels = iris.target


# ==========================================================
# One-Hot Encode Targets
# ==========================================================

num_classes = 3

y = np.eye(num_classes)[labels]


# ==========================================================
# Train/Test Split
# ==========================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)


# ==========================================================
# Convert to Tensors
# ==========================================================

x_train = tensor(x_train)
y_train = tensor(y_train)

x_test = tensor(x_test)
y_test = tensor(y_test)


# ==========================================================
# Dataset / DataLoader
# ==========================================================

train_dataset = TensorDataset(
    x_train,
    y_train,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
)


# ==========================================================
# Model
# ==========================================================

model = Sequential(
    Linear(4, 16),
    ReLU(),
    Linear(16, 3),
    Softmax(),
)


criterion = CrossEntropyLoss()

optimizer = SGD(
    model.parameters(),
    lr=0.05,
)


# ==========================================================
# Training
# ==========================================================

epochs = 200

for epoch in range(epochs):

    epoch_loss = 0.0

    for batch_x, batch_y in train_loader:

        prediction = model(batch_x)

        loss = criterion(
            prediction,
            batch_y,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.data

    if epoch % 20 == 0:

        print(
            f"Epoch {epoch:3d}"
            f" | Loss = {epoch_loss / len(train_loader):.6f}"
        )


# ==========================================================
# Evaluation
# ==========================================================

prediction = model(x_test)

predicted_classes = prediction.data.argmax(axis=1)

target_classes = y_test.data.argmax(axis=1)

accuracy = (
    predicted_classes == target_classes
).mean()

print()

print("=" * 50)

print(f"Test Accuracy : {accuracy * 100:.2f}%")

print("=" * 50)

print()

print("Predictions")

print(predicted_classes)

print()

print("Targets")

print(target_classes)