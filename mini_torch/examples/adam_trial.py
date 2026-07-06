from mini_torch.nn.sequential import Sequential
from mini_torch.nn.linear import Linear
from mini_torch.optim.adam import Adam

model = Sequential(
    Linear(2, 4),
    Linear(4, 1),
)

optimizer = Adam(model.parameters())

print(len(optimizer.state))