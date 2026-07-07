# Changelog

All notable changes to **mini_torch** will be documented in this file.

This project follows Semantic Versioning.

---

# v0.6.0

## Added

### Neural Network Layers
- Embedding layer
- Layer Normalization (LayerNorm)
- Dropout (Inverted Dropout)

### Module System
- `train()` mode
- `eval()` mode
- Recursive module traversal
- `modules()` iterator

### Tensor Operations
- Tensor indexing with autograd support
- Gather-style indexing operation

### Optimizers
- Adam optimizer

### Weight Initialization
- Xavier Uniform
- Xavier Normal
- He Uniform
- He Normal

### Model Serialization
- `state_dict()`
- `load_state_dict()`
- `save()`
- `load()`

## Improved

- Modular neural network architecture
- Recursive parameter registration
- Recursive module registration
- Improved autograd graph construction
- Improved optimizer abstraction
- Expanded unit test coverage

---

# v0.5.0

## Added

### Data Utilities
- Dataset abstraction
- TensorDataset
- DataLoader

### Activations
- Softmax

### Losses
- CrossEntropyLoss

### Tensor Operations
- Max reduction with autograd
- Generalized Sum reduction
- Generalized Mean reduction

### Examples
- Iris Classification

## Improved

- Generalized reduction operations
- Expanded testing

---

# v0.4.0

## Added

### Neural Networks
- Sequential container
- Linear layer
- Parameter abstraction

### Optimizers
- SGD optimizer

### Examples
- Linear Regression
- XOR using MSE Loss

## Improved

- Module API
- Parameter management

---

# v0.3.0

## Added

- Sigmoid activation
- BCELoss
- Clip operation
- `tensor.backward()`
- XOR example using BCE Loss

## Improved

- Tensor internals refactor
- Centralized graph construction
- Reduced code duplication

---

# v0.2.0

## Added

### Core Autograd
- Tensor class
- Automatic differentiation engine
- Computational graph
- Broadcasting support
- Matrix multiplication
- Arithmetic operations
- ReLU activation
- MSE Loss

## Improved

- Initial project architecture