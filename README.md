# mini_torch

<div align="center">

A lightweight deep learning framework built completely from scratch using **NumPy**.

Designed to understand how modern deep learning frameworks and Large Language Models work internally.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/version-v0.6.0-orange.svg)
![Tests](https://img.shields.io/badge/tests-220%2B%20passing-success.svg)

</div>

---

# Why mini_torch?

Most developers learn deep learning by importing PyTorch or TensorFlow.

Very few understand what happens underneath.

**mini_torch** is my attempt to build a modern deep learning framework completely from scratch using only **NumPy**, implementing every major component myself—from automatic differentiation to neural network layers and, eventually, a GPT-style Small Language Model.

The goal is not to compete with PyTorch.

The goal is to understand it.

---

# Current Features

## Automatic Differentiation

- Dynamic computation graph
- Reverse-mode automatic differentiation
- Broadcasting-aware gradients
- Gradient accumulation
- Gradient checking utilities

---

## Tensor Operations

- Addition
- Subtraction
- Multiplication
- Division
- Power
- Matrix Multiplication
- Exponential
- Logarithm
- Maximum
- Clip
- Sum
- Mean
- Tensor Indexing
- Reshape
- Transpose

---

## Neural Network Layers

- Linear
- Sequential
- Embedding
- LayerNorm
- Dropout

---

## Activation Functions

- ReLU
- Sigmoid
- Softmax

---

## Loss Functions

- Mean Squared Error (MSE)
- Binary Cross Entropy (BCE)
- Cross Entropy

---

## Optimizers

- Stochastic Gradient Descent (SGD)
- Adam

---

## Weight Initialization

- Xavier Uniform
- Xavier Normal
- He Uniform
- He Normal

---

## Data Pipeline

- Dataset
- TensorDataset
- DataLoader

---

## Model Utilities

- Parameter registration
- Recursive module hierarchy
- `train()`
- `eval()`
- Serialization
- `state_dict()`
- `save()`
- `load()`

---

# Project Structure

```text
mini_torch/
│
├── autograd/
│   ├── engine.py
│   ├── operation.py
│   └── operations/
│
├── data/
│   ├── dataset.py
│   ├── tensor_dataset.py
│   └── dataloader.py
│
├── nn/
│   ├── activations.py
│   ├── dropout.py
│   ├── embedding.py
│   ├── init.py
│   ├── layernorm.py
│   ├── linear.py
│   ├── losses.py
│   ├── module.py
│   └── sequential.py
│
├── optim/
│
├── tests/
│
└── examples/
```

---

# Examples

The framework currently includes complete working examples.

| Example | Description |
|----------|-------------|
| Linear Regression | First regression model |
| XOR (MSE) | XOR classification using MSE |
| XOR (BCE) | XOR classification using Binary Cross Entropy |
| Iris Classification | Multi-class classification with Softmax + CrossEntropy |

Example:

```python
import numpy as np

from mini_torch.tensors import tensor

from mini_torch.nn.linear import Linear
from mini_torch.nn.activations import ReLU
from mini_torch.nn.sequential import Sequential

model = Sequential(
    Linear(4, 16),
    ReLU(),
    Linear(16, 3),
)

x = tensor(np.random.randn(32, 4))

prediction = model(x)
```

---

# Testing

The project is heavily tested.

Current test coverage includes:

- Tensor operations
- Autograd engine
- Activations
- Loss functions
- Optimizers
- Layers
- Serialization
- Data loading

```
220+ Passing Tests
```

Run the test suite:

```bash
pytest
```

---

# Development Roadmap

## Framework Core

- [x] Tensor implementation
- [x] Automatic differentiation
- [x] Broadcasting support
- [x] Neural network modules
- [x] Optimizers
- [x] Weight initialization
- [x] Data pipeline
- [x] Serialization

---

## Transformer Components

- [x] Embedding
- [x] LayerNorm
- [x] Dropout
- [ ] Positional Embeddings
- [ ] Scaled Dot Product Attention
- [ ] Multi-Head Attention
- [ ] Feed Forward Network
- [ ] Transformer Block

---

## Language Model

- [ ] GPT Decoder
- [ ] Tokenizer
- [ ] Vocabulary
- [ ] Text Dataset
- [ ] TinyGPT Training
- [ ] Text Generation

---

# Learning Goals

This project explores the implementation of:

- Automatic Differentiation
- Computational Graphs
- Backpropagation
- Neural Network Training
- Optimization Algorithms
- Weight Initialization
- Transformer Architecture
- Language Model Training

Everything is implemented from first principles without relying on existing deep learning frameworks.

---

# Inspiration

This project draws inspiration from:

- PyTorch
- micrograd
- tinygrad
- The Transformer paper ("Attention Is All You Need")
- GPT architecture

while implementing every component independently as an educational exercise.

---

# Future Vision

The long-term objective is to evolve **mini_torch** into a framework capable of training a small decoder-only Transformer language model completely from scratch.

Planned milestones include:

- Learned positional embeddings
- Multi-head self-attention
- Transformer decoder blocks
- Byte Pair Encoding (BPE) tokenizer
- TinyGPT training on a custom corpus
- Text generation

---

# Contributing

Suggestions, bug reports and discussions are welcome.

Feel free to open an issue or submit a pull request.

---

# License

This project is licensed under the **Apache License 2.0**.

See the `LICENSE` file for details.