# mini_torch

<div align="center">

A lightweight deep learning framework built completely from scratch using **NumPy**.

Designed to understand how modern deep learning frameworks and GPT-style language models work internally.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/version-v1.0.0-orange.svg)
![Tests](https://img.shields.io/badge/tests-330%2B%20passing-success.svg)

</div>

---

# Overview

**mini_torch** is an educational deep learning framework implemented entirely from scratch using **NumPy**.

The project began as an implementation of automatic differentiation and gradually evolved into a complete neural network framework capable of building and training modern Transformer-based language models.

Unlike production frameworks, every major component is implemented manually to provide a clear understanding of how deep learning systems work internally.

---

# Features

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

## Neural Network Modules

- Module
- Sequential
- ModuleList
- Parameter
- Linear
- Embedding
- Learned Positional Embedding
- LayerNorm
- Dropout
- FeedForward

---

## Transformer Components

- Scaled Dot Product Attention
- Multi-Head Attention
- GPT-style Transformer Block
- Causal Masking
- GPT Decoder Architecture

---

## Activation Functions

- ReLU
- Sigmoid
- Softmax

---

## Loss Functions

- Mean Squared Error (MSE)
- Binary Cross Entropy (BCE)
- Cross Entropy Loss

---

## Optimizers

- SGD
- Adam

---

## Weight Initialization

- Xavier Uniform
- Xavier Normal
- He Uniform
- He Normal

---

## Data Utilities

- Dataset
- TensorDataset
- DataLoader

---

## Model Utilities

- Recursive module hierarchy
- Parameter registration
- Model serialization
- `state_dict()`
- `load_state_dict()`
- `save()`
- `load()`
- `train()`
- `eval()`

---

# Project Structure

```text
mini_torch/
│
├── autograd/
│
├── data/
│
├── nn/
│   ├── activations.py
│   ├── attention.py
│   ├── dropout.py
│   ├── embedding.py
│   ├── feedforward.py
│   ├── functional.py
│   ├── gpt.py
│   ├── init.py
│   ├── layernorm.py
│   ├── linear.py
│   ├── losses.py
│   ├── module.py
│   ├── module_list.py
│   ├── multihead_attention.py
│   ├── position.py
│   ├── sequential.py
│   └── transformer_block.py
│
├── optim/
│
├── tests/
│
└── examples/
```

---

# Examples

Current examples included with the framework:

| Example | Description |
|----------|-------------|
| Linear Regression | Simple regression |
| XOR (MSE) | Binary classification |
| XOR (BCE) | Binary classification using BCE |
| Iris Classification | Multi-class classification |

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

The project currently contains **330+ automated unit tests** covering:

- Tensor operations
- Automatic differentiation
- Broadcasting
- Matrix multiplication
- Activations
- Loss functions
- Optimizers
- Weight initialization
- Neural network layers
- Serialization
- Dataset utilities
- Attention
- Multi-head attention
- Transformer blocks
- GPT architecture

Run the complete suite:

```bash
pytest
```

---

# Roadmap

## Framework

- [x] Tensor
- [x] Automatic Differentiation
- [x] Neural Network Modules
- [x] Optimizers
- [x] Serialization
- [x] Data Pipeline
- [x] Transformer Components
- [x] GPT Architecture

---

## Language Model

- [ ] Character Tokenizer
- [ ] Vocabulary
- [ ] Text Dataset
- [ ] GPT Training Pipeline
- [ ] Tiny Shakespeare Training
- [ ] Text Generation

---

## Future Improvements

- [ ] GELU activation
- [ ] Weight tying
- [ ] Learning rate schedulers
- [ ] Gradient clipping
- [ ] Byte Pair Encoding (BPE)
- [ ] Top-k sampling
- [ ] Top-p sampling
- [ ] Rotary Positional Embeddings (RoPE)
- [ ] KV Cache for inference

---

# Design Goals

The objective of this project is educational.

Every major component—including the autograd engine, optimizers, neural network layers, attention mechanism and GPT architecture—is implemented manually without relying on existing deep learning frameworks.

The focus is on understanding the internal mechanics of modern deep learning systems rather than maximizing performance.

---

# Inspiration

This project draws inspiration from:

- PyTorch
- micrograd
- tinygrad
- nanoGPT
- The Transformer paper ("Attention Is All You Need")
- GPT-2 architecture

All implementations are written independently as educational exercises.

---

# License

Licensed under the Apache License 2.0.