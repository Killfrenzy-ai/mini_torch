# mini_torch

<div align="center">

A lightweight deep learning framework built from scratch using **NumPy and CuPy**.

Designed to explore how modern deep learning frameworks, automatic differentiation systems, and Transformer-based language models work internally.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/version-v1.0.0-orange.svg)
![Tests](https://img.shields.io/badge/tests-330%2B%20passing-success.svg)

</div>

---

# Overview

**mini_torch** is an educational deep learning framework implemented from scratch using **NumPy**, with optional **CuPy-based CUDA acceleration**.

The project began as an experiment in building an automatic differentiation engine and gradually evolved into a neural network framework capable of training GPT-style Transformer language models end-to-end.

The framework now includes:

- A custom Tensor and autograd engine
- CPU and CUDA backends
- Neural network layers
- Optimizers and learning-rate schedulers
- Dataset and DataLoader abstractions
- Model and optimizer checkpointing
- Character-level and BPE tokenization
- GPT-style language models
- Modern Transformer components including RoPE, RMSNorm, SiLU, and SwiGLU
- Autoregressive text generation with modern sampling strategies

Unlike production frameworks, the major components are implemented manually to explore both **how they work** and **why modern deep learning systems are designed this way**.

---

# Features

## Automatic Differentiation

- Dynamic computation graphs
- Reverse-mode automatic differentiation
- Broadcasting-aware gradients
- Gradient accumulation
- Gradient propagation through tensor indexing
- Custom backward operations
- Gradient checking utilities

---

## Tensor Operations

The custom Tensor implementation supports operations including:

- Addition
- Subtraction
- Multiplication
- Division
- Power
- Matrix multiplication
- Exponential
- Logarithm
- Maximum
- Clip
- Sum
- Mean
- Tensor indexing
- Reshape
- Flatten
- Squeeze
- Unsqueeze
- Transpose
- Stack
- Sine
- Cosine

All supported differentiable operations integrate with the custom autograd engine.

---

## CPU and GPU Backend

`mini_torch` provides a backend abstraction capable of switching between:

- **NumPy** for CPU execution
- **CuPy** for CUDA-accelerated GPU execution

This allows the same model and tensor code to execute across CPU and GPU backends.

```python
from mini_torch.backend import use_gpu

use_gpu()
```

---

## Neural Network Modules

- `Module`
- `Parameter`
- `Sequential`
- `ModuleList`
- `Linear`
- `Embedding`
- Learned Positional Embeddings
- `LayerNorm`
- `RMSNorm`
- `Dropout`
- Feed-Forward Networks
- `SwiGLU`

The module system supports recursive parameter discovery and nested model architectures.

---

## Activation Functions

- ReLU
- Sigmoid
- Softmax
- SiLU

---

# Transformer Components

## Classic GPT Components

- Scaled Dot-Product Attention
- Multi-Head Self-Attention
- Causal Attention Masking
- Learned Positional Embeddings
- GPT-style Transformer Blocks
- GPT Decoder Architecture

---

## Modern Transformer Components

`mini_torch` now also includes components commonly found in more recent Transformer architectures.

### Rotary Positional Embeddings — RoPE

RoPE applies positional information directly to the Query and Key representations inside attention rather than adding learned positional embeddings to token representations.

Benefits include:

- Relative positional information inside attention
- No learned positional embedding table
- Position-aware Query/Key interactions

---

### RMSNorm

RMSNorm provides a simpler normalization mechanism based on the root mean square of activations.

Compared with LayerNorm, it removes the mean-centering operation while retaining activation normalization.

---

### SiLU

The SiLU activation function is defined as:

```text
SiLU(x) = x × sigmoid(x)
```

It provides a smooth, differentiable activation used as the gating function inside SwiGLU.

---

### SwiGLU

The modern feed-forward network uses a gated architecture:

```text
gate = SiLU(W_gate(x))
up   = W_up(x)

hidden = gate × up

output = W_down(hidden)
```

This allows the network to learn which transformed features should pass through the feed-forward layer.

---

### Modern Multi-Head Attention

The modern attention implementation integrates RoPE directly into each attention head:

```text
Input
  │
  ├── Query Projection ──→ RoPE ──┐
  ├── Key Projection ────→ RoPE ──┼──→ Attention
  └── Value Projection ────────────┘
```

Gradients propagate through the complete:

```text
RoPE
 ↓
Attention
 ↓
Q/K Projections
 ↓
Transformer Block
```

pipeline using the custom autograd engine.

---

# Language Models

## GPT

The original GPT implementation uses:

```text
Token Embeddings
        +
Learned Positional Embeddings
        ↓
Transformer Blocks
        ↓
LayerNorm
        ↓
Language Model Head
```

---

## ModernGPT

The newer architecture introduces several modern Transformer components:

```text
BPE Tokens
    ↓
Token Embeddings
    ↓
Modern Transformer Blocks
    │
    ├── RMSNorm
    ├── RoPE Self-Attention
    ├── Residual Connection
    ├── RMSNorm
    ├── SwiGLU
    └── Residual Connection
    ↓
Final RMSNorm
    ↓
Language Model Head
    ↓
Next-Token Logits
```

Compared with the original GPT implementation:

| Original GPT | ModernGPT |
|---|---|
| Learned positional embeddings | Rotary Positional Embeddings |
| LayerNorm | RMSNorm |
| Standard Feed-Forward Network | SwiGLU |
| Traditional attention | RoPE-enhanced attention |

The two architectures can be trained at similar parameter counts to experimentally compare convergence, validation perplexity, generation quality, and training throughput.

---

# Tokenization

## Character Tokenizer

The character-level tokenizer provides:

- Vocabulary construction
- Text encoding
- Token decoding
- Tokenizer serialization
- Checkpoint loading

---

## Byte Pair Encoding — BPE

`mini_torch` includes a BPE tokenizer implemented from scratch.

It supports:

- Corpus construction
- Pair-frequency counting
- Iterative BPE merge learning
- Configurable vocabulary size
- Text encoding
- Text decoding
- Learned merge replay
- Vocabulary serialization
- Tokenizer checkpointing

Example:

```python
from mini_torch.text.bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer()

tokenizer.fit(
    text,
    vocab_size=512,
)

tokens = tokenizer.encode(text)

text = tokenizer.decode(tokens)

tokenizer.save(
    "checkpoints/bpe_tokenizer.pkl"
)
```

---

# Training Features

The language-model training pipeline currently supports:

- Mini-batch training
- GPU training
- Gradient accumulation
- Gradient clipping
- Weight tying
- Train/validation splits
- Validation loss
- Training perplexity
- Validation perplexity
- Learning-rate warmup
- Cosine annealing
- Model checkpointing
- Optimizer checkpointing
- Scheduler checkpointing
- Training resumption

---

## Gradient Accumulation

Gradient accumulation allows the framework to simulate larger effective batch sizes when GPU memory is limited.

```text
Physical batch size     = 32
Accumulation steps      = 4
Effective batch size    = 128
```

---

## Gradient Clipping

Global gradient norm clipping is supported to improve training stability:

```python
grad_norm = clip_grad_norm(
    model.parameters(),
    max_norm=1.0,
)
```

---

# Optimizers

- SGD
- Adam
- AdamW

## AdamW

AdamW provides decoupled weight decay, allowing regularization to be applied independently from Adam's adaptive gradient update.

Optimizer state can also be checkpointed and restored for continued training.

---

# Learning Rate Scheduling

## Cosine Annealing

The framework includes cosine learning-rate decay with optional linear warmup.

```text
Warmup
   ↓
Peak Learning Rate
   ↓
Cosine Decay
   ↓
Minimum Learning Rate
```

Scheduler state can be saved and restored to resume interrupted training runs.

---

# Loss Functions

- Mean Squared Error — MSE
- Binary Cross Entropy — BCE
- Cross Entropy Loss

---

# Weight Initialization

- Xavier Uniform
- Xavier Normal
- He Uniform
- He Normal

---

# Data Utilities

- `Dataset`
- `TensorDataset`
- `TextDataset`
- `DataLoader`

The language-model pipeline supports automatic creation of input/target sequences for next-token prediction.

---

# Text Generation

`mini_torch` supports autoregressive text generation using:

- Temperature scaling
- Top-k sampling
- Top-p / nucleus sampling

Example generation pipeline:

```text
Prompt
  ↓
BPE Encoding
  ↓
GPT Forward Pass
  ↓
Temperature Scaling
  ↓
Top-k Filtering
  ↓
Top-p Filtering
  ↓
Sampling
  ↓
Next Token
  ↓
Repeat
```

---

# Model and Training Checkpointing

The framework supports serialization for:

- Model parameters
- Optimizer state
- Learning-rate scheduler state
- Tokenizer vocabulary and merge rules

This allows training to resume while preserving:

```text
Model weights
Optimizer moment estimates
Optimizer step state
Scheduler progress
Tokenizer vocabulary
BPE merge rules
```

---

# Project Structure

```text
mini_torch/
│
├── autograd/
│
├── data/
│   ├── dataloader.py
│   └── ...
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
│   ├── rmsnorm.py
│   ├── rotary_embedding.py
│   ├── sequential.py
│   ├── swiglu.py
│   ├── modern_attention.py
│   ├── modern_transformer_block.py
│   ├── modern_gpt.py
│   └── transformer_block.py
│
├── optim/
│   ├── optimizer.py
│   ├── sgd.py
│   ├── adam.py
│   └── adamW.py
│
├── scheduler/
│   ├── LRScheduler.py
│   └── cosine_annealing_lr.py
│
├── text/
│   ├── tokenizer.py
│   ├── character_tokenizer.py
│   ├── bpe_tokenizer.py
│   └── text_dataset.py
│
├── tests/
│
└── examples/
    ├── train_gpt.py
    ├── train_modern_gpt.py
    └── ...
```

---

# Examples

The framework includes examples covering both traditional neural networks and Transformer language models.

| Example | Description |
|---|---|
| Linear Regression | Basic regression |
| XOR with MSE | Binary learning experiment |
| XOR with BCE | Binary classification |
| Iris Classification | Multi-class classification |
| GPT Training | Train a GPT-style language model |
| ModernGPT Training | Train the modern Transformer architecture |
| Text Generation | Generate text from trained checkpoints |

Basic neural network example:

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

x = tensor(
    np.random.randn(32, 4)
)

prediction = model(x)
```

---

# Testing

The project contains an expanding automated test suite covering:

- Tensor operations
- Automatic differentiation
- Broadcasting
- Matrix multiplication
- Tensor indexing
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
- RoPE
- RMSNorm
- SiLU
- SwiGLU
- Modern Transformer components

Run the complete suite:

```bash
pytest
```

---

# Roadmap

## Framework

- [x] Tensor implementation
- [x] Automatic differentiation
- [x] Neural network module system
- [x] CPU backend
- [x] CUDA backend
- [x] Optimizers
- [x] Serialization
- [x] Data pipeline
- [x] Gradient accumulation
- [x] Gradient clipping
- [x] Learning-rate schedulers
- [x] Optimizer checkpointing
- [x] Scheduler checkpointing

---

## Language Modeling

- [x] Character tokenizer
- [x] BPE tokenizer
- [x] Vocabulary construction
- [x] Text dataset
- [x] GPT training pipeline
- [x] Tiny Shakespeare training
- [x] Train/validation split
- [x] Validation loss
- [x] Perplexity evaluation
- [x] Weight tying
- [x] Temperature sampling
- [x] Top-k sampling
- [x] Top-p sampling
- [x] Text generation

---

## Modern Transformer Architecture

- [x] SiLU
- [x] RMSNorm
- [x] SwiGLU
- [x] Rotary Positional Embeddings
- [x] RoPE Multi-Head Attention
- [x] Modern Transformer Block
- [x] ModernGPT
- [ ] ModernGPT training and evaluation
- [ ] Architecture benchmarking

---

## Future Improvements

- [ ] Numerically stable logits-based Cross Entropy
- [ ] GELU activation
- [ ] Mixed precision training
- [ ] KV cache for faster inference
- [ ] Improved BPE training performance
- [ ] Parameter counting utilities
- [ ] Training throughput metrics
- [ ] Additional datasets
- [ ] Knowledge distillation
- [ ] `no_grad()` execution mode
- [ ] Vision Transformer experiments

---

# Current Experiments

The project is currently being used to compare two Transformer architectures at similar model sizes.

### Original GPT

```text
Learned Positional Embeddings
LayerNorm
Standard Feed-Forward Network
Multi-Head Self-Attention
```

### ModernGPT

```text
Rotary Positional Embeddings
RMSNorm
SwiGLU
RoPE Multi-Head Self-Attention
```

The experiments compare:

- Training loss
- Validation loss
- Perplexity
- Convergence behaviour
- Text generation quality
- Parameter count
- Training throughput

---

# Design Goals

The primary objective of `mini_torch` is educational experimentation.

Every major component—including the tensor system, autograd engine, optimizers, neural network layers, attention mechanisms, tokenizers, training pipeline, and Transformer architectures—is implemented manually without relying on existing deep learning frameworks.

The focus is not on competing with highly optimized production frameworks.

Instead, the goal is to understand the complete path:

```text
Tensor Operations
      ↓
Automatic Differentiation
      ↓
Neural Network Layers
      ↓
Attention
      ↓
Transformer Blocks
      ↓
Language Models
      ↓
Training and Optimization
      ↓
Text Generation
```

By implementing each layer of this stack directly, `mini_torch` provides a playground for understanding and experimenting with the internal mechanics of modern deep learning systems.

---

# Inspiration

This project draws inspiration from:

- PyTorch
- micrograd
- tinygrad
- nanoGPT
- *Attention Is All You Need*
- GPT architectures
- Modern Transformer architectures

All core implementations are written independently as educational exercises.

---

# License

Licensed under the Apache License 2.0.