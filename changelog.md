# Changelog

All notable changes to **mini_torch** will be documented in this file.

This project follows Semantic Versioning.

---

# v1.1.0

## Added

### Modern Transformer Architecture

- Rotary Positional Embeddings (RoPE)
- RMSNorm
- SiLU activation
- SwiGLU feed-forward networks
- RoPE-enabled Multi-Head Self-Attention
- Modern pre-normalized Transformer blocks
- ModernGPT decoder architecture
- Bias-free attention projections

### Tokenization

- Byte Pair Encoding (BPE) tokenizer
- BPE vocabulary learning
- Iterative pair-frequency-based merge learning
- BPE encoding and decoding
- Learned merge replay
- Configurable BPE vocabulary size
- Tokenizer serialization and checkpoint loading

### Training

- Train/validation dataset splitting
- Validation loss evaluation
- Training perplexity tracking
- Validation perplexity tracking
- Gradient accumulation
- Global gradient norm clipping
- Weight tying
- Learning-rate warmup
- Cosine annealing learning-rate scheduler
- Continued training from model checkpoints

### Optimizers

- AdamW optimizer
- Decoupled weight decay
- Optimizer state checkpointing
- Optimizer state restoration

### Learning Rate Scheduling

- Learning-rate scheduler abstraction
- Cosine annealing scheduler
- Linear learning-rate warmup
- Scheduler state serialization
- Scheduler checkpoint restoration

### Text Generation

- Temperature-based sampling
- Top-k sampling
- Top-p / nucleus sampling
- Autoregressive text generation using trained GPT checkpoints
- BPE-based text generation

### GPU Acceleration

- NumPy/CuPy backend abstraction
- CUDA-based tensor execution
- GPU model execution
- GPU-accelerated language-model training
- CPU/GPU tensor transfer utilities

### Tensor Operations

- Stack operation with autograd support
- Sine operation with autograd support
- Cosine operation with autograd support
- Additional tensor shape manipulation utilities

### Model Architectures

Two GPT-style architectures are now supported:

#### Original GPT

- Learned positional embeddings
- LayerNorm
- Standard position-wise feed-forward network
- Multi-Head Self-Attention

#### ModernGPT

- Rotary Positional Embeddings
- RMSNorm
- SwiGLU
- RoPE-enhanced Multi-Head Self-Attention

### Examples

- End-to-end GPT training pipeline
- ModernGPT training pipeline
- Tiny Shakespeare language-model training
- BPE-based language modeling
- Autoregressive text generation

## Improved

- Language-model training stability through gradient clipping
- Effective batch-size scaling through gradient accumulation
- Generalization monitoring using validation datasets
- Training resumption through optimizer and scheduler checkpointing
- Text generation quality through configurable sampling strategies
- Token efficiency through BPE tokenization
- Transformer positional representation through RoPE
- Feed-forward expressiveness through SwiGLU
- Normalization efficiency through RMSNorm
- GPU training support through the NumPy/CuPy backend
- BPE decoding and end-of-word marker handling
- Training pipeline checkpoint organization

---

# v1.0.0

## Added

### Transformer Architecture

- Learned Positional Embedding
- Scaled Dot Product Attention
- Multi-Head Attention
- Position-wise Feed Forward Network
- GPT-style Pre-LayerNorm Transformer Block
- GPT Decoder Model
- Causal Attention Masking

### Neural Network

- ModuleList container
- FeedForward layer

### Tensor Operations

- Reshape operation with autograd
- Transpose operation with autograd

### Framework

- Transformer functional utilities
- Recursive module containers

## Improved

- Full Transformer support
- Batched matrix multiplication gradients
- Improved autograd graph for complex residual networks
- Improved serialization support for nested modules
- Expanded unit test coverage to 330+ tests

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