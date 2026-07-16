from pathlib import Path
import numpy as np

from mini_torch.nn.gpt import GPT
from mini_torch.tensors import tensor
from mini_torch.text.bpe_tokenizer import BPETokenizer
from mini_torch.backend import use_gpu, to_cpu
from mini_torch.nn.sampling import (top_k_logits, top_p_logits)

use_gpu()

tokenizer = BPETokenizer()
tokenizer.load( "checkpoints/tokenizer.pkl")

model = GPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    max_seq_len=128,
    dropout=0.0,
)

model.load(
    "checkpoints/gpt_best.npz"
)

model.cuda()
model.eval()

prompt = "ROMEO: "

tokens = tokenizer.encode(prompt)

context_length = 128
max_new_tokens = 300
temperature = 0.8

for _ in range(max_new_tokens):

    x = tensor([tokens[-context_length:]])

    logits = model(x)

    next_logits = logits[:, -1, :]

    filtered = top_k_logits(next_logits.data[0] / temperature, k=40,)

    filtered = top_p_logits(filtered,p=0.95,)

    filtered = tensor(filtered)

    probs = filtered.softmax(axis=-1)

    p = to_cpu(probs.data)

    next_token = np.random.choice(len(p),p=p,)

    tokens.append(next_token)      # <- VERY IMPORTANT

print(tokenizer.decode(tokens))