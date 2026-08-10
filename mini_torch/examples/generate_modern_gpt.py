from pathlib import Path
import numpy as np

from mini_torch.nn.gpt import GPT
from mini_torch.tensors import tensor
from mini_torch.text.bpe_tokenizer import BPETokenizer
from mini_torch.backend import use_gpu, to_cpu
from mini_torch.nn.sampling import (top_k_logits, top_p_logits)
from mini_torch.nn.modern_gpt import ModernGPT
from mini_torch.amp.autocast import autocast
use_gpu()

tokenizer = BPETokenizer()
tokenizer.load( "checkpoints/modern_gpt/bpe_tokenizer.pkl")

model = ModernGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    max_seq_len=128,
    ff_hidden_dim=384,
    dropout=0.1,
)

model.load(
    "checkpoints/modern_gpt/modern_gpt_epoch_5.npz"
)

model.cuda()
model.eval()

prompt = "ACT I  Scene I. "

tokens = tokenizer.encode(prompt)

context_length = 128
max_new_tokens = 500
temperature = 0.6

for _ in range(max_new_tokens):

    x = tensor([tokens[-context_length:]])

    logits = model(x)

    next_logits = logits[:, -1, :]

    filtered = top_k_logits(next_logits.data[0] / temperature, k=20,)

    filtered = top_p_logits(filtered,p=0.9,)

    filtered = tensor(filtered)

    probs = filtered.softmax(axis=-1)

    p = to_cpu(probs.data)

    next_token = np.random.choice(len(p),p=p,)

    tokens.append(next_token)      # <- VERY IMPORTANT

print(tokenizer.decode(tokens))