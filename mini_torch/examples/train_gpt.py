import time
import math
from pathlib import Path

from mini_torch.nn.gpt import GPT
from mini_torch.nn.losses import CrossEntropyLoss
from mini_torch.optim.adamW import AdamW
from mini_torch.backend import use_gpu
from mini_torch.text.bpe_tokenizer import BPETokenizer
from mini_torch.text.text_dataset import TextDataset
from mini_torch.scheduler.cosine_annealing_lr import CosineAnnealingLR
from mini_torch.data.dataloader import DataLoader
from mini_torch.nn.utils import clip_grad_norm


# ==========================================================
# Hyperparameters
# ==========================================================

CONTEXT_LENGTH = 128

EMBED_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 4

BATCH_SIZE = 32

LEARNING_RATE = 3e-4

EPOCHS = 5

DROPOUT = 0.1

PRINT_EVERY = 1000

accumulation_steps = 4

use_gpu()
# ==========================================================
# Load Dataset
# ==========================================================

print("Loading dataset...")

DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "tiny_shakespeare.txt"
)

with open(
    DATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    text = f.read()


# ==========================================================
# Build Tokenizer
# ==========================================================

print("Building tokenizer...")

tokenizer = BPETokenizer()
try:
    print("loading tokenizer...")
    tokenizer.load("checkpoints/tokenizer.pkl")
except:
    print("No previous tokenizer checkpoint found.")
    tokenizer.fit(text)
    tokenizer.save("checkpoints/tokenizer.pkl")

print(f"Vocabulary Size : {tokenizer.vocab_size}")


# ==========================================================
# Encode Dataset
# ==========================================================

print("Encoding text...")

tokens = tokenizer.encode(text)

split = int(0.9 * len(tokens))

train_tokens = tokens[:split]
test_tokens = tokens[split:]


# ==========================================================
# Dataset
# ==========================================================

train_dataset = TextDataset(
    train_tokens,
    context_length=CONTEXT_LENGTH,
)

test_dataset = TextDataset(test_tokens, context_length=CONTEXT_LENGTH)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training Samples   : {len(train_dataset)}")
print(f"Validation Samples : {len(test_dataset)}")


# ==========================================================
# Build Model
# ==========================================================

print("Building GPT...")

model = GPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=CONTEXT_LENGTH,
    dropout=DROPOUT,
)
model.cuda()


print("loading checkpoint")
model.load("checkpoints/gpt_best.npz")
model.cuda()
print("checkpoint loaded")


criterion = CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

try:
    optimizer.load(
        "checkpoints/optimizer_latest.pkl"
    )
    print("Optimizer state loaded.")
except FileNotFoundError:
    print("No optimizer checkpoint found.")

steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)

total_steps = EPOCHS * steps_per_epoch
warmup_steps = int(0.03 * total_steps)

scheduler = CosineAnnealingLR(optimizer, total_steps=total_steps,warmup_steps = warmup_steps, eta_min=1e-5,)

try:
    scheduler.load(
        "checkpoints/scheduler_latest.pkl"
    )
    print("Scheduler state loaded.")
except FileNotFoundError:
    print("No scheduler checkpoint found.")

print("Model created successfully.\n")

def evaluate(
    model,
    loader,
    criterion,
    tokenizer,
):
    """
    Evaluate the model on the validation dataset.
    """

    was_training = model.training

    model.eval()

    total_loss = 0.0
    batch_count = 0

    for x, y in loader:

        x = x.cuda()
        y = y.cuda()

        logits = model(x)

        probabilities = logits.softmax(axis=-1)

        probabilities = probabilities.reshape(
            -1,
            tokenizer.vocab_size,
        )

        targets = y.reshape(-1)

        loss = criterion(
            probabilities,
            targets,
        )

        total_loss += loss.item()
        batch_count += 1

    if was_training:
        model.train()

    return total_loss / batch_count


# ==========================================================
# Training
# ==========================================================

print("Starting training...\n")

start = time.time()

best_loss = float("inf")

for epoch in range(EPOCHS):

    epoch_loss = 0.0
    batch_count = 0

    model.train()

    optimizer.zero_grad()

    for i, ( x, y) in enumerate(train_loader):

        x = x.cuda()
        y = y.cuda()

        logits = model(x)

        probabilities = logits.softmax(axis=-1)

        probabilities = probabilities.reshape(
            -1,
            tokenizer.vocab_size,
        )

        targets = y.reshape(-1)

        raw_loss = criterion(
            probabilities,
            targets,
        )

        loss = raw_loss / accumulation_steps

        loss.backward()

        if( (i +1) % accumulation_steps == 0 or (i+1) == len(train_loader)):

            grad_norm = clip_grad_norm(model.parameters(), max_norm= 1.0)

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()

        epoch_loss += raw_loss.item()

        batch_count += 1

        if batch_count % PRINT_EVERY == 0:

            running_loss = epoch_loss / batch_count

            print(
                f"Epoch [{epoch+1}/{EPOCHS}]"
                f"| Batch [{batch_count}/{len(train_loader)}]"
                f"| Batch Loss: {raw_loss.item():.4f} "
                f"| Avg Loss: {running_loss:.4f}"
                f"| Grad Norm: {grad_norm:.3f}"
                f"| LR: {optimizer.lr:.6f}"
            )

    average_loss = epoch_loss / batch_count
    validation_loss = evaluate(model, test_loader, criterion, tokenizer,)

    train_ppl = math.exp(average_loss)
    val_ppl = math.exp(validation_loss)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    print(f"Train Loss        : {average_loss:.4f}")
    print(f"Validation Loss   : {validation_loss:.4f}")

    print(f"Train Perplexity  : {train_ppl:.3f}")
    print(f"Val Perplexity    : {val_ppl:.3f}")

    if validation_loss < best_loss:

        best_loss = validation_loss

        model.save(
            "checkpoints/gpt_best.npz"
        )

        optimizer.save(
        "checkpoints/optimizer_latest.pkl"
        )

        scheduler.save(
            "checkpoints/scheduler_latest.pkl"
        )

        print("New best model saved.")

    model.save(
        f"checkpoints/gpt_epoch_{epoch+1}.npz"
    )

    optimizer.save(
        "checkpoints/optimizer_latest.pkl"
    )

    scheduler.save(
        "checkpoints/scheduler_latest.pkl"
    )

    print("Checkpoint Saved.\n")

end = time.time()

print("=" * 60)
print("Training Complete!")
print(f"Best Validation Loss : {best_loss:.4f}")
print(f"Total Time : {end - start:.2f} seconds")
print("=" * 60)