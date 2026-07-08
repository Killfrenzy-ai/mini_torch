import time
from pathlib import Path

from mini_torch.nn.gpt import GPT
from mini_torch.nn.losses import CrossEntropyLoss
from mini_torch.optim.adam import Adam

from mini_torch.text.character_tokenizer import CharacterTokenizer
from mini_torch.text.text_dataset import TextDataset

from mini_torch.data.dataloader import DataLoader


# ==========================================================
# Hyperparameters
# ==========================================================

CONTEXT_LENGTH = 64

EMBED_DIM = 64

NUM_HEADS = 4

NUM_LAYERS = 2

BATCH_SIZE = 32

LEARNING_RATE = 3e-4

EPOCHS = 5

DROPOUT = 0.1

PRINT_EVERY = 25


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

tokenizer = CharacterTokenizer()

tokenizer.fit(text)

tokenizer.save(
    "checkpoints/tokenizer.pkl"
)

print(f"Vocabulary Size : {tokenizer.vocab_size}")


# ==========================================================
# Encode Dataset
# ==========================================================

print("Encoding text...")

tokens = tokenizer.encode(text)


# ==========================================================
# Dataset
# ==========================================================

dataset = TextDataset(
    tokens,
    context_length=CONTEXT_LENGTH,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

print(f"Training Samples : {len(dataset)}")


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

criterion = CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print("Model created successfully.\n")


# ==========================================================
# Training
# ==========================================================

print("Starting training...\n")

start = time.time()

for epoch in range(EPOCHS):

    epoch_loss = 0.0

    batch_count = 0

    model.train()

    for x, y in loader:

        optimizer.zero_grad()

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

        loss.backward()

        optimizer.step()

        epoch_loss += loss.data.item()

        batch_count += 1

        if batch_count % PRINT_EVERY == 0:

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_count}] "
                f"Loss: {loss.data.item():.4f}"
            )

    average_loss = epoch_loss / batch_count

    print(
        f"\nEpoch {epoch+1} Complete "
        f"| Average Loss = {average_loss:.4f}"
    )

    model.save(
        f"checkpoints/gpt_epoch_{epoch+1}.npz"
    )

    print("Checkpoint Saved.\n")


end = time.time()

print("=" * 60)
print("Training Complete!")
print(f"Total Time : {end - start:.2f} seconds")
print("=" * 60)