import time
import math
from pathlib import Path

from mini_torch.nn.modern_gpt import ModernGPT
from mini_torch.nn.losses import CrossEntropyLoss
from mini_torch.optim.adamW import AdamW
from mini_torch.backend import use_gpu
from mini_torch.text.bpe_tokenizer import BPETokenizer
from mini_torch.text.text_dataset import TextDataset
from mini_torch.scheduler.cosine_annealing_lr import CosineAnnealingLR
from mini_torch.data.dataloader import DataLoader
from mini_torch.nn.utils import clip_grad_norm

from mini_torch.amp.autocast import autocast
from mini_torch.amp.grad_scaler import GradScaler


# ==========================================================
# Hyperparameters
# ==========================================================

CONTEXT_LENGTH = 128
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FF_HIDDEN_DIM = 384

BATCH_SIZE = 32

LEARNING_RATE = 3e-4
EPOCHS = 10
DROPOUT = 0.1

PRINT_EVERY = 1000

ACCUMULATION_STEPS = 4

WEIGHT_DECAY = 0.01


# ==========================================================
# Checkpoint paths
# ==========================================================

CHECKPOINT_DIR = Path(
    "checkpoints/modern_gpt"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    CHECKPOINT_DIR
    / "modern_gpt_best.npz"
)

OPTIMIZER_PATH = (
    CHECKPOINT_DIR
    / "optimizer.pkl"
)

SCHEDULER_PATH = (
    CHECKPOINT_DIR
    / "scheduler.pkl"
)

TOKENIZER_PATH = (
    CHECKPOINT_DIR
    / "bpe_tokenizer.pkl"
)


# ==========================================================
# Backend
# ==========================================================

use_gpu()


# ==========================================================
# AMP GradScaler
# ==========================================================

scaler = GradScaler(
    init_scale=2.0 ** 16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)


# ==========================================================
# Load dataset
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
# Build BPE tokenizer
# ==========================================================

print("Building tokenizer...")

tokenizer = BPETokenizer()

try:

    print("Loading tokenizer...")

    tokenizer.load(
        TOKENIZER_PATH
    )

except FileNotFoundError:

    print(
        "No previous tokenizer "
        "checkpoint found."
    )

    tokenizer.fit(text)

    tokenizer.save(
        TOKENIZER_PATH
    )


print(
    f"Vocabulary Size : "
    f"{tokenizer.vocab_size}"
)


# ==========================================================
# Encode dataset
# ==========================================================

print("Encoding text...")

tokens = tokenizer.encode(text)


# ==========================================================
# Train / validation split
# ==========================================================

split = int(
    0.9
    * len(tokens)
)

train_tokens = (
    tokens[:split]
)

validation_tokens = (
    tokens[split:]
)


# ==========================================================
# Datasets
# ==========================================================

train_dataset = TextDataset(
    train_tokens,
    context_length=CONTEXT_LENGTH,
)

validation_dataset = TextDataset(
    validation_tokens,
    context_length=CONTEXT_LENGTH,
)


# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


print(
    f"Training Samples   : "
    f"{len(train_dataset)}"
)

print(
    f"Validation Samples : "
    f"{len(validation_dataset)}"
)


# ==========================================================
# Build ModernGPT
# ==========================================================

print("Building ModernGPT...")

model = ModernGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=CONTEXT_LENGTH,
    ff_hidden_dim=FF_HIDDEN_DIM,
    dropout=DROPOUT,
)

model.cuda()


try:

    print("Loading checkpoint...")

    model.load(
        MODEL_PATH
    )

    model.cuda()

    print(
        "Checkpoint loaded."
    )

except FileNotFoundError:

    print(
        "No model checkpoint found."
    )


# ==========================================================
# Loss
# ==========================================================

criterion = CrossEntropyLoss()


# ==========================================================
# Optimizer
# ==========================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


try:

    optimizer.load(
        OPTIMIZER_PATH
    )

    print(
        "Optimizer state loaded."
    )

except FileNotFoundError:

    print(
        "No optimizer checkpoint found."
    )


# ==========================================================
# Scheduler
# ==========================================================

steps_per_epoch = math.ceil(
    len(train_loader)
    / ACCUMULATION_STEPS
)

total_steps = (
    EPOCHS
    * steps_per_epoch
)

warmup_steps = int(
    0.03
    * total_steps
)


scheduler = CosineAnnealingLR(
    optimizer,
    total_steps=total_steps,
    warmup_steps=warmup_steps,
    eta_min=1e-5,
)


try:

    scheduler.load(
        SCHEDULER_PATH
    )

    print(
        "Scheduler state loaded."
    )

except FileNotFoundError:

    print(
        "No scheduler checkpoint found."
    )


print(
    f"Optimizer Steps/Epoch : "
    f"{steps_per_epoch}"
)

print(
    f"Total Optimizer Steps : "
    f"{total_steps}"
)

print(
    f"Warmup Steps          : "
    f"{warmup_steps}"
)

print(
    "Model created successfully.\n"
)


# ==========================================================
# Evaluation
# ==========================================================

def evaluate(
    model,
    loader,
    criterion,
    vocab_size,
):

    model.eval()

    total_loss = 0.0
    batch_count = 0


    for x, y in loader:

        x = x.cuda()
        y = y.cuda()


        # ------------------------------------------
        # Mixed-precision forward
        # ------------------------------------------

        with autocast():

            logits = model(x)


        # ------------------------------------------
        # Loss-sensitive operations in FP32
        # ------------------------------------------

        logits = logits.float()

        probabilities = logits.softmax(
            axis=-1
        )

        probabilities = probabilities.reshape(
            -1,
            vocab_size,
        )

        targets = y.reshape(-1)


        loss = criterion(
            probabilities,
            targets,
        )


        total_loss += (
            loss.item()
        )

        batch_count += 1


    return (
        total_loss
        / batch_count
    )


# ==========================================================
# Training
# ==========================================================

print(
    "Starting ModernGPT training...\n"
)


start = time.time()

best_validation_loss = (
    float("inf")
)


for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0

    batch_count = 0

    grad_norm = 0.0

    optimizer.zero_grad()


    for i, (x, y) in enumerate(
        train_loader
    ):

        x = x.cuda()
        y = y.cuda()


        # ==================================================
        # Mixed-precision forward
        # ==================================================

        with autocast():

            logits = model(x)


        # ==================================================
        # FP32 loss computation
        #
        # Softmax and CrossEntropy remain FP32 for
        # numerical stability.
        # ==================================================

        logits = logits.float()

        probabilities = logits.softmax(
            axis=-1
        )

        probabilities = (
            probabilities.reshape(
                -1,
                tokenizer.vocab_size,
            )
        )

        targets = y.reshape(-1)


        raw_loss = criterion(
            probabilities,
            targets,
        )


        # ==================================================
        # Gradient accumulation
        # ==================================================

        loss = (
            raw_loss
            / ACCUMULATION_STEPS
        )


        # ==================================================
        # Loss scaling
        # ==================================================

        scaled_loss = (
            scaler.scale_loss(
                loss
            )
        )


        # ==================================================
        # Backward
        # ==================================================

        scaled_loss.backward()


        # ==================================================
        # Optimizer step
        # ==================================================

        should_step = (

            (
                (i + 1)
                % ACCUMULATION_STEPS
                == 0
            )

            or

            (
                (i + 1)
                == len(train_loader)
            )
        )


        if should_step:

            # ----------------------------------------------
            # Convert scaled gradients back to their
            # true magnitude BEFORE clipping.
            # ----------------------------------------------

            scaler.unscale_(
                optimizer
            )


            # ----------------------------------------------
            # Detect FP16 overflow
            # ----------------------------------------------

            found_inf = (
                scaler.found_inf(
                    optimizer
                )
            )


            if not found_inf:

                # ------------------------------------------
                # Gradient clipping must happen after
                # gradient unscaling.
                # ------------------------------------------

                grad_norm = (
                    clip_grad_norm(
                        model.parameters(),
                        max_norm=1.0,
                    )
                )


                # ------------------------------------------
                # FP32 optimizer update
                # ------------------------------------------

                optimizer.step()


                # ------------------------------------------
                # Advance scheduler only when the
                # optimizer actually performed an update.
                # ------------------------------------------

                scheduler.step()


            else:

                print(
                    "Gradient overflow detected. "
                    "Skipping optimizer step. "
                    f"Scale: {scaler.scale}"
                )


            # ----------------------------------------------
            # Clear accumulated gradients regardless of
            # whether the optimizer step was skipped.
            # ----------------------------------------------

            optimizer.zero_grad()


            # ----------------------------------------------
            # Update dynamic loss scale
            # ----------------------------------------------

            scaler.update(
                found_inf
            )


        # ==================================================
        # Statistics
        # ==================================================

        epoch_loss += (
            raw_loss.item()
        )

        batch_count += 1


        if (
            batch_count
            % PRINT_EVERY
            == 0
        ):

            running_loss = (
                epoch_loss
                / batch_count
            )


            print(
                f"Epoch "
                f"[{epoch + 1}/{EPOCHS}] "
                f"| Batch "
                f"[{batch_count}/"
                f"{len(train_loader)}] "
                f"| Batch Loss: "
                f"{raw_loss.item():.4f} "
                f"| Avg Loss: "
                f"{running_loss:.4f} "
                f"| Grad Norm: "
                f"{grad_norm:.3f} "
                f"| LR: "
                f"{optimizer.lr:.6f} "
                f"| Loss Scale: "
                f"{scaler.scale:.0f}"
            )


    # ======================================================
    # Epoch statistics
    # ======================================================

    train_loss = (
        epoch_loss
        / batch_count
    )


    # ======================================================
    # Validation
    # ======================================================

    validation_loss = evaluate(
        model,
        validation_loader,
        criterion,
        tokenizer.vocab_size,
    )


    # ======================================================
    # Perplexity
    # ======================================================

    train_perplexity = (
        math.exp(
            train_loss
        )
    )

    validation_perplexity = (
        math.exp(
            validation_loss
        )
    )


    print(
        f"\nEpoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss        : "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation Loss   : "
        f"{validation_loss:.4f}"
    )

    print(
        f"Train Perplexity  : "
        f"{train_perplexity:.3f}"
    )

    print(
        f"Val Perplexity    : "
        f"{validation_perplexity:.3f}"
    )

    print(
        f"AMP Loss Scale    : "
        f"{scaler.scale:.0f}"
    )


    # ======================================================
    # Best checkpoint
    # ======================================================

    if (
        validation_loss
        < best_validation_loss
    ):

        best_validation_loss = (
            validation_loss
        )


        model.save(
            str(MODEL_PATH)
        )


        print(
            "New best ModernGPT "
            "model saved."
        )


    # ======================================================
    # Epoch checkpoint
    # ======================================================

    model.save(
        str(
            CHECKPOINT_DIR
            / (
                f"modern_gpt_epoch_"
                f"{epoch + 1}.npz"
            )
        )
    )


    optimizer.save(
        str(
            OPTIMIZER_PATH
        )
    )


    scheduler.save(
        str(
            SCHEDULER_PATH
        )
    )


    print(
        "Checkpoint saved.\n"
    )


# ==========================================================
# Complete
# ==========================================================

end = time.time()


print(
    "=" * 60
)

print(
    "ModernGPT Training Complete!"
)

print(
    f"Best Validation Loss : "
    f"{best_validation_loss:.4f}"
)

print(
    f"Total Time : "
    f"{end - start:.2f} seconds"
)

print(
    "=" * 60
)