"""
ModernGPT Checkpoint Round-Trip Diagnostic

Purpose:
    Determine whether model.save() / model.load() correctly preserve
    every trained parameter in ModernGPT.

Process:
    1. Load the existing BPE tokenizer and Tiny Shakespeare dataset.
    2. Build a fresh ModernGPT.
    3. Train on ONE fixed batch for a small number of steps.
    4. Record model outputs.
    5. Save the model.
    6. Create a completely new ModernGPT.
    7. Load the saved checkpoint.
    8. Compare:
        - state_dict keys
        - parameter values
        - model outputs
        - predictions

Expected runtime:
    Usually well under one minute.
"""

from pathlib import Path

import numpy as np

from mini_torch.nn.modern_gpt import ModernGPT
from mini_torch.nn.losses import CrossEntropyLoss

from mini_torch.optim.adam import Adam

from mini_torch.backend import (
    use_gpu,
    to_cpu,
)

from mini_torch.text.bpe_tokenizer import (
    BPETokenizer,
)

from mini_torch.text.text_dataset import (
    TextDataset,
)

from mini_torch.data.dataloader import (
    DataLoader,
)


# ============================================================
# Configuration
# ============================================================

CONTEXT_LENGTH = 128

EMBED_DIM = 128

NUM_HEADS = 4

NUM_LAYERS = 4

FF_HIDDEN_DIM = 384

DROPOUT = 0.1


BATCH_SIZE = 32


# We do NOT need real training.
# We only need the parameters to change.

TRAINING_STEPS = 20


LEARNING_RATE = 3e-4


# ============================================================
# Paths
# ============================================================

CHECKPOINT_DIR = Path(
    "checkpoints/modern_gpt"
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TOKENIZER_PATH = (
    CHECKPOINT_DIR
    / "bpe_tokenizer.pkl"
)


ROUNDTRIP_PATH = (
    CHECKPOINT_DIR
    / "roundtrip_test.npz"
)


# ============================================================
# Backend
# ============================================================

print(
    "Initializing GPU..."
)


use_gpu()


# ============================================================
# Load dataset
# ============================================================

print(
    "Loading dataset..."
)


DATA_PATH = (
    Path(__file__).parent
    /"examples"
    / "data"
    / "tiny_shakespeare.txt"
)


with open(
    DATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    text = f.read()


# ============================================================
# Load tokenizer
# ============================================================

print(
    "Loading tokenizer..."
)


tokenizer = BPETokenizer()


tokenizer.load(
    TOKENIZER_PATH
)


print(

    f"Vocabulary Size : "

    f"{tokenizer.vocab_size}"

)


# ============================================================
# Encode text
# ============================================================

print(
    "Encoding text..."
)


tokens = tokenizer.encode(
    text
)


# ============================================================
# Dataset
# ============================================================

dataset = TextDataset(

    tokens,

    context_length=CONTEXT_LENGTH,

)


loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

)


# ============================================================
# Get ONE fixed batch
# ============================================================

print(
    "Loading fixed training batch..."
)


x, y = next(
    iter(loader)
)


x = x.cuda()

y = y.cuda()


print(

    "Input shape  :",

    x.shape,

)


print(

    "Target shape :",

    y.shape,

)


# ============================================================
# Build original model
# ============================================================

print()

print(
    "Building original ModernGPT..."
)


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


# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(

    model.parameters(),

    lr=LEARNING_RATE,

)


criterion = CrossEntropyLoss()


# ============================================================
# Initial output
# ============================================================

model.eval()


initial_logits = model(
    x
)


initial_logits_cpu = to_cpu(

    initial_logits.data

).copy()


# ============================================================
# Train for a few steps
# ============================================================

print()

print(
    "=" * 70
)

print(
    "SHORT TRAINING TEST"
)

print(
    "=" * 70
)


model.train()


for step in range(

    TRAINING_STEPS

):


    # --------------------------------------------------------
    # Clear gradients
    # --------------------------------------------------------

    optimizer.zero_grad()


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits = model(
        x
    )


    probabilities = logits.softmax(
        axis=-1
    )


    probabilities = probabilities.reshape(

        -1,

        tokenizer.vocab_size,

    )


    targets = y.reshape(
        -1
    )


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = criterion(

        probabilities,

        targets,

    )


    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer.step()


    print(

        f"Step "

        f"{step + 1:02d}/"

        f"{TRAINING_STEPS}"

        f" | Loss: "

        f"{loss.item():.6f}"

    )


# ============================================================
# Evaluate trained model
# ============================================================

model.eval()


original_logits = model(
    x
)


original_logits_cpu = to_cpu(

    original_logits.data

).copy()


# ============================================================
# Confirm training actually changed the model
# ============================================================

training_difference = np.abs(

    initial_logits_cpu

    - original_logits_cpu

)


print()

print(
    "=" * 70
)

print(
    "TRAINING CHANGE CHECK"
)

print(
    "=" * 70
)


print(

    "Maximum output change :",

    float(
        training_difference.max()
    ),

)


print(

    "Mean output change    :",

    float(
        training_difference.mean()
    ),

)


if (

    training_difference.max()

    == 0

):


    print()

    print(

        "WARNING: Training did not change model outputs."

    )


    print(

        "The serialization test may not detect "
        "unregistered trainable parameters."

    )


# ============================================================
# Capture original state
#
# IMPORTANT:
# Do this BEFORE saving.
# ============================================================

original_state = (
    model.state_dict()
)


print()

print(

    "Original state_dict entries:",

    len(
        original_state
    ),

)


print()

print(
    "Registered state_dict keys:"
)


for name in sorted(

    original_state.keys()

):


    print(

        "  ",

        name,

    )


# ============================================================
# Save model
# ============================================================

print()

print(
    "=" * 70
)

print(
    "SAVING MODEL"
)

print(
    "=" * 70
)


model.save(
    ROUNDTRIP_PATH
)


print(

    "Saved to:",

    ROUNDTRIP_PATH,

)


# ============================================================
# Create completely fresh model
# ============================================================

print()

print(
    "Building fresh ModernGPT..."
)


loaded_model = ModernGPT(

    vocab_size=tokenizer.vocab_size,

    embed_dim=EMBED_DIM,

    num_heads=NUM_HEADS,

    num_layers=NUM_LAYERS,

    max_seq_len=CONTEXT_LENGTH,

    ff_hidden_dim=FF_HIDDEN_DIM,

    dropout=DROPOUT,

)


# ============================================================
# Capture random model output BEFORE loading
#
# This confirms that loading actually changes the model.
# ============================================================

loaded_model.cuda()

loaded_model.eval()


random_logits = loaded_model(
    x
)


random_logits_cpu = to_cpu(

    random_logits.data

).copy()


# ============================================================
# Load checkpoint
# ============================================================

print()

print(
    "=" * 70
)

print(
    "LOADING MODEL"
)

print(
    "=" * 70
)


loaded_model.load(
    ROUNDTRIP_PATH
)


loaded_model.cuda()

loaded_model.eval()


# ============================================================
# Loaded output
# ============================================================

loaded_logits = loaded_model(
    x
)


loaded_logits_cpu = to_cpu(

    loaded_logits.data

).copy()


# ============================================================
# Capture loaded state
# ============================================================

loaded_state = (
    loaded_model.state_dict()
)


# ============================================================
# Compare fresh model vs loaded model
#
# Loading should have changed the fresh model.
# ============================================================

load_effect = np.abs(

    random_logits_cpu

    - loaded_logits_cpu

)


print()

print(
    "=" * 70
)

print(
    "LOAD EFFECT CHECK"
)

print(
    "=" * 70
)


print(

    "Maximum output change after load :",

    float(
        load_effect.max()
    ),

)


print(

    "Mean output change after load    :",

    float(
        load_effect.mean()
    ),

)


# ============================================================
# Compare state dictionaries
# ============================================================

print()

print(
    "=" * 70
)

print(
    "STATE DICT COMPARISON"
)

print(
    "=" * 70
)


original_keys = set(

    original_state.keys()

)


loaded_keys = set(

    loaded_state.keys()

)


missing_keys = (

    original_keys

    - loaded_keys

)


extra_keys = (

    loaded_keys

    - original_keys

)


print(

    "Original parameters :",

    len(
        original_keys
    ),

)


print(

    "Loaded parameters   :",

    len(
        loaded_keys
    ),

)


print(

    "Missing keys        :",

    len(
        missing_keys
    ),

)


print(

    "Extra keys          :",

    len(
        extra_keys
    ),

)


# ============================================================
# Print missing keys
# ============================================================

if missing_keys:


    print()

    print(
        "Missing keys:"
    )


    for key in sorted(

        missing_keys

    ):


        print(

            "  ",

            key,

        )


# ============================================================
# Print extra keys
# ============================================================

if extra_keys:


    print()

    print(
        "Extra keys:"
    )


    for key in sorted(

        extra_keys

    ):


        print(

            "  ",

            key,

        )


# ============================================================
# Compare parameter values
# ============================================================

parameter_mismatches = []

maximum_parameter_difference = 0.0


def extract_array(value):
    """
    Convert Tensor / NumPy array / CuPy array
    into a NumPy array safely.
    """

    # Tensor objects in mini_torch have .data,
    # but NumPy/CuPy arrays also expose a .data attribute.
    #
    # Therefore, do NOT use hasattr(value, "data")
    # to detect Tensor objects.

    if hasattr(value, "requires_grad") and hasattr(value, "data"):
        value = value.data

    return np.asarray(
        to_cpu(value)
    )


for name in sorted(
    original_keys & loaded_keys
):

    original_parameter = extract_array(
        original_state[name]
    )

    loaded_parameter = extract_array(
        loaded_state[name]
    )


    # --------------------------------------------------------
    # Shape comparison
    # --------------------------------------------------------

    if original_parameter.shape != loaded_parameter.shape:

        parameter_mismatches.append(
            (
                name,
                "SHAPE MISMATCH",
                original_parameter.shape,
                loaded_parameter.shape,
            )
        )

        continue


    # --------------------------------------------------------
    # Dtype comparison
    # --------------------------------------------------------

    if original_parameter.dtype != loaded_parameter.dtype:

        print(
            f"DTYPE WARNING: {name}"
            f" | original={original_parameter.dtype}"
            f" | loaded={loaded_parameter.dtype}"
        )


    # --------------------------------------------------------
    # Numerical comparison
    # --------------------------------------------------------

    difference = np.abs(
        original_parameter
        - loaded_parameter
    )


    max_difference = float(
        difference.max()
    )


    mean_difference = float(
        difference.mean()
    )


    maximum_parameter_difference = max(
        maximum_parameter_difference,
        max_difference,
    )


    if max_difference != 0:

        parameter_mismatches.append(
            (
                name,
                max_difference,
                mean_difference,
            )
        )


# ============================================================
# Parameter comparison results
# ============================================================

print()

print(

    "Maximum parameter difference :",

    maximum_parameter_difference,

)


print(

    "Mismatched parameters        :",

    len(
        parameter_mismatches
    ),

)


if parameter_mismatches:


    print()

    print(
        "Parameter mismatches:"
    )


    for mismatch in (

        parameter_mismatches

    ):


        print(

            "  ",

            mismatch,

        )


# ============================================================
# Compare model outputs
# ============================================================

output_difference = np.abs(

    original_logits_cpu

    - loaded_logits_cpu

)


maximum_output_difference = float(

    output_difference.max()

)


mean_output_difference = float(

    output_difference.mean()

)


# ============================================================
# Prediction comparison
# ============================================================

original_predictions = (

    original_logits_cpu.argmax(
        axis=-1
    )

)


loaded_predictions = (

    loaded_logits_cpu.argmax(
        axis=-1
    )

)


prediction_agreement = np.mean(

    original_predictions

    ==

    loaded_predictions

)


# ============================================================
# Final results
# ============================================================

print()

print(
    "=" * 70
)

print(
    "OUTPUT COMPARISON"
)

print(
    "=" * 70
)


print(

    "Maximum logit difference :",

    maximum_output_difference,

)


print(

    "Mean logit difference    :",

    mean_output_difference,

)


print(

    "Prediction agreement     :",

    f"{prediction_agreement * 100:.4f}%",

)


print()

print(

    "Original last prediction :",

    int(

        original_predictions[
            0,
            -1
        ]

    ),

)


print(

    "Loaded last prediction   :",

    int(

        loaded_predictions[
            0,
            -1
        ]

    ),

)


# ============================================================
# Final diagnosis
# ============================================================

print()

print(
    "=" * 70
)

print(
    "FINAL DIAGNOSIS"
)

print(
    "=" * 70
)


# ------------------------------------------------------------
# Perfect round trip
# ------------------------------------------------------------

if (

    not missing_keys

    and

    not extra_keys

    and

    len(
        parameter_mismatches
    )
    == 0

    and

    maximum_output_difference
    == 0

):


    print(

        "PASS: Checkpoint round-trip is exact."

    )


    print()

    print(

        "model.save() and model.load() preserve "
        "the complete serialized model state."

    )


    print()

    print(

        "The checkpoint problem is therefore likely "
        "outside basic serialization."

    )


# ------------------------------------------------------------
# Parameters differ
# ------------------------------------------------------------

elif (

    parameter_mismatches

    or

    missing_keys

    or

    extra_keys

):


    print(

        "FAIL: Model parameters are not preserved "
        "during checkpoint round-trip."

    )


    print()

    print(

        "The problem is in one or more of:"

    )


    print(

        "- state_dict()"

    )


    print(

        "- load_state_dict()"

    )


    print(

        "- save()"

    )


    print(

        "- load()"

    )


    print(

        "- recursive module registration"

    )


# ------------------------------------------------------------
# Parameters same but outputs differ
# ------------------------------------------------------------

elif (

    maximum_output_difference

    != 0

):


    print(

        "WARNING: Serialized parameters match, "
        "but model outputs do not."

    )


    print()

    print(

        "This strongly suggests that some non-parameter "
        "state affecting the forward pass is not being "
        "serialized."

    )


    print()

    print(

        "Primary suspects include:"

    )


    print(

        "- tied weights"

    )


    print(

        "- cached RoPE state"

    )


    print(

        "- dtype state"

    )


    print(

        "- unregistered tensors"

    )


    print(

        "- train/eval state"

    )


print(
    "=" * 70
)