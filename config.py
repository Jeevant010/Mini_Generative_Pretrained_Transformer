"""
config.py — Production hyperparameters for the Transformer Training Pipeline.

This file is the single source of truth for all hyperparameters, ablation toggles,
hardware presets, and evaluation settings. Every module imports this file directly.

Usage:
    Active preset: Change ACTIVE_PRESET below, or leave as None to use manual values.
    Ablation mode: Set USE_RMSNORM / USE_ROPE / USE_FLASH_ATTENTION / USE_GQA to False.
    Quick test:    Set ACTIVE_PRESET = "wizard_of_oz_smoke" for a 5-minute sanity check.
"""

import torch

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION TOGGLES — Flip these to False to prove why each component matters.
#   Run ablation/run_ablation.py to test all combinations automatically.
# ═══════════════════════════════════════════════════════════════════════════════
USE_RMSNORM         = True   # False → skip normalization (gradients explode to NaN)
USE_ROPE            = True   # False → no positional encoding (model becomes order-blind)
USE_FLASH_ATTENTION = True   # False → standard matmul attention (2-3× slower, 2× more VRAM)
USE_GQA             = True   # False → full MHA (n_kv_heads = n_head, more VRAM)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS (active values — overridden if ACTIVE_PRESET is set)
# ═══════════════════════════════════════════════════════════════════════════════
batch_size = 20
block_size = 384
max_iters = 300000
# max_iters = 300         #testing
learning_rate = 2.5e-4
min_lr = 2.5e-5
warmup_iters = 2000
lr_decay_iters = max_iters
grad_clip = 1.0
eval_iters = 25
# eval_iters = 10         #testing
eval_interval = 2000
# eval_interval = 100     #testing
checkpoint_interval = 1000
# checkpoint_interval = 150   #testing

# --- Sample Generation During Training ---
SAMPLE_PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a land far away",
    "In the beginning, there was nothing but",
]
SAMPLE_MAX_TOKENS   = 80
SAMPLE_TEMPERATURE  = 0.8
SAMPLE_TOP_K        = 50
GENERATE_SAMPLES    = True    # Set False to skip sample generation during eval

# --- Metrics Logging ---
LOG_DIR             = "logs"
LOG_METRICS_CSV     = True    # Write step metrics to logs/training_metrics.csv
LOG_GRAD_NORM       = True    # Track & print gradient norm after backward pass
LOG_VRAM            = True    # Track & print peak VRAM usage (CUDA only)




# ENABLE_PROFILING = False
# TIMER_TARGET_ITERATION = None

# --- Model Architecture (rtx_4060_quality) ---
n_embd = 768
n_layer = 12
n_head = 12
n_kv_heads = 4  # For GQA
dropout = 0.1
ffn_mult = 3.5  # For SwiGLU
vocab_size = 32000

# --- Data Path ---
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"
TOKENIZER_PATH = "bpe_tokenizer_32k.json"

# --- Profiling ---
ENABLE_PROFILING = False
PROFILING_WINDOW = (100, 110) # Start, End step

# --- Iteration Timer ---
# Set to an integer step index (e.g., 250) to print detailed timing for only that step.
# Set to None to disable.
TIMER_TARGET_ITERATION = None

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTED HYPERPARAMETER PRESETS
#
# These are hardware-safe defaults for your RTX 4060 (8 GB VRAM).
# Set ACTIVE_PRESET to a key below to override the manual values above.
# Set ACTIVE_PRESET = None to use the manual values above as-is.
#
# Estimated training times assume ~10,000 tokens/sec on RTX 4060.
# ═══════════════════════════════════════════════════════════════════════════════

ACTIVE_PRESET = None  # ← Set to a preset name string to auto-apply

PRESETS = {
    # ── Quick Ablation Test (wizard_of_oz.txt, ~43K tokens, ~5 minutes) ──
    "wizard_of_oz_smoke": {
        "batch_size": 8,
        "block_size": 128,
        "max_iters": 500,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 50,
        "eval_iters": 10,
        "eval_interval": 50,
        "checkpoint_interval": 100,
        "n_embd": 384,
        "n_layer": 6,
        "n_head": 6,
        "n_kv_heads": 2,
        "dropout": 0.1,
        "ffn_mult": 3.5,
        "vocab_size": 32000,
        "_description": "5-min sanity check on wizard_of_oz.txt. Perfect for ablation testing.",
        "_est_time": "~5 minutes",
        "_est_params": "~15M",
    },

    # ── Wizard of Oz Full (deeper training on small corpus, ~30 min) ──
    "wizard_of_oz_full": {
        "batch_size": 8,
        "block_size": 256,
        "max_iters": 3000,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 200,
        "eval_iters": 15,
        "eval_interval": 200,
        "checkpoint_interval": 500,
        "n_embd": 512,
        "n_layer": 8,
        "n_head": 8,
        "n_kv_heads": 4,
        "dropout": 0.1,
        "ffn_mult": 3.5,
        "vocab_size": 32000,
        "_description": "30-min deep run on wizard_of_oz.txt. Good for evaluation testing.",
        "_est_time": "~30 minutes",
        "_est_params": "~40M",
    },

    # ── 1 GB Subset (~500M tokens, ~14 hours) ──
    "subset_1gb": {
        "batch_size": 16,
        "block_size": 256,
        "max_iters": 20000,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 1000,
        "eval_iters": 20,
        "eval_interval": 1000,
        "checkpoint_interval": 2000,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_kv_heads": 4,
        "dropout": 0.1,
        "ffn_mult": 3.5,
        "vocab_size": 32000,
        "_description": "Overnight run on a 1GB data subset. First real training milestone.",
        "_est_time": "~14 hours",
        "_est_params": "~85M",
    },

    # ── 3 GB Subset (~1.5B tokens, ~2 days) ──
    "subset_3gb": {
        "batch_size": 20,
        "block_size": 384,
        "max_iters": 50000,
        "learning_rate": 2.5e-4,
        "min_lr": 2.5e-5,
        "warmup_iters": 2000,
        "eval_iters": 25,
        "eval_interval": 2000,
        "checkpoint_interval": 2000,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_kv_heads": 4,
        "dropout": 0.1,
        "ffn_mult": 3.5,
        "vocab_size": 32000,
        "_description": "Weekend run on 3GB subset. Strong baseline for paper results.",
        "_est_time": "~2 days",
        "_est_params": "~85M",
    },

    # ── 10 GB Subset (~5B tokens, ~5-6 days) ──
    "subset_10gb": {
        "batch_size": 20,
        "block_size": 384,
        "max_iters": 150000,
        "learning_rate": 2.5e-4,
        "min_lr": 2.5e-5,
        "warmup_iters": 2000,
        "eval_iters": 25,
        "eval_interval": 2000,
        "checkpoint_interval": 5000,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_kv_heads": 4,
        "dropout": 0.1,
        "ffn_mult": 3.5,
        "vocab_size": 32000,
        "_description": "Your target: 10GB over ~1 week. Show checkpoints and paper-ready results.",
        "_est_time": "~5-6 days (24/7)",
        "_est_params": "~85M",
    },

    # ── 60 GB Full OpenWebText (~30B tokens, ~5-6 weeks) ──
    "full_60gb": {
        "batch_size": 20,
        "block_size": 384,
        "max_iters": 300000,
        "learning_rate": 2.5e-4,
        "min_lr": 2.5e-5,
        "warmup_iters": 2000,
        "eval_iters": 25,
        "eval_interval": 2000,
        "checkpoint_interval": 5000,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_kv_heads": 4,
        "dropout": 0.1,
        "ffn_mult": 3.5,
        "vocab_size": 32000,
        "_description": "Full 60GB OpenWebText. Multi-week training run.",
        "_est_time": "~5-6 weeks (24/7)",
        "_est_params": "~85M",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-APPLY PRESET (do not edit below this line)
# ═══════════════════════════════════════════════════════════════════════════════
if ACTIVE_PRESET is not None:
    if ACTIVE_PRESET not in PRESETS:
        raise ValueError(
            f"Unknown preset '{ACTIVE_PRESET}'. "
            f"Available: {', '.join(PRESETS.keys())}"
        )
    _preset = PRESETS[ACTIVE_PRESET]
    print(f"⚙️  Applying preset: {ACTIVE_PRESET}")
    print(f"   {_preset.get('_description', '')}")
    print(f"   Est. time: {_preset.get('_est_time', 'N/A')} | Est. params: {_preset.get('_est_params', 'N/A')}")
    for _k, _v in _preset.items():
        if not _k.startswith("_"):  # Skip metadata keys
            globals()[_k] = _v
    # Recompute derived values
    lr_decay_iters = max_iters
    del _preset, _k, _v
