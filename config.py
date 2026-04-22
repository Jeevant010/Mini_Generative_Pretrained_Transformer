"""
config.py — Production hyperparameters for the Transformer Training Pipeline.
"""

import torch

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Training Hyperparameters ---
batch_size = 20
block_size = 384
max_iters = 50000 
learning_rate = 2.5e-4
eval_iters = 100
eval_interval = 500
checkpoint_interval = 1000

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
ENABLE_PROFILING = True
PROFILING_WINDOW = (100, 110) # Start, End step

# --- Iteration Timer ---
# Set to an integer step index (e.g., 250) to print detailed timing for only that step.
# Set to None to disable.
TIMER_TARGET_ITERATION = 1000
