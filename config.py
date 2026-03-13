"""
config.py — Hyperparameters and device configuration for the SLM.
"""

import torch

# ─── Device Setup ──────────────────────────────────────
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ─── Hyperparameters ───────────────────────────────────
batch_size = 32
block_size = 128
max_iters = 20000
learning_rate = 3e-4
eval_iters = 200

# ─── Model Architecture ───────────────────────────────
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.1

# ─── Data Path ─────────────────────────────────────────
DATA_FILE = "wizard_of_oz.txt"
