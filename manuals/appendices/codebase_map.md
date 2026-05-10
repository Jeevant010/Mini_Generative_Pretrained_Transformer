# Codebase Map

## File Overview

Every file in the project, what it does, and how it connects to other files.

## Core Pipeline Files

### `config.py` — Single Source of Truth

All hyperparameters, file paths, toggle switches, and preset configurations live here. Every other file imports from `config.py`.

**Key sections:**
- Model architecture (layers, dimensions, heads)
- Training settings (learning rate, batch size, steps)
- Data pipeline settings (dataset path, target size, filters)
- Ablation toggles (USE_RMSNORM, USE_ROPE, etc.)
- Preset system (wizard_of_oz_smoke, subset_10gb, etc.)

**Used by:** Every other file

---

### `prepare_data.py` — Data Preparation

Converts raw parquet web text into tokenized binary files.

**What it does:**
1. Finds parquet files in `DATASET_PATH`
2. Filters documents (English, quality, length)
3. Trains tokenizer if not present
4. Tokenizes all filtered documents
5. Writes `train.bin` and `val.bin`

**Inputs:** Parquet files in `D:\Openweb`
**Outputs:** `train.bin`, `val.bin`, `bpe_tokenizer_32k.json`
**Used by:** Run once before training

---

### `tokenizer.py` — Tokenizer Wrapper

Wraps the HuggingFace `tokenizers` library. Provides `encode()`, `decode()`, `encode_batch()`.

**Used by:** `prepare_data.py`, `training.py`, `generate.py`, evaluation scripts

---

### `dataset.py` — Memory-Mapped Data Loader

Opens `train.bin` and `val.bin` with `numpy.memmap` and provides random batch sampling.

**Key function:** `get_batch(split)` — returns a random batch of (input, target) token pairs.

**Used by:** `training.py`, `evaluation/perplexity.py`

---

### `model.py` — The GPT Model

Contains the full Transformer architecture.

**Classes:**
- `RMSNorm` — Root Mean Square normalization
- `CausalSelfAttention` — Multi-head attention with GQA and RoPE
- `SwiGLU` — Gated feed-forward layer
- `TransformerBlock` — One block (attention + SwiGLU + residuals)
- `GPTLanguageModel` — Full model (embeddings + 12 blocks + LM head + generation)

**Key methods:**
- `forward(idx, targets)` — Compute logits and loss for training
- `generate(idx, max_new_tokens, ...)` — Autoregressive text generation with temperature, top-k, top-p, repetition penalty

**Used by:** `training.py`, `generate.py`, evaluation scripts

---

### `training.py` — Training Loop

The main training script. Handles the full training lifecycle.

**What it does:**
1. Validates config and data files
2. Builds model and optimizer
3. Resumes from checkpoint if available
4. Runs forward/backward/update loop
5. Logs metrics (loss, LR, throughput, VRAM, gradient norm)
6. Evaluates on validation set periodically
7. Saves checkpoints and best model
8. Generates sample text periodically

**Outputs:** `checkpoints/*.pt`, `logs/training_metrics.csv`, `logs/samples/*.txt`

---

### `generate.py` — Text Generation CLI

Command-line script for generating text from a trained checkpoint.

**Usage:**
```bash
python generate.py --prompt "The future of AI" --max-tokens 100
```

---

## Evaluation Files

### `evaluation/perplexity.py`

Computes validation loss and perplexity.

### `evaluation/quality_metrics.py`

Computes Distinct-N, Self-BLEU, Entropy, Repetition Ratio across checkpoints. Produces comparison tables.

### `evaluation/sample_generator.py`

Generates and saves text samples for qualitative review.

---

## Supporting Files

| File | Purpose |
|---|---|
| `project_report.py` | Generates a repository summary with parameter counts |
| `present.md` | Project presentation / summary document |
| `ablation/` | Ablation runner scripts |
| `tools/` | Utility scripts |
| `legacy/` | Older training code (kept for comparison) |
| `Research/` | Jupyter notebooks and exploratory studies |

---

## Data Flow

```
Raw parquet files (D:\Openweb)
    ↓  prepare_data.py
train.bin + val.bin + bpe_tokenizer_32k.json
    ↓  training.py
checkpoints/*.pt + logs/samples/*.txt
    ↓  generate.py OR evaluation/quality_metrics.py
Generated text OR metrics report
```
