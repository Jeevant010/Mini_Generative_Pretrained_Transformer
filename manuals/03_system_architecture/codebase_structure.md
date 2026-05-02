# Codebase Structure

## Core Files

| File | Purpose |
| --- | --- |
| `config.py` | Single source of truth for model, training, data, logging, and ablation settings |
| `prepare_data.py` | Reads parquet shards, filters documents, trains/loads tokenizer, writes `train.bin` and `val.bin` |
| `tokenizer.py` | Byte-level BPE tokenizer wrapper |
| `dataset.py` | Memory-mapped dataset loader and random batch sampler |
| `model.py` | Decoder-only GPT model with RMSNorm, RoPE, GQA, SwiGLU, and generation |
| `training.py` | Production training loop with metrics, evaluation, checkpointing, and resume |
| `generate.py` | CLI generation script for trained checkpoints |
| `project_report.py` | Generates a repository report with parameter counts and artifact summaries |

## Supporting Directories

| Directory | Purpose |
| --- | --- |
| `manuals/` | Current technical documentation |
| `paper_content/` | Paper-oriented Markdown drafts |
| `Research/` | Notebook studies and component walkthroughs |
| `evaluation/` | Perplexity and sample-generation utilities |
| `ablation/` | Ablation runner |
| `tools/` | Utility scripts |
| `legacy/` | Older training/model code kept for comparison |
| `Prompt_Outputs/` | Saved qualitative generations at different training milestones |
| `logs/` | Training metrics CSV and generated samples |
| `checkpoints/` | Periodic and best-model checkpoints |

## Main Runtime Flow

```text
prepare_data.py
    -> tokenizer.py
    -> train.bin / val.bin

training.py
    -> config.py
    -> dataset.py
    -> model.py
    -> logs/training_metrics.csv
    -> logs/samples/
    -> checkpoints/

generate.py
    -> config.py
    -> tokenizer.py
    -> model.py
    -> checkpoints/
```

## Data Preparation Flow

`prepare_data.py` performs:

1. Find parquet files under `DATASET_PATH`.
2. Detect text, language, and quality columns.
3. Filter documents.
4. Train `bpe_tokenizer_32k.json` if missing.
5. Load tokenizer if it already exists.
6. Encode text batches.
7. Append `<eos>`.
8. Split documents into train/val.
9. Write `uint16` token IDs to binary files.

## Training Flow

`training.py` performs:

1. Validate config and data files.
2. Construct `GPTLanguageModel(config)`.
3. Create AdamW optimizer.
4. Resume from latest `ckpt_step_*.pt` if present.
5. Sample memory-mapped batches.
6. Run forward/backward/update.
7. Log loss, learning rate, throughput, TFLOPS, gradient norm, and VRAM.
8. Evaluate validation loss and perplexity.
9. Save periodic checkpoints and `best_model.pt`.

## Important Version Note

`training.py` is the current production training script. `train.py` is older and no longer reflects the current model constructor or complete training pipeline. For current experiments and paper results, use:

```powershell
python training.py
```

not:

```powershell
python train.py
```

## Dependency Direction

The core dependency direction is intentionally simple:

```text
config.py
  used by -> prepare_data.py, dataset.py, model.py, training.py, generate.py

tokenizer.py
  used by -> prepare_data.py, training.py, generate.py

dataset.py
  used by -> training.py, evaluation/perplexity.py

model.py
  used by -> training.py, generate.py, evaluation/perplexity.py
```

This keeps the training system understandable and makes ablation changes easy to trace.

