# Quick Start Guide

## Requirements

- Python 3.8 or newer
- NVIDIA GPU with 6+ GB VRAM (we use RTX 4060 Laptop, 8 GB)
- CUDA toolkit installed
- ~15 GB free disk space

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tokenizers numpy tqdm
```

### 2. Verify GPU

```python
import torch
print(torch.cuda.is_available())       # Should print: True
print(torch.cuda.get_device_name(0))   # Should print your GPU name
```

## Running the Pipeline

### Step 1: Prepare Data

```bash
python prepare_data.py
```

This will:
- Find parquet files in `DATASET_PATH` (configured in `config.py`)
- Train a tokenizer if one does not exist
- Filter and tokenize documents
- Write `train.bin` and `val.bin`

Time: 2-4 hours for 10 GB target.

### Step 2: Train

```bash
python training.py
```

This will:
- Build the model
- Resume from the latest checkpoint if one exists
- Train for `max_iters` steps (default: 150,000)
- Save checkpoints every 2,000 steps
- Generate samples every 2,000 steps

Time: 2-3 days on RTX 4060.

### Step 3: Generate Text

```bash
python generate.py --prompt "The future of AI is" --max-tokens 100
```

### Step 4: Evaluate Quality

```bash
# Evaluate the latest checkpoint
python -m evaluation.quality_metrics

# Compare all checkpoints
python -m evaluation.quality_metrics --all-checkpoints
```

## Quick Smoke Test

To verify everything works without training for days:

1. Set the preset to `wizard_of_oz_smoke` in `config.py`
2. Run `python training.py`
3. Training should complete in 5-10 minutes
4. Run `python generate.py --prompt "Dorothy said"` to see output

## Common Issues

| Issue | Solution |
|---|---|
| `CUDA out of memory` | Reduce `batch_size` in `config.py` |
| `No parquet files found` | Check `DATASET_PATH` in `config.py` |
| `Tokenizer not found` | Run `prepare_data.py` first |
| `No checkpoints found` | Run `training.py` first |
| `ModuleNotFoundError: tokenizers` | `pip install tokenizers` |
