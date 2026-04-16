# Mini Generative Pretrained Transformer

A personal educational project for building a small GPT-style language model from scratch using PyTorch.

The repository includes:
- Core training and generation scripts
- Memory-mapped dataset batching for efficient text sampling
- Data extraction utilities for OpenWebText `.xz` files
- Research notes and notebooks explaining attention, tokenization, embeddings, and full architecture

## Project Goals

- Learn transformer internals by implementing core blocks manually
- Train a compact autoregressive language model on text data
- Experiment with architecture and hyperparameters on local hardware (CPU/GPU)
- Maintain notebook-based research notes for iterative learning

## Repository Structure

```text
Mini_Generative_Pretrained_Transformer/
|- app.py                      # Currently empty
|- config.py                   # Device + hyperparameters + data path
|- dataset.py                  # mmap-based random chunk loading and batching
|- extract.py                  # OpenWebText .xz extraction + vocab dump utility
|- generate.py                 # Inference script (loads checkpoint and generates text)
|- training.py                 # Main model/training implementation (monolithic script)
|- wizard_of_oz.txt            # Default training text corpus
|- requirements.txt            # Python dependencies
|- Research/                   # Learning notes + notebooks + tuning guides
```

## Current Model/Training Configuration

Defined in `config.py`:

- `batch_size = 32`
- `block_size = 128`
- `max_iters = 20000`
- `learning_rate = 3e-4`
- `eval_iters = 200`
- `n_embd = 256`
- `n_head = 8`
- `n_layer = 6`
- `dropout = 0.1`
- `DATA_FILE = "wizard_of_oz.txt"`

Device auto-selection priority:
1. CUDA
2. Apple MPS
3. CPU

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you want CUDA wheels for PyTorch (supported NVIDIA setup):

```bash
pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

## How To Run

### Train

```bash
python training.py
```

Expected output includes periodic train/val loss logs and a saved checkpoint:
- `model-01.pt`

### Generate text from checkpoint

```bash
python generate.py --prompt "Once upon a time" --max-tokens 200 --checkpoint model-01.pt
```

### Build text files from OpenWebText `.xz` archives

```bash
python extract.py
```

This creates:
- `output_train.txt`
- `output_val.txt`
- `vocab.txt`

## Research Folder

The `Research/` directory contains:
- Step-by-step notebook walkthroughs for tokenizer, embeddings, attention, and full architecture
- Tuning and setup guides (including GPU-specific notes)
- Experimental checkpoints/notebook artifacts

These files are useful if you want the theory and experimentation context behind the implementation.

## Important Notes (Current State)

- `generate.py` and `dataset.py` reference `model.py` and/or `tokenizer.py`, but these files are not currently in the project root.
- `training.py` is currently the most complete implementation and includes model classes inline.
- `app.py` exists but is empty.

If you plan to use modular imports (`model.py`, `tokenizer.py`), add those files or refactor `training.py` into separate modules.

## Suggested Next Improvements

1. Split `training.py` into `model.py`, `tokenizer.py`, and `train.py`.
2. Add a simple `argparse` interface for training hyperparameters.
3. Add a train/validation text split pipeline used consistently by all scripts.
4. Save tokenizer artifacts (`stoi/itos` or vocab) with checkpoints.
5. Add a quick smoke test and reproducibility seed setup.

## License

See `LICENSE` for project licensing details.
