# Quick Start Guide — End-to-End in 5 Commands

## Prerequisites

- Python 3.10+ installed
- NVIDIA GPU with CUDA support (optional but recommended)
- Parquet data files in a local directory

---

## Step 1: Install Dependencies

```powershell
pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

## Step 2: Configure Dataset Path

Edit `prepare_data.py` and set your parquet directory:

```python
DATASET_PATH = r"D:\Openweb"    # ← Your path here
```

## Step 3: Prepare Data

```powershell
python prepare_data.py
```

This trains the BPE tokenizer (if needed) and creates `train.bin` + `val.bin`.

## Step 4: Train the Model

```powershell
python training.py
```

Training auto-resumes from the latest checkpoint if one exists.

## Step 5: Generate Text

```powershell
python generate.py --prompt "The future of AI is" --max-tokens 100
```

---

## Smoke Test (Before Long Runs)

Temporarily set in `config.py`:

```python
max_iters = 300
eval_iters = 10
eval_interval = 100
checkpoint_interval = 150
```

Run `python training.py`, verify checkpoints are created, run again to test resume, then test generation. Restore production values when satisfied.

---

## Useful Commands

| Task | Command |
|------|---------|
| Prepare data | `python prepare_data.py` |
| Train | `python training.py` |
| Generate (latest) | `python generate.py --prompt "..." --max-tokens 100` |
| Generate (best) | `python generate.py --checkpoint checkpoints/best_model.pt --prompt "..."` |
| Syntax check | `python -m py_compile training.py prepare_data.py dataset.py generate.py config.py tokenizer.py model.py` |
| Project report | `python project_report.py` |
| Profiler summary | `python profiler_quickview.py` |
