# Quick Start Guide

## 1. Install Dependencies

```powershell
pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

## 2. Configure Dataset

Edit `config.py`:

```python
DATASET_PATH = r"D:\Openweb"
ACTIVE_PRESET = "subset_10gb"
```

For a short test, use:

```python
ACTIVE_PRESET = "wizard_of_oz_smoke"
```

## 3. Prepare Data

```powershell
python prepare_data.py
```

Expected outputs:

```text
train.bin
val.bin
bpe_tokenizer_32k.json
```

## 4. Train

```powershell
python training.py
```

The script auto-resumes from the latest checkpoint if one exists.

## 5. Generate Text

Latest checkpoint:

```powershell
python generate.py --prompt "The future of AI is" --max-tokens 100
```

Best checkpoint:

```powershell
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The future of AI is" --max-tokens 100
```

## 6. Evaluate Perplexity

```powershell
python -m evaluation.perplexity --checkpoint checkpoints/best_model.pt --split val --batches 50
```

## 7. Generate Project Report

```powershell
$env:PYTHONIOENCODING='utf-8'
python project_report.py
```

## Expected Current Result

For the existing 60k run, the project report should show:

- 117,787,392 parameters
- 10 GB tokenized subset
- latest checkpoint around `ckpt_step_60000.pt`
- validation loss around 3.517
- perplexity around 33.69 in logs

