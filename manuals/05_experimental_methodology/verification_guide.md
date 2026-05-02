# Verification Guide

## Before A Long Training Run

Run these checks:

```powershell
python -m py_compile training.py prepare_data.py dataset.py generate.py config.py tokenizer.py model.py
```

Then confirm:

- `bpe_tokenizer_32k.json` exists
- `train.bin` exists
- `val.bin` exists
- `train.bin` and `val.bin` are large enough for `block_size`
- CUDA is available if training on GPU

## Data Verification

Current expected artifacts:

| File | Expected status |
| --- | --- |
| `train.bin` | exists, about 9.50 GB |
| `val.bin` | exists, about 511 MB |
| `bpe_tokenizer_32k.json` | exists, about 2.16 MB |

Token count is:

$$
\text{tokens} = \frac{\text{file bytes}}{2}
$$

because token IDs are stored as `uint16`.

## Training Verification

Run:

```powershell
python training.py
```

Confirm:

- preset message prints
- model parameter count prints
- step logs appear
- loss is finite
- learning rate increases during warmup
- gradient norm is finite
- VRAM is logged on CUDA
- eval runs at step 0 and every 2,000 steps
- checkpoints are written

## Resume Verification

Stop and restart:

```powershell
python training.py
```

Expected behavior:

- latest `ckpt_step_*.pt` is loaded
- training resumes from `step + 1`
- best validation loss is restored

## Generation Verification

Run:

```powershell
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The future of AI is" --max-tokens 80
```

Expected behavior:

- tokenizer loads
- checkpoint loads
- model prints generated text

The text does not need to be instruction-following. It should be judged as base-model continuation.

## Metrics Verification

Check:

```text
logs/training_metrics.csv
```

Look for:

- non-empty step rows
- eval rows with validation loss and perplexity
- improving validation loss over time
- no long sequence of NaN values

## Known Issue

On Windows terminals using CP1252, importing `config.py` can fail if Unicode status symbols are printed. Running with UTF-8 output fixes this:

```powershell
$env:PYTHONIOENCODING='utf-8'
python project_report.py
```

Long-term fix: remove non-ASCII print symbols from `config.py`.

