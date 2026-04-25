# Legacy Code — Original Versions Before Ablation Upgrades

This folder contains the **original versions** of files that were enhanced with ablation toggles, evaluation suite, and metrics logging.

These files are preserved for reference so you can always compare the old code with the new code.

## Files

| File | Original Location | What Changed |
|------|------------------|-------------|
| `config_v1.py` | `config.py` | Added ablation toggles, presets, sample generation, CSV logging |
| `model_v1.py` | `model.py` | Added `Identity` class, `manual_causal_attention()`, RoPE/GQA/Flash toggles |
| `training_v1.py` | `training.py` | Added PPL reporting, sample generation, grad norm, VRAM, CSV logger |

## How These Relate

```
config_v1.py   →  config.py    (ablation toggles + 6 presets + logging config added)
model_v1.py    →  model.py     (4 ablation toggles wired into forward passes)
training_v1.py →  training.py  (evaluation suite + CSV + sample gen integrated)
```

The new versions are **fully backward compatible** — if all ablation flags are `True` (default), the behavior is identical to these originals.
