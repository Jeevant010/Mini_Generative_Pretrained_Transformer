# Hyperparameter Presets — RTX 4060 Safe Defaults for Every Data Scale

## 1. The Problem

Choosing hyperparameters for LLM training requires understanding:
- How much VRAM each setting consumes
- How long training will take at a given data scale
- Which learning rate, batch size, and warmup work for your model size

This project solves that by providing **pre-computed presets** for every common scenario, all tested to be safe on an RTX 4060 (8 GB VRAM).

---

## 2. Available Presets

### Quick Reference

| Preset Name | Data Size | Est. Time | Params | Use Case |
|-------------|-----------|-----------|--------|----------|
| `wizard_of_oz_smoke` | 237 KB | ~5 min | ~15M | Ablation testing, quick sanity check |
| `wizard_of_oz_full` | 237 KB | ~30 min | ~40M | Deeper evaluation on small corpus |
| `subset_1gb` | 1 GB | ~14 hrs | ~85M | First real training overnight |
| `subset_3gb` | 3 GB | ~2 days | ~85M | Weekend baseline run |
| `subset_10gb` | 10 GB | ~5-6 days | ~85M | Production results for paper |
| `full_60gb` | 60 GB | ~5-6 weeks | ~85M | Full OpenWebText training |

### Usage

```python
# In config.py, set:
ACTIVE_PRESET = "subset_10gb"  # ← Change this
```

All hyperparameters are automatically applied when Python imports `config.py`.

---

## 3. Detailed Preset Specifications

### 3.1 `wizard_of_oz_smoke` — Quick Ablation Test

```
Data:       wizard_of_oz.txt (~43K tokens)
Time:       ~5 minutes
Parameters: ~15M
VRAM:       ~1.5 GB
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| batch_size | 8 | Small corpus → small batch is fine |
| block_size | 128 | Short context for speed |
| max_iters | 500 | Enough to see loss curve shape |
| n_embd | 384 | Small model for fast iteration |
| n_layer | 6 | Half the production depth |
| n_head | 6 | Matches embedding dim / 64 |
| n_kv_heads | 2 | GQA ratio maintained |
| warmup_iters | 50 | 10% of total steps |
| eval_interval | 50 | Frequent eval for visibility |

---

### 3.2 `subset_10gb` — Your Target for Paper Results

```
Data:       10 GB subset (~5B tokens)
Time:       ~5-6 days (24/7)
Parameters: ~85M
VRAM:       ~3.5 GB (plenty of headroom)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| batch_size | 20 | Maximizes throughput within VRAM |
| block_size | 384 | Good context for coherent generation |
| max_iters | 150,000 | ~1 epoch through 10GB |
| n_embd | 768 | Production model dimension |
| n_layer | 12 | Full depth |
| n_head | 12 | Standard configuration |
| n_kv_heads | 4 | 3:1 GQA ratio |
| warmup_iters | 2,000 | Standard LLM warmup |
| checkpoint_interval | 5,000 | Every ~2.5 hours |

---

## 4. Custom Presets

To add your own preset, add an entry to the `PRESETS` dict in `config.py`:

```python
"my_custom": {
    "batch_size": 16,
    "block_size": 512,
    "max_iters": 100000,
    "learning_rate": 2e-4,
    "min_lr": 2e-5,
    "warmup_iters": 1500,
    # ... all other params ...
    "_description": "My custom training configuration.",
    "_est_time": "~3 days",
    "_est_params": "~85M",
},
```

Keys starting with `_` are metadata and won't be applied as hyperparameters.

---

## 5. Safety Rules

> **Never exceed these on RTX 4060 (8 GB VRAM):**
> - `batch_size × block_size > 15,000` tokens per step
> - `n_embd > 1024` (model too large)
> - `n_layer > 16` (too deep for 8GB)
> - `block_size > 1024` (quadratic attention memory)

If you get `CUDA out of memory`, reduce `batch_size` first, then `block_size`.
