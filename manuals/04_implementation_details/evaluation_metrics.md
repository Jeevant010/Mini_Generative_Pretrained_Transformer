# Evaluation Metrics — Perplexity, Sample Generation & Hardware Profiling

## 1. Perplexity (PPL)

### 1.1 What Is Perplexity?

Perplexity measures how "surprised" a language model is by unseen text. It is the exponential of the average cross-entropy loss:

$$\text{PPL} = e^{\mathcal{L}} = e^{-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t})}$$

### 1.2 Interpretation

| PPL Range | Meaning |
|-----------|---------|
| 1 | Perfect prediction (impossible in practice) |
| 10-30 | Excellent (GPT-3 level on standard benchmarks) |
| 50-100 | Good (usable language generation) |
| 100-500 | Fair (coherent words, rough grammar) |
| 500-5000 | Learning (recognizable words, poor structure) |
| 5000+ | Random (essentially guessing) |

### 1.3 Usage

```bash
# Standalone evaluation
python -m evaluation.perplexity --checkpoint checkpoints/best_model.pt --batches 50

# Automatic during training (built into training.py)
# PPL is printed at every eval_interval:
# >>> EVAL Step 2000: train_loss 6.12 | val_loss 6.45 | PPL 632.70
```

---

## 2. Sample Generation

### 2.1 Purpose

Loss and perplexity are numbers — they don't show you what the model actually writes. Sample generation at regular intervals lets you physically watch the model evolve:

- **Step 0**: `"asd asdf asjdf asjhf"` (random noise)
- **Step 1000**: `"the the was a the was"` (learned common words)
- **Step 5000**: `"the man who was the king of"` (basic grammar)
- **Step 20000**: `"The invention of the printing press"` (coherent)

### 2.2 Configuration

In `config.py`:

```python
SAMPLE_PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a land far away",
    "In the beginning, there was nothing but",
]
SAMPLE_MAX_TOKENS   = 80
SAMPLE_TEMPERATURE  = 0.8
SAMPLE_TOP_K        = 50
GENERATE_SAMPLES    = True
```

### 2.3 Output

Samples are saved to `logs/samples/step_<N>.txt` at each evaluation interval.

---

## 3. Hardware Profiling Metrics

### 3.1 Metrics Tracked

| Metric | Formula | Unit |
|--------|---------|------|
| Tokens/sec | `batch_size × block_size / step_time` | tok/s |
| TFLOPS | `6 × params × batch_size × block_size / step_time / 1e12` | TFLOPS |
| VRAM | `torch.cuda.max_memory_allocated()` | MB |
| Gradient Norm | `torch.nn.utils.clip_grad_norm_()` | scalar |

### 3.2 CSV Logging

All metrics are logged to `logs/training_metrics.csv`:

```csv
timestamp,step,loss,lr,tokens_per_sec,tflops,grad_norm,vram_mb,val_loss,perplexity
2026-04-25 14:30:00,0,10.345600,1.25000000e-07,42350,0.4312,1.2345,3200,,
```

### 3.3 Profiling History

View cumulative training statistics:

```bash
python -m tools.profiling_history
python -m tools.profiling_history --plot  # Generate loss/VRAM curve images
```

---

## 4. Hyperparameter Presets

See `config.py` — `PRESETS` dictionary with pre-computed, hardware-safe defaults:

| Preset | Data Size | Est. Time | Est. Params |
|--------|-----------|-----------|-------------|
| `wizard_of_oz_smoke` | 237 KB | ~5 min | ~15M |
| `wizard_of_oz_full` | 237 KB | ~30 min | ~40M |
| `subset_1gb` | 1 GB | ~14 hrs | ~85M |
| `subset_3gb` | 3 GB | ~2 days | ~85M |
| `subset_10gb` | 10 GB | ~5-6 days | ~85M |
| `full_60gb` | 60 GB | ~5-6 weeks | ~85M |

Usage: Set `ACTIVE_PRESET = "subset_10gb"` in `config.py` — all hyperparameters are applied automatically.
