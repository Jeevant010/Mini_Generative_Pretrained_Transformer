# Configuration Reference — Every Hyperparameter Explained

## 1. Overview

All configuration is centralized in `config.py`. Every module in the project imports this file as a namespace and reads values directly. This document explains every parameter, its default value, valid range, and effect on training behavior.

---

## 2. Device Setup

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | Auto-detected | `"cuda"` if NVIDIA GPU with CUDA is available; otherwise `"cpu"`. All tensors and the model are placed on this device. |

---

## 3. Training Hyperparameters

### 3.1 Batch & Sequence

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `batch_size` | 20 | 1–128+ | Number of independent sequences per training step. Larger = more stable gradients but more memory. |
| `block_size` | 384 | 32–2048+ | Context window (sequence length) in tokens. Each training example is `block_size` contiguous tokens. Attention compute scales quadratically with this value. |

**Effective batch size** (in tokens): `batch_size × block_size = 20 × 384 = 7,680 tokens/step`

### 3.2 Training Duration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_iters` | 300,000 | 100–1M+ | Total number of training steps. Each step processes one batch. |

**Total tokens seen** (assuming no repeat): `max_iters × batch_size × block_size = 300,000 × 7,680 = 2.3B tokens`

### 3.3 Learning Rate

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `learning_rate` | 2.5e-4 | 1e-5–1e-3 | Peak learning rate (used after warmup). |
| `min_lr` | 2.5e-5 | 0–`learning_rate` | Minimum learning rate at end of cosine decay. Typically 10% of peak. |
| `warmup_iters` | 2,000 | 0–10,000 | Number of linear warmup steps from 0 to `learning_rate`. |
| `lr_decay_iters` | `max_iters` | 1–`max_iters` | Number of steps for cosine decay from peak to `min_lr`. |

### 3.4 Learning Rate Schedule

The schedule has three phases:

```
Phase 1: Linear Warmup     (step 0 → warmup_iters)
Phase 2: Cosine Decay       (warmup_iters → lr_decay_iters)
Phase 3: Constant min_lr    (lr_decay_iters → max_iters)
```

Mathematical formulation:

$$\text{lr}(t) = \begin{cases} \text{learning\_rate} \times \frac{t+1}{\text{warmup\_iters}} & \text{if } t < \text{warmup\_iters} \\ \text{min\_lr} + \frac{1}{2}(\text{learning\_rate} - \text{min\_lr})(1 + \cos(\pi \cdot \frac{t - \text{warmup\_iters}}{\text{lr\_decay\_iters} - \text{warmup\_iters}})) & \text{if } \text{warmup\_iters} \leq t \leq \text{lr\_decay\_iters} \\ \text{min\_lr} & \text{if } t > \text{lr\_decay\_iters} \end{cases}$$

### 3.5 Gradient Clipping

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `grad_clip` | 1.0 | 0.0–10.0 | Maximum L2 norm of the gradient vector. Set to 0 to disable. Prevents gradient explosions during training. |

### 3.6 Evaluation

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `eval_iters` | 25 | 5–100 | Number of batches averaged for each evaluation. More = more accurate loss estimate but slower evaluation. |
| `eval_interval` | 2,000 | 100–10,000 | Run evaluation every N training steps. Also evaluates at the final step. |

### 3.7 Checkpointing

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `checkpoint_interval` | 1,000 | 100–10,000 | Save a periodic checkpoint (`ckpt_step_<N>.pt`) every N steps. |

---

## 4. Model Architecture

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_embd` | 768 | 128–4096 | Embedding and hidden dimension ($d$). Must be divisible by `n_head`. |
| `n_layer` | 12 | 1–48 | Number of Transformer blocks. |
| `n_head` | 12 | 1–32 | Number of query attention heads. `n_embd / n_head` must be an integer. |
| `n_kv_heads` | 4 | 1–`n_head` | Number of Key-Value heads for GQA. `n_head / n_kv_heads` must be an integer. Set equal to `n_head` for full MHA. Set to 1 for MQA. |
| `dropout` | 0.1 | 0.0–0.5 | Dropout probability applied in SwiGLU and attention (during training only). |
| `ffn_mult` | 3.5 | 2.0–8.0 | FFN hidden dimension multiplier. Hidden dim = `int(ffn_mult × n_embd)`. |
| `vocab_size` | 32,000 | 256–100,000 | Must match the tokenizer's vocabulary size exactly. |

### 4.1 Derived Values

| Derived | Formula | Value |
|---------|---------|-------|
| Head dimension | `n_embd // n_head` | 64 |
| KV dimension | `n_kv_heads × head_dim` | 256 |
| Heads per KV group | `n_head // n_kv_heads` | 3 |
| FFN hidden dim | `int(ffn_mult × n_embd)` | 2,688 |

---

## 5. Data Paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_BIN` | `"train.bin"` | Path to training data binary (uint16 token IDs). |
| `VAL_BIN` | `"val.bin"` | Path to validation data binary. |
| `TOKENIZER_PATH` | `"bpe_tokenizer_32k.json"` | Path to trained BPE tokenizer JSON. |

All paths are relative to the project root (working directory).

---

## 6. Profiling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_PROFILING` | `False` | Enable PyTorch profiler during training. Set to `True` for performance analysis runs. |
| `PROFILING_WINDOW` | `(100, 110)` | `(start_step, end_step)` for profiler data collection. Only active when `ENABLE_PROFILING=True`. |

**Output**: Chrome trace files under `log/profiler/`, viewable in Chrome's `chrome://tracing` or Perfetto.

---

## 7. Iteration Timer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIMER_TARGET_ITERATION` | `None` | Set to a specific step index (integer) to print detailed timing breakdown for that step. Set to `None` to disable. |

**Timer output** (when enabled):
```
Data load          : X.XX ms
Forward pass       : X.XX ms
Backward pass      : X.XX ms
Optimizer step     : X.XX ms
Full step          : X.XX ms
```

---

## 8. Smoke Test Configuration

For quick validation before a long training run, temporarily override these values:

```python
# Smoke test overrides
max_iters = 300
eval_iters = 10
eval_interval = 100
checkpoint_interval = 150
ENABLE_PROFILING = False
TIMER_TARGET_ITERATION = None
```

**Verification checklist**:
- [ ] Step logs appear normally
- [ ] LR is printed in training log
- [ ] Evaluation runs at least once
- [ ] `checkpoints/ckpt_step_150.pt` appears
- [ ] `checkpoints/best_model.pt` appears
- [ ] Resume works on second launch
- [ ] Generation works from saved checkpoint

---

## 9. Configuration Anti-Patterns

| ❌ Don't | ✅ Do Instead | Why |
|----------|--------------|-----|
| Set `batch_size > 64` on RTX 4060 | Start with 20, increase cautiously | OOM risk with large activations |
| Set `block_size > 512` without testing | Increase gradually: 128→256→384 | Attention compute is quadratic |
| Set `n_kv_heads > n_head` | Keep `n_head % n_kv_heads == 0` | GQA requires clean divisibility |
| Change `vocab_size` after training | Always retrain tokenizer + model together | Embedding matrix dimension mismatch |
| Set `eval_interval = 1` | Use 500–2000 | Frequent eval dominates wall-clock time |
| Enable profiling for production runs | Use only for dedicated profiling sessions | Profiler has significant overhead |
