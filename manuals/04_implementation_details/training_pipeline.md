# Training Pipeline — Loop, LR Schedule, Checkpointing & Profiling

## 1. Overview

The production training pipeline (`training.py`) manages the full training lifecycle: validation, model init/resume, mixed-precision training, LR scheduling, evaluation, checkpointing, and profiling.

---

## 2. Pre-Flight Validation

`validate_training_setup()` checks all hyperparameters are valid and that `train.bin`/`val.bin` exist with sufficient tokens (`> block_size + 1`).

---

## 3. Model Initialization & Auto-Resume

- Fresh start: `GPTLanguageModel(config).to(device)` with AdamW optimizer.
- Auto-resume: Scans `checkpoints/` for `ckpt_step_*.pt`, loads the latest, restores model + optimizer state, and resumes from `step + 1`.

---

## 4. Training Loop (Per Step)

1. Compute LR via cosine schedule with warmup
2. Load batch `(x, y)` from memory-mapped data
3. Forward pass under `torch.autocast(dtype=torch.bfloat16)`
4. Cross-entropy loss
5. `optimizer.zero_grad(set_to_none=True)` — memory-efficient gradient clearing
6. `loss.backward()`
7. Gradient clipping to max norm 1.0
8. `optimizer.step()`
9. Profiler step (if enabled)

---

## 5. Learning Rate Schedule

Three-phase cosine schedule:

| Phase | Steps | Formula |
|-------|-------|---------|
| Linear warmup | 0 → 2,000 | `lr × (step+1) / warmup_iters` |
| Cosine decay | 2,000 → 300,000 | `min_lr + 0.5 × (lr - min_lr) × (1 + cos(π × ratio))` |
| Constant floor | > 300,000 | `min_lr = 2.5e-5` |

Peak LR: 2.5e-4 → Floor: 2.5e-5 (10× reduction).

---

## 6. Evaluation

- Runs every `eval_interval` (2,000) steps and at final step.
- Averages loss over `eval_iters` (25) batches per split.
- Model set to `eval()` mode (dropout disabled).
- `@torch.no_grad()` for speed and memory.

Output: `>>> EVAL Step 2000: train_loss 6.1234 | val_loss 6.4567`

---

## 7. Checkpointing

| Type | File | Trigger | Purpose |
|------|------|---------|---------|
| Periodic | `ckpt_step_<N>.pt` | Every 1,000 steps | Resume |
| Best | `best_model.pt` | Val loss improves | Inference |

Checkpoint contents: `step`, `model_state_dict`, `optimizer_state_dict`, `loss`, `best_val_loss`.

---

## 8. Monitoring (Every 100 Steps)

```
Step   100 | Loss: 10.34 | LR: 1.25e-05 | 45,312 tok/s | 0.43 TFLOPS
```

TFLOPS = `(6 × params × batch_size × block_size) / step_time / 1e12`

---

## 9. Iteration Timer

Set `TIMER_TARGET_ITERATION = 250` for per-phase breakdown:

```
Data load     : 0.45 ms
Forward pass  : 12.34 ms
Backward pass : 24.56 ms
Optimizer step: 3.21 ms
Full step     : 40.56 ms
```

---

## 10. Profiler Integration

When `ENABLE_PROFILING = True`, PyTorch profiler captures CPU+CUDA traces in `PROFILING_WINDOW`. Outputs: top 10 GPU operators, performance table, Chrome trace (`performance_trace.json`). View with `chrome://tracing` or Perfetto.
