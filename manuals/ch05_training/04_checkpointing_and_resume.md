# Chapter 5.4 — Checkpointing and Resume

## Why Checkpointing Matters

Training our model takes 2-3 days. In that time:

- Power can go out
- The computer might overheat and shut down
- Windows can install updates and restart
- A bug in logging code might crash the script
- You might want to stop and inspect results, then continue later

Without checkpointing, **any interruption means starting from scratch**. With checkpointing, you can always resume exactly where you left off.

## What Gets Saved

Each checkpoint is a `.pt` file (PyTorch format) containing:

| Component | Why It Is Saved |
|---|---|
| Model state dict | All 118 million parameters — the learned knowledge |
| Optimizer state dict | Momentum and variance tracking for AdamW — needed for smooth resume |
| Step number | So training knows where to continue |
| Best validation loss | So the "best model" tracking continues correctly |
| Learning rate | So the schedule resumes at the right point |
| Random number generator states | So the exact same random batches are not repeated |

## How Often We Save

```python
checkpoint_interval = 2000  # Save every 2,000 steps
```

At ~1 step per second, this means a checkpoint every ~33 minutes. Each checkpoint file is about 500 MB (the model in float32 + optimizer states).

## The Checkpoint Directory

```
checkpoints/
├── ckpt_step_2000.pt
├── ckpt_step_4000.pt
├── ckpt_step_6000.pt
├── ...
├── ckpt_step_120000.pt
├── ckpt_step_122000.pt
└── best_model.pt         ← Special: the best-performing checkpoint
```

## Automatic Resume

When `training.py` starts, it checks:

1. Is there a checkpoint directory?
2. Are there any `ckpt_step_*.pt` files?
3. If yes, load the latest one and continue from that step

This means you can safely kill and restart the script:

```bash
# Start training
python training.py
# ... training runs for 6 hours, reaches step 30,000 ...
# [Ctrl+C to stop]

# Later, restart — it automatically resumes from step 30,000
python training.py
```

## Best Model Tracking

Separately from regular checkpoints, the script tracks the **best model** — the checkpoint with the lowest validation loss ever seen.

Every time a new lowest validation loss is reached:
```
[Step 52000] New best val loss: 3.5575 → saving best_model.pt
```

This is the checkpoint you should use for generation and evaluation. The latest checkpoint is not always the best — sometimes later checkpoints are slightly worse due to noise.

## Checkpoint Size and Disk Space

| Component | Size per Checkpoint |
|---|---|
| Model (float32) | ~450 MB |
| Optimizer states | ~450 MB |
| Metadata | ~1 MB |
| **Total** | **~900 MB** |

With checkpoints every 2,000 steps and 150,000 total steps, that is 75 checkpoints × 900 MB = **~67 GB**.

If disk space is a concern, you can keep only the last 5 checkpoints and `best_model.pt`, deleting older ones.
