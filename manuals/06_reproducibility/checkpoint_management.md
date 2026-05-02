# Checkpoint Management

## Checkpoint Types

The project uses two checkpoint types:

| Type | Path | Purpose |
| --- | --- | --- |
| Periodic | `checkpoints/ckpt_step_<N>.pt` | Resume training |
| Best | `checkpoints/best_model.pt` | Best validation loss so far |

## Checkpoint Contents

Each checkpoint stores:

```text
step
model_state_dict
optimizer_state_dict
loss
best_val_loss
```

This means training can resume with optimizer momentum and best validation state intact.

## Current Observed State

The latest observed periodic checkpoint is:

```text
checkpoints/ckpt_step_60000.pt
```

Each checkpoint is approximately 1.32 GB.

## Resume Behavior

`training.py` scans for:

```text
ckpt_step_*.pt
```

sorts by step number, loads the latest, and starts from:

$$
\text{start step} = \text{checkpoint step} + 1
$$

## Best Model Behavior

At each evaluation interval, if:

$$
\mathcal{L}_{val} < \mathcal{L}_{best}
$$

then `best_model.pt` is overwritten.

## Disk Management

Many checkpoints consume large disk space. A typical policy is:

- keep `best_model.pt`
- keep the latest checkpoint
- keep milestone checkpoints such as 10k, 30k, 60k
- remove intermediate checkpoints only after confirming training can resume

Do not delete checkpoints while training is running.

## Loading For Generation

Use:

```powershell
python generate.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"
```

or omit `--checkpoint` to use the latest periodic checkpoint.

