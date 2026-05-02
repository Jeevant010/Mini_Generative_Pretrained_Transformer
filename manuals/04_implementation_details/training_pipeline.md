# Training Pipeline

## Entry Point

Use:

```powershell
python training.py
```

`training.py` is the current production training script.

## Setup Validation

Before training, the script checks:

- `batch_size > 0`
- `block_size > 1`
- `max_iters > 0`
- `eval_iters > 0`
- `eval_interval > 0`
- `checkpoint_interval > 0`
- `train.bin` exists and is large enough
- `val.bin` exists and is large enough

This catches missing preprocessing or invalid configuration before a long run starts.

## Model And Optimizer

The model is created with:

```python
model = GPTLanguageModel(config).to(device)
```

The optimizer is:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
```

## Resume Logic

The script scans:

```text
checkpoints/ckpt_step_*.pt
```

and loads the latest step checkpoint. It restores:

- model weights
- optimizer state
- next step number
- best validation loss

This allows interrupted training to continue automatically.

## Learning Rate

Warmup:

$$
\eta_t = \eta_{max}\frac{t+1}{T_{warmup}}
$$

Cosine decay:

$$
r_t = \frac{t - T_{warmup}}{T_{decay} - T_{warmup}}
$$

$$
\eta_t = \eta_{min} + \frac{1}{2}(1+\cos(\pi r_t))(\eta_{max}-\eta_{min})
$$

Current values:

| Symbol | Value |
| --- | ---: |
| `eta_max` | 2.5e-4 |
| `eta_min` | 2.5e-5 |
| `T_warmup` | 2000 |
| `T_decay` | 150000 |

## Optimization Step

Each step performs:

1. Sample batch from `dataset.py`.
2. Run forward pass under autocast.
3. Compute cross-entropy loss.
4. Zero gradients.
5. Backpropagate.
6. Clip gradients.
7. Step AdamW optimizer.
8. Log metrics.

The loss is:

$$
\mathcal{L} =
-\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
\log P_\theta(y_{b,t}\mid x_{b,\le t})
$$

## Gradient Clipping

The global norm is:

$$
\|g\|_2 = \sqrt{\sum_i \|g_i\|_2^2}
$$

If:

$$
\|g\|_2 > 1.0
$$

then gradients are rescaled before the optimizer step.

## Mixed Precision

The training loop uses:

```python
torch.autocast(device_type="cuda", dtype=torch.bfloat16)
```

This reduces memory pressure and improves throughput while keeping a wide exponent range.

## Evaluation

Every `eval_interval = 2000` steps, the script evaluates:

- train loss
- validation loss
- perplexity

Perplexity is:

$$
\operatorname{PPL} = e^{\mathcal{L}_{val}}
$$

At step 60,000:

$$
e^{3.517095} \approx 33.69
$$

## Checkpointing

Periodic checkpoints:

```text
checkpoints/ckpt_step_<N>.pt
```

Best validation model:

```text
checkpoints/best_model.pt
```

Each checkpoint stores:

```text
step
model_state_dict
optimizer_state_dict
loss
best_val_loss
```

## Metrics

When enabled, metrics are appended to:

```text
logs/training_metrics.csv
```

Tracked fields include:

- timestamp
- step
- loss
- learning rate
- tokens per second
- estimated TFLOPS
- gradient norm
- VRAM
- validation loss
- perplexity

