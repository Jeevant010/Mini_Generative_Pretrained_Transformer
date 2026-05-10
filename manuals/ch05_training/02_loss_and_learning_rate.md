# Chapter 5.2 — Loss, Learning Rate, and Checkpoints

## Loss: Measuring How Wrong the Model Is

At every step, the model predicts what the next token should be for each position in the batch. The **loss** measures how far off these predictions are from the actual tokens.

- **High loss** (e.g., 10.5 at step 0): The model is very wrong — basically guessing randomly
- **Low loss** (e.g., 3.5 at step 60,000): The model is much better — its predictions are close to reality

The loss is computed using **cross-entropy**, which measures the difference between the model's probability distribution and the correct answer.

### Perplexity: A More Intuitive Version

Perplexity = e^(loss). It tells you: "On average, how many tokens is the model choosing between?"

| Loss | Perplexity | Meaning |
|---|---|---|
| 10.5 | 37,780 | The model is considering all 32,000+ tokens equally |
| 5.0 | 148 | The model has narrowed it down to about 148 candidates |
| 3.5 | 33 | The model is choosing between about 33 likely tokens |
| 2.0 | 7.4 | The model has narrowed it down to about 7 candidates |

Our model went from PPL 37,780 to PPL 33 — a 1,121× improvement.

## Learning Rate Schedule

The learning rate controls how big each parameter update is. Too high = the model overshoots and destabilizes. Too low = the model learns too slowly.

We use a **cosine decay** schedule:

1. **Warmup** (first 1,000 steps): Learning rate ramps up from 0 to 3e-4
2. **Cosine decay** (steps 1,000 to 150,000): Learning rate smoothly decreases from 3e-4 to 3e-5

Why warmup? At the very start, the model's parameters are random. Large updates on random parameters can cause instability. The warmup lets the model "find its footing" before the full learning rate kicks in.

Why decay? As training progresses, the model is already close to a good solution. Large updates would overshoot. Gradually reducing the learning rate allows the model to fine-tune its parameters more precisely.

## Checkpoints: Saving Progress

Training takes 2-3 days. If the power goes out or the program crashes, we would lose everything without checkpoints.

Our training script saves a checkpoint every 2,000 steps:

```
checkpoints/
├── ckpt_step_2000.pt
├── ckpt_step_4000.pt
├── ckpt_step_6000.pt
├── ...
├── ckpt_step_120000.pt
└── best_model.pt
```

Each checkpoint contains:
- All 118 million model parameters
- The optimizer state (so training can resume exactly)
- The current step number
- The current learning rate
- The best validation loss seen so far

### Best Model

Whenever the validation loss reaches a new lowest value, the model is saved as `best_model.pt`. This is the checkpoint you should use for generation — it represents the model at its best, not its most recent.

### Resuming Training

If training is interrupted, the script automatically finds the latest checkpoint and resumes:

```python
# On startup, training.py checks for existing checkpoints
if latest_checkpoint_exists:
    load_checkpoint()
    resume_from_that_step()
```

This means you can safely stop and restart training without losing progress.

## Train Loss vs Validation Loss

We track two types of loss:

- **Training loss**: How well the model predicts tokens from the training data (the data it is learning from)
- **Validation loss**: How well the model predicts tokens from held-out data it has never seen during training

If both losses decrease together, the model is learning well. If training loss decreases but validation loss increases, the model is **overfitting** — it is memorizing the training data instead of learning general patterns. More on this in Chapter 9.
