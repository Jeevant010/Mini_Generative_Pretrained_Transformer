# Chapter 5.3 — The Learning Rate Schedule

## Why Not a Fixed Learning Rate?

If you set the learning rate to 0.001 and keep it there for the entire training run, two things go wrong:

1. **At the start**, the model's parameters are random. A high learning rate on random parameters can cause wild, destructive updates.
2. **Near the end**, the model is already close to a good solution. A high learning rate overshoots — the model bounces around the optimum instead of settling into it.

The solution is a **learning rate schedule** — a plan for how the learning rate changes over time.

## Our Schedule: Warmup + Cosine Decay

Our schedule has two phases:

### Phase 1: Linear Warmup (Steps 0–1,000)

The learning rate starts at zero and increases linearly to the peak value:

```
Step    0:  lr = 0.000000
Step  250:  lr = 0.000075
Step  500:  lr = 0.000150
Step  750:  lr = 0.000225
Step 1000:  lr = 0.000300  ← Peak
```

**Why warmup?** At step 0, the model is random noise. Large updates on random parameters can push the model into a bad state it cannot recover from. The warmup lets the model "find its footing" with gentle updates first.

### Phase 2: Cosine Decay (Steps 1,000–150,000)

After warmup, the learning rate follows a smooth cosine curve from the peak (3e-4) down to the minimum (3e-5):

```
Step   1,000:  lr = 0.000300  ← Peak
Step  40,000:  lr = 0.000220
Step  75,000:  lr = 0.000165  ← Halfway
Step 110,000:  lr = 0.000080
Step 150,000:  lr = 0.000030  ← Minimum
```

**Why cosine?** The cosine shape provides a smooth, gradual decrease. Unlike a step schedule (which drops suddenly), cosine decay never makes an abrupt change. This leads to more stable training.

**Why not decay to zero?** The minimum learning rate (3e-5) is not zero — it is 10% of the peak. This ensures the model can always make small adjustments, even at the very end of training. Decaying to zero would freeze the model completely.

## The Learning Rate Trap

One of the mistakes we observed (Chapter 9) was related to learning rate decay. At very late steps (100K+), the learning rate is very small. This means:

- The model **cannot easily unlearn bad patterns** it picked up earlier
- If the model memorized a non-English fragment like "ibn nimy," the low learning rate makes it nearly impossible to correct

This is why data quality filtering (Chapter 4) is so important — bad patterns must be prevented from entering the training data in the first place, because correcting them later is very difficult.

## Settings

| Parameter | Value | Description |
|---|---|---|
| `learning_rate` | 3e-4 | Peak learning rate |
| `min_lr` | 3e-5 | Minimum learning rate |
| `warmup_steps` | 1,000 | Steps for linear warmup |
| `max_iters` | 150,000 | Total training steps (end of decay) |
