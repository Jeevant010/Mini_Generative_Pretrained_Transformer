# Chapter 9.4 — Overfitting Signals

## What Is Overfitting?

Overfitting happens when the model **memorizes** the training data instead of **learning** general patterns. It is like a student who memorizes the answers to practice tests but cannot solve new problems.

## How to Detect Overfitting

### The Train-Val Loss Gap

The clearest signal: **training loss keeps decreasing but validation loss starts increasing.**

```
Healthy training:
  Step 40K: Train loss 3.62, Val loss 3.61  (gap: 0.01)
  Step 60K: Train loss 3.50, Val loss 3.52  (gap: 0.02)
  Step 80K: Train loss 3.40, Val loss 3.45  (gap: 0.05)  ← gap growing slowly

Overfitting:
  Step 40K: Train loss 3.62, Val loss 3.61  (gap: 0.01)
  Step 60K: Train loss 3.20, Val loss 3.55  (gap: 0.35)  ← gap exploding
  Step 80K: Train loss 2.80, Val loss 3.70  (gap: 0.90)  ← model is memorizing
```

In our actual training run, the gap stays small (0.01–0.05), which means we are **not overfitting**. This makes sense — we have only seen ~18% of the training data by step 122K.

### Verbatim Reproduction

Another sign: the model generates text that exactly matches training documents, word for word. If you can find the model's output in the training data, it has memorized rather than learned.

### Decreased Diversity

If Distinct-N scores decrease over time:
- Early training: Distinct-2 = 0.75 (diverse output)
- Late training: Distinct-2 = 0.40 (repetitive, possibly memorized)

## Why We Are NOT Overfitting (Yet)

Our model has favorable conditions:

| Factor | Our Situation | Overfitting Risk |
|---|---|---|
| Dataset size | 5.1 billion tokens | Low — massive dataset |
| Steps completed | 122,000 | Low — only seen 18% of data |
| Epochs completed | 0.18 | Low — less than 1 pass |
| Train-val gap | ~0.02-0.05 | Low — small gap |
| Dropout | 0.1 | Helps — regularization active |

Overfitting typically becomes a concern when:
- The model has seen the full dataset multiple times (epoch > 1)
- The train-val gap exceeds 0.5
- Generated text starts reproducing training documents verbatim

## What to Do If Overfitting Occurs

1. **Stop training earlier** — Use the `best_model.pt` checkpoint
2. **Increase dropout** — From 0.1 to 0.2
3. **Increase weight decay** — From 0.1 to 0.3
4. **Add more data** — Use a larger dataset subset
5. **Add label smoothing** — Reduces overconfidence in predictions

## The Overfitting vs Degeneration Distinction

These are **different problems**:

| Property | Overfitting | Degeneration |
|---|---|---|
| Cause | Too much training on same data | Autoregressive feedback loop |
| Detection | Train-val loss gap | Repetition in generated text |
| When it happens | After multiple epochs | Can happen at any time during generation |
| Fix | Regularization, early stopping | Repetition penalty, top-p sampling |
| Model change? | Yes — model itself is worse | No — model is fine, sampling is wrong |

Our "ibn nimy" problem was degeneration (a sampling/data issue), not overfitting (the model itself was not overfit).
