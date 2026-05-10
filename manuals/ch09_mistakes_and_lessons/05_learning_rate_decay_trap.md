# Chapter 9.5 — The Learning Rate Decay Trap

## The Problem

Our learning rate schedule decays from 3e-4 to 3e-5 over 150,000 steps. By step 100,000+, the learning rate is very small (~5e-5). This creates a subtle but important problem.

## What Goes Wrong

At the start of training (high learning rate), the model can make big adjustments. If it encounters a non-English fragment in the training data, it absorbs it, but it can also un-learn it if it encounters enough English data afterward.

At late stages (low learning rate), the model can only make tiny adjustments. Any pattern learned earlier is now effectively **frozen in place**. The model cannot easily:

- Correct mistakes it made during early training
- Un-learn bad patterns (like the "ibn nimy" sequence)
- Adapt to new patterns that conflict with early learning

## Why This Is a "Trap"

It is a trap because:

1. You do not see the problem during early training (loss is still decreasing)
2. The bad patterns are hidden in the model's parameters
3. They only surface during generation, when the model's overconfidence triggers them
4. By the time you notice, the learning rate is too low to fix it

## Real Example

Consider this timeline:

```
Step   5,000 (lr = 2.7e-4):  Model encounters "ibn nimy" in a training document
                               → Learns it as a valid token sequence
Step  50,000 (lr = 2.0e-4):  Model has seen enough English to dilute the pattern
                               → "ibn nimy" probability is low but non-zero
Step 100,000 (lr = 5.0e-5):  Model is overconfident + learning rate is tiny
                               → Cannot unlearn "ibn nimy"
                               → During generation, if "ibn" is sampled, the loop starts
```

## How to Avoid the Trap

### Prevention (Before Training)

1. **Clean your data thoroughly** before training starts (Chapter 4.2)
2. Use paragraph-level language detection, not just document-level
3. Set stricter thresholds for non-ASCII content

### Mitigation (During Training)

1. **Monitor validation loss closely** — if it starts plateauing or rising while train loss drops, investigate
2. **Generate samples regularly** — the automatic 2,000-step sampling catches degeneration early
3. **Use early stopping** — if quality degrades, use the best checkpoint, not the latest

### Fix (After Training)

1. **Repetition penalty at generation time** — breaks loops regardless of what the model learned
2. **Top-p sampling** — prevents overconfident predictions
3. **Retrain with cleaner data** — the nuclear option, but effective

## The Deeper Lesson

The learning rate decay trap illustrates a fundamental principle: **in machine learning, mistakes made early are harder to correct later.** This applies beyond just learning rates:

- Bad data early → persistent bad patterns
- Wrong architecture choices early → hard to fix without retraining
- Poor tokenizer → affects all downstream training

This is why data preparation and architecture design deserve as much care as the training loop itself.
