# Chapter 5.1 — What Happens During Training

## The Training Loop

Training is a repeated cycle. Each cycle (called a "step") does the same four things:

### Step 1: Get a Batch

Grab 20 random 384-token chunks from the training data. This gives us a batch of 7,680 tokens to learn from.

### Step 2: Forward Pass

Feed the batch through the model. The model predicts the next token at every position. We compare these predictions against the actual next tokens and calculate the **loss** — a number that measures how wrong the model was.

### Step 3: Backward Pass

Calculate how each of the 118 million parameters contributed to the error. This uses a technique called **backpropagation** — it traces the error backward through the model, computing a "gradient" for each parameter that says "adjusting this parameter by this much would reduce the error."

### Step 4: Update

Adjust all 118 million parameters slightly in the direction that reduces the error. The amount of adjustment is controlled by the **learning rate**.

Then go back to Step 1 and repeat.

## How Long It Takes

| Property | Value |
|---|---|
| Tokens per step | 7,680 |
| Steps per second | ~1-2 |
| Time per step | ~500-1000 ms |
| Target total steps | 150,000 |
| Total training time | ~2-3 days |

## The Optimizer: AdamW

We use the **AdamW** optimizer, which is the standard for training Transformers. AdamW is smarter than simple gradient descent — it keeps track of the history of gradients and uses that to make more informed updates.

Key settings:
- Learning rate: starts at 3e-4 (0.0003)
- Weight decay: 0.1 (prevents parameters from growing too large)
- Betas: (0.9, 0.95) (momentum parameters)

## Mixed Precision Training

Our model uses **bfloat16** (brain floating point 16-bit) during the forward pass. This uses half the memory compared to full 32-bit precision, and modern GPUs compute bfloat16 operations twice as fast.

The key numbers (like the loss and gradient accumulation) are kept in full 32-bit precision for accuracy. Only the main computation uses bfloat16.

## Gradient Clipping

If the gradients become very large (due to a bad batch or numerical instability), updating the parameters by that much could destroy the model. **Gradient clipping** limits the total gradient magnitude to 1.0:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

This acts as a safety valve — large gradients get scaled down, preventing catastrophic updates.

## What "Learning" Actually Means

The model starts with 118 million random numbers. After 150,000 steps, those numbers have been adjusted so that the model can predict the next word with reasonable accuracy.

The model does not "memorize" the training text. Instead, it learns **patterns** — grammar rules, word associations, topic structures — that generalize to new text it has never seen before. This is what we test with the validation set.
