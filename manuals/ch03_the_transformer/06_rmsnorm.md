# Chapter 3.6 — RMSNorm: Keeping Numbers Stable

## The Problem

During training, numbers flow through 12 Transformer blocks. At each block, numbers get multiplied, added, and transformed. Without any control, these numbers can grow extremely large or shrink to nearly zero. This is called the **numerical instability** problem.

When numbers get too large, the model crashes with "NaN" (Not a Number) errors. When they get too small, the model stops learning because gradients vanish.

## What Normalization Does

Normalization rescales the numbers at each layer so they stay in a manageable range. It is like a thermostat for numbers — if they get too hot (large), it cools them down; if they get too cold (small), it warms them up.

## RMSNorm vs LayerNorm

There are two common types of normalization:

**LayerNorm** (used in original Transformers):
1. Compute the mean of all 768 values
2. Subtract the mean from each value (centering)
3. Compute the standard deviation
4. Divide by the standard deviation (scaling)
5. Multiply by a learnable scale parameter

**RMSNorm** (used in our model):
1. Compute the root-mean-square of all 768 values
2. Divide by that value (scaling)
3. Multiply by a learnable scale parameter

Notice: RMSNorm **skips the centering step**. It does not subtract the mean. This makes it simpler and faster, while performing equally well in practice.

## How RMSNorm Works

Given a vector of 768 numbers:

1. Square each number
2. Compute the average of all squared numbers
3. Take the square root (this is the "RMS" — Root Mean Square)
4. Divide each original number by the RMS value
5. Multiply by a learnable scale parameter

After this operation, the values are roughly in the range [-1, 1], regardless of what they were before.

## The Math (Optional)

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}
$$

$$
\text{RMSNorm}(x)_i = g_i \cdot \frac{x_i}{\text{RMS}(x)}
$$

Where $g$ is a learnable scale vector (768 values) and $\epsilon = 10^{-6}$ prevents division by zero.

## Where RMSNorm Is Applied

In our model, RMSNorm is applied at two points in each Transformer block:

1. **Before attention**: Normalize, then run attention
2. **Before feed-forward**: Normalize, then run SwiGLU

This is called **pre-normalization**. Older models used post-normalization (normalize after attention/feed-forward), but pre-normalization has been shown to make training more stable.

There is also one final RMSNorm after all 12 blocks, right before the output projection.

## What Happens Without RMSNorm

In our ablation studies (Chapter 8), we tested disabling RMSNorm. Without it:

- Gradient norms become very large and unstable
- Training loss fluctuates wildly
- The model may crash with NaN errors
- If it survives, quality is worse

RMSNorm is not optional — it is essential for stable training.

## Parameters

RMSNorm has very few parameters — just the learnable scale vector $g$:

- Per RMSNorm layer: 768 parameters
- 2 per Transformer block × 12 blocks + 1 final = 25 RMSNorm layers
- Total: 25 × 768 = **19,200 parameters** (tiny compared to the full model)
