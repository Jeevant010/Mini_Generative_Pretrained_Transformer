# Chapter 5.5 — Mixed Precision Training

## The Idea

Every number in the model is stored in a specific format. The traditional format is **float32** (32-bit floating point) — each number takes 4 bytes and can represent values with about 7 decimal digits of precision.

**Mixed precision** uses a smaller format — **bfloat16** (16-bit brain floating point) — for most of the computation. Each number takes only 2 bytes and has about 3 decimal digits of precision.

## Why It Helps

### 1. Half the Memory

The model has 118 million parameters. In float32, that is:

```
118,000,000 × 4 bytes = 472 MB
```

In bfloat16:

```
118,000,000 × 2 bytes = 236 MB
```

The activations (intermediate values during computation) also shrink by half. This is critical for fitting the model on a GPU with limited VRAM.

### 2. Twice the Speed

Modern NVIDIA GPUs (like our RTX 4060) have specialized hardware for bfloat16 computation. They can perform roughly 2× more bfloat16 operations per second compared to float32.

### 3. Same Quality

The reduced precision of bfloat16 does not noticeably affect model quality. This is because:
- bfloat16 has the same exponent range as float32 (it can represent very large and very small numbers)
- Only the mantissa (precision) is reduced
- Neural networks are naturally robust to small numerical errors

## What Is "Mixed" About It?

Not everything uses bfloat16. Some operations need full precision:

| Operation | Precision | Why |
|---|---|---|
| Forward pass (attention, SwiGLU) | bfloat16 | Speed and memory |
| Loss calculation | float32 | Loss values need accuracy |
| Gradient accumulation | float32 | Small gradients can vanish in bfloat16 |
| Optimizer (AdamW updates) | float32 | Parameter updates need precision |
| Weight storage (master copy) | float32 | Prevents drift from many small updates |

The "mix" is: compute fast in bfloat16, but keep the important bookkeeping in float32.

## How It Works in Code

PyTorch makes this easy with automatic mixed precision (AMP):

```python
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits, loss = model(x, y)  # This runs in bfloat16
```

The `autocast` context manager automatically converts operations to bfloat16 where safe, and keeps them in float32 where needed.

## bfloat16 vs float16

There are two 16-bit formats:

| Format | Exponent bits | Mantissa bits | Range | Precision |
|---|---|---|---|---|
| float32 | 8 | 23 | ±3.4×10³⁸ | ~7 digits |
| float16 | 5 | 10 | ±65,504 | ~3 digits |
| bfloat16 | 8 | 7 | ±3.4×10³⁸ | ~2 digits |

**bfloat16 has the same range as float32** — it can represent very large and very small numbers. float16 has a much smaller range and can overflow (produce infinity) during training. This is why bfloat16 is preferred for language model training — it does not need the extra complexity of a loss scaler.
