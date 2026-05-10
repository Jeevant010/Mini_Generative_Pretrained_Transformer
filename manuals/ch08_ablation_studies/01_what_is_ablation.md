# Chapter 8.1 — What Is an Ablation Study?

## The Idea

An ablation study is one of the most powerful tools in machine learning research. The concept is simple: **remove one component at a time and measure what happens.**

The word "ablation" comes from medicine, where it means removing tissue to study its function. In ML, we "remove" a model component (by disabling it) to understand how much it contributes.

## Why Ablation Studies Matter

When we build a model with RMSNorm, RoPE, GQA, SwiGLU, and Flash Attention, a natural question is: **Are all of these necessary?**

Maybe the model would work just as well without GQA. Maybe RoPE does not actually help. The only way to know is to test — and ablation is the rigorous way to do it.

## How We Run Ablations

Our `config.py` has toggle switches for each component:

| Toggle | Default | What It Controls |
|---|---|---|
| `USE_RMSNORM` | True | RMSNorm normalization layers |
| `USE_ROPE` | True | Rotary Positional Embeddings |
| `USE_FLASH_ATTENTION` | True | PyTorch Flash Attention kernel |
| `USE_GQA` | True | Grouped-Query Attention (4 KV heads vs 12) |

To run an ablation, change exactly **one** toggle to False, train for a fixed number of steps (e.g., 5,000), and compare results against the full model.

### The Rules

1. **Change one thing at a time.** If you disable two components, you cannot tell which one caused the difference.
2. **Use the same data.** Every ablation must use the same training and validation data.
3. **Use the same random seed.** This ensures differences come from the component, not from randomness.
4. **Use the same step count.** Train each variant for the same number of steps.
5. **Measure multiple metrics.** Do not just look at loss — check perplexity, generation quality, speed, and memory.

## What We Expect

| Ablation | Expected Effect |
|---|---|
| Remove RMSNorm | Training becomes unstable. Large gradient norms. Possible NaN crash. |
| Remove RoPE | Model loses positional awareness. Grammar degrades. |
| Remove Flash Attention | Same quality, but slower and more memory. |
| Remove GQA (use full MHA) | More parameters, more memory, possibly slightly better quality. |

## Why This Matters for Understanding

Ablation studies turn "we used RMSNorm because it is modern" into "we used RMSNorm because without it, training loss diverges after 500 steps." The second statement is scientific — it is backed by evidence.

For your project report or presentation, ablation results are the strongest evidence that your architecture choices are justified.

## How to Read the Results Table

After running all ablations, the results table looks like:

| Variant | Val Loss | PPL | Tok/s | VRAM | Status |
|---|---|---|---|---|---|
| Full model | 3.52 | 33.7 | 7600 | 2.5GB | Stable |
| No RMSNorm | — | — | — | — | NaN at step 500 |
| No RoPE | 4.10 | 60.3 | 7600 | 2.4GB | Stable but poor |
| No Flash Attn | 3.52 | 33.7 | 4200 | 3.8GB | Stable, slower |
| Full MHA | 3.48 | 32.5 | 6800 | 3.1GB | Stable, better |

This table tells a clear story:
- **RMSNorm is essential** — without it, training crashes
- **RoPE is essential** — without it, quality drops significantly
- **Flash Attention is an efficiency optimization** — same quality, but 1.8× faster
- **GQA is a parameter trade-off** — slightly worse quality, but fewer parameters and less memory

> **Note:** The values above are illustrative. Run the actual ablations to fill in real numbers for your project.
