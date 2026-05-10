# Chapter 8.2 — Component Ablation Details

## RMSNorm Ablation

**What changes:** Remove all normalization layers from the model. The residual blocks become:

```
u = h + Attention(h)          (no normalization before attention)
h_next = u + SwiGLU(u)        (no normalization before SwiGLU)
```

**Why it matters:** Without normalization, activations grow uncontrollably through 12 layers. Each residual addition increases the magnitude:

```
Layer 1 output magnitude: ~1.0
Layer 6 output magnitude: ~50.0
Layer 12 output magnitude: ~10,000.0   ← numerical overflow territory
```

**Expected result:** Gradient norms spike, loss becomes NaN within a few hundred steps. This proves RMSNorm is load-bearing — the model literally cannot train without it.

**How to run:**
```python
# In config.py:
USE_RMSNORM = False
```

---

## RoPE Ablation

**What changes:** Attention is computed without any positional information. Queries and keys are not rotated.

**Why it matters:** Without positional encoding, the model treats "The cat ate the fish" and "The fish ate the cat" as identical — same words, no way to distinguish order.

**Expected result:** The model can still learn unigram and local bigram statistics (which words are common), but it loses grammar. Generated text has real words in random order. Validation loss is significantly worse (~4.0 vs ~3.5).

**How to run:**
```python
# In config.py:
USE_ROPE = False
```

---

## Flash Attention Ablation

**What changes:** Switches from PyTorch's optimized `scaled_dot_product_attention` to a manual implementation that builds the full attention matrix.

**Why it matters:** Flash Attention is an efficiency optimization, not a modeling change. The mathematical result is identical.

**Expected result:** Same validation loss and perplexity. Slower throughput (fewer tokens/sec). Higher VRAM usage (the full T×T attention matrix is materialized).

**How to run:**
```python
# In config.py:
USE_FLASH_ATTENTION = False
```

---

## GQA Ablation

**What changes:** Switches from 4 KV heads (Grouped-Query Attention) to 12 KV heads (full Multi-Head Attention).

**Why it matters:** Full MHA has more parameters per layer (3× more KV projection parameters), which gives the model more capacity but uses more memory.

**Expected result:** Slightly better quality (lower perplexity). Higher VRAM. More total parameters (the model grows from ~118M to ~127M). The quality difference is usually small — GQA trades a tiny quality loss for significant memory savings.

**How to run:**
```python
# In config.py:
USE_GQA = False
```

---

## Running a Full Ablation Study

For a paper-quality ablation study:

1. Use the `wizard_of_oz_smoke` preset for quick tests (small data, few steps)
2. Use a fixed budget of 2,000-5,000 steps on the same data slice
3. Record: final train loss, final val loss, PPL, tokens/sec, peak VRAM, gradient norm, generation samples
4. Use the same random seed for all runs
5. Report results in a table with the full model as the baseline
