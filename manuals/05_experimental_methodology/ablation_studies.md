# Ablation Studies — Methodology, Toggle Reference & Expected Results

## 1. What Are Ablation Studies?

An ablation study systematically **removes or disables** individual components of a system to measure their contribution. In deep learning, this proves that each architectural choice is mathematically necessary — not just copied from a tutorial.

> **The difference between a student project and rigorous research:**
> - Student: "I built a Transformer and it works."
> - Researcher: "I built a Transformer, systematically isolated every variable, and proved exactly why it works."

---

## 2. Ablation Toggles in This Project

All toggles are defined in `config.py` and read by `model.py` at construction time:

| Toggle | Default | What It Disables | Expected Failure Mode |
|--------|---------|------------------|----------------------|
| `USE_RMSNORM` | `True` | Pre-normalization in Transformer blocks | Gradient explosion → NaN loss within 100-500 steps |
| `USE_ROPE` | `True` | Rotary Positional Embeddings on Q/K | Order-blindness → flat perplexity, broken grammar |
| `USE_FLASH_ATTENTION` | `True` | PyTorch SDPA → manual matmul attention | 2-3× slower, 2× more VRAM (functionally identical) |
| `USE_GQA` | `True` | Grouped-Query → Full Multi-Head Attention | More VRAM, slightly better quality at small scale |

---

## 3. How To Run Ablation Studies

### 3.1 Automated (Recommended)

Run the full ablation suite on `wizard_of_oz.txt` — takes ~5 minutes:

```bash
python -m ablation.run_ablation --steps 100
```

This produces a comparison table like:

```
Config                    |     Loss |      PPL |    Tok/s |  VRAM MB | Grad Norm | Status
------------------------------------------------------------------------------------------
Full Baseline (all ON)    |   7.1234 |   1241.3 |   45,000 |     3200 |      1.23 | ✅ Stable
No RMSNorm                |      NaN |        ∞ |   44,000 |     3180 |        ∞  | 💥 Exploded
No RoPE                   |   7.8456 |   2546.8 |   46,000 |     3100 |      1.18 | ⚠️ Bad grammar
No Flash Attention         |   7.1189 |   1235.6 |   22,000 |     7800 |      1.24 | 🐢 2× slower
Full MHA (no GQA)          |   7.0987 |   1206.1 |   38,000 |     4100 |      1.25 | ✅ More VRAM
```

### 3.2 Manual (Individual Toggle)

Edit `config.py`:

```python
USE_RMSNORM = False  # ← Flip one toggle at a time
```

Then run training:

```bash
python training.py
```

Watch the loss — you'll see the effect within minutes.

---

## 4. Expected Results — Detailed Analysis

### 4.1 No RMSNorm → Gradient Explosion

**What happens**: Without normalization before the attention and FFN sub-layers, each residual connection compounds the variance of the activations. After a few hundred steps, the floating-point numbers overflow.

**Mathematical explanation**: The residual connection computes $x_{l+1} = x_l + f(x_l)$. Without normalization, $\|x_l\|$ grows unboundedly:

$$\|x_L\| \approx \|x_0\| + \sum_{l=0}^{L-1} \|f(x_l)\|$$

With 12 layers, the variance accumulates rapidly, leading to numerical overflow.

**In the code**: When `USE_RMSNORM = False`, `model.py` replaces `RMSNorm` with `Identity` (a no-op pass-through).

**What to report**: Loss curve showing stable training → sudden NaN. Screenshot of the exact step where it explodes. Gradient norm spiking to infinity.

---

### 4.2 No RoPE → Order-Blind Model

**What happens**: Self-attention is permutation-equivariant by default. Without positional encoding, the model treats "dog bites man" and "man bites dog" as mathematically identical inputs.

**Observable symptoms**:
- Loss decreases (the model still learns vocabulary frequencies).
- But generated text has random word order.
- Perplexity plateaus higher than baseline.
- Grammar quality is severely degraded.

**In the code**: When `USE_ROPE = False`, the `apply_rope()` calls are skipped. Q and K are used directly without rotation.

**What to report**: Side-by-side text samples from baseline vs. no-RoPE. Perplexity comparison table. The model becomes a "bag of words".

---

### 4.3 No Flash Attention → VRAM Explosion

**What happens**: Standard attention materializes the full $T \times T$ attention matrix in GPU memory. Flash Attention (Dao, 2023) avoids this by computing attention in tiles.

**Observable symptoms**:
- **Identical loss and perplexity** (mathematically equivalent).
- **2-3× slower** per step (memory bandwidth bottleneck).
- **2× more VRAM** usage (the attention matrix is $B \times H \times T \times T$).

**In the code**: When `USE_FLASH_ATTENTION = False`, `model.py` uses `manual_causal_attention()` which explicitly computes `Q @ K.T`, applies a mask, softmax, and `@ V`.

**What to report**: VRAM comparison table. Tokens/sec comparison. This proves the hardware optimization, not model quality.

---

### 4.4 Full MHA vs. GQA → VRAM vs. Quality Trade-off

**What happens**: Full MHA uses 12 KV heads (one per query head). GQA uses 4 KV heads shared across groups. Full MHA is slightly higher quality but uses more memory.

**Observable symptoms**:
- Full MHA may achieve slightly lower loss (more capacity in attention).
- Full MHA uses ~25% more VRAM for the KV projections.
- At this model scale (~85M params), the difference is small.

**In the code**: When `USE_GQA = False`, the attention module sets `n_kv_heads = n_head`, making it standard multi-head attention.

**What to report**: Parameter count comparison, VRAM comparison, loss comparison after N steps.

---

## 5. Interpreting Results for a Paper

### 5.1 The Ablation Table

The comparison table from `ablation/run_ablation.py` is designed to be copy-pasted directly into a LaTeX or Markdown paper. It proves:

1. **RMSNorm is load-bearing** — remove it and training collapses.
2. **RoPE is essential** — remove it and the model loses language structure.
3. **Flash Attention is a hardware optimization** — identical quality, 2× efficiency.
4. **GQA is a memory optimization** — small quality trade-off for significant VRAM savings.

### 5.2 What Reviewers Look For

| ✅ Strong | ❌ Weak |
|-----------|---------|
| Quantitative comparison (PPL, VRAM, tok/s) | "I tried removing X and it broke" |
| Multiple runs with standard deviation | Single run, no error bars |
| Clear causal explanation (math + evidence) | "It just didn't work" |
| Control variables (same data, same steps) | Different data or steps per config |

---

## 6. Files Reference

| File | Purpose |
|------|---------|
| `config.py` — ablation toggles | `USE_RMSNORM`, `USE_ROPE`, `USE_FLASH_ATTENTION`, `USE_GQA` |
| `model.py` — conditional paths | `Identity` class, `manual_causal_attention()`, toggle checks |
| `ablation/run_ablation.py` | Automated runner, comparison table, JSON results |
| `logs/ablation/ablation_results.json` | Machine-readable results for plotting |
