# Model Architecture — Layer-by-Layer Specification & Parameter Budget

## 1. Architecture Summary

The model is a **decoder-only Transformer** with the following specification:

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Embedding dimension | $d$ | 768 |
| Number of layers | $L$ | 12 |
| Query heads | $h$ | 12 |
| Key-Value heads (GQA) | $h_{kv}$ | 4 |
| Head dimension | $d_h = d/h$ | 64 |
| FFN multiplier | $m$ | 3.5 |
| FFN hidden dimension | $h_{ffn} = \lfloor m \cdot d \rfloor$ | 2688 |
| Vocabulary size | $V$ | 32,000 |
| Context length | $T$ | 384 |
| Dropout rate | $p$ | 0.1 |

---

## 2. Layer-by-Layer Breakdown

### 2.1 Token Embedding

```
nn.Embedding(V, d) = nn.Embedding(32000, 768)
```

**Parameters**: $V \times d = 32{,}000 \times 768 = 24{,}576{,}000$

> **Note**: This weight is **tied** with the LM Head. The 24.6M parameters are counted only once.

---

### 2.2 Transformer Block (× 12)

Each of the 12 identical blocks contains:

#### A) Pre-Attention RMSNorm

```
RMSNorm(d) = RMSNorm(768)
```

**Parameters**: $d = 768$ (scale vector $\boldsymbol{\gamma}$)

#### B) Grouped-Query Attention

| Projection | Shape | Parameters |
|-----------|-------|------------|
| $\mathbf{W}^Q$ (q_proj) | $d \times (h \cdot d_h) = 768 \times 768$ | 589,824 |
| $\mathbf{W}^K$ (k_proj) | $d \times (h_{kv} \cdot d_h) = 768 \times 256$ | 196,608 |
| $\mathbf{W}^V$ (v_proj) | $d \times (h_{kv} \cdot d_h) = 768 \times 256$ | 196,608 |
| $\mathbf{W}^O$ (o_proj) | $(h \cdot d_h) \times d = 768 \times 768$ | 589,824 |
| **Attention total** | | **1,572,864** |

> No bias terms are used (`bias=False`). RoPE has no learnable parameters.

#### C) Pre-FFN RMSNorm

```
RMSNorm(d) = RMSNorm(768)
```

**Parameters**: $d = 768$

#### D) SwiGLU Feed-Forward Network

| Projection | Shape | Parameters |
|-----------|-------|------------|
| $\mathbf{W}_1$ (gate) | $d \times h_{ffn} = 768 \times 2688$ | 2,064,384 |
| $\mathbf{W}_2$ (up) | $d \times h_{ffn} = 768 \times 2688$ | 2,064,384 |
| $\mathbf{W}_{out}$ (down) | $h_{ffn} \times d = 2688 \times 768$ | 2,064,384 |
| **FFN total** | | **6,193,152** |

#### Per-Block Total

| Component | Parameters |
|-----------|------------|
| RMSNorm (×2) | 1,536 |
| GQA Attention | 1,572,864 |
| SwiGLU FFN | 6,193,152 |
| **Block total** | **7,767,552** |

#### All 12 Blocks

$$12 \times 7{,}767{,}552 = 93{,}210{,}624$$

---

### 2.3 Final RMSNorm

```
RMSNorm(d) = RMSNorm(768)
```

**Parameters**: $d = 768$

---

### 2.4 LM Head (Output Projection)

```
nn.Linear(d, V, bias=False) = nn.Linear(768, 32000, bias=False)
```

**Parameters**: $d \times V = 768 \times 32{,}000 = 24{,}576{,}000$

> **Weight tied** with Token Embedding — **zero additional parameters**.

---

## 3. Total Parameter Budget

| Component | Parameters | % of Total |
|-----------|------------|------------|
| Token Embedding (tied) | 24,576,000 | 20.85% |
| Transformer Blocks (×12) | 93,210,624 | 79.09% |
| Final RMSNorm | 768 | <0.01% |
| LM Head (tied, no extra) | 0 | 0% |
| **Grand Total** | **117,787,392** | 100% |
| **Unique / Trainable** | **~93,211,392** | — |

> Due to weight tying, the **effective trainable parameter count** excludes the duplicate LM Head weights. The actual count reported by `sum(p.numel() for p in model.parameters() if p.requires_grad)` accounts for tied weights being a single tensor.

---

## 4. Memory Estimates

### 4.1 Model Weights (float32)

$$93{,}211{,}392 \times 4 \text{ bytes} \approx 355 \text{ MB}$$

### 4.2 Model Weights (bfloat16 training)

Forward pass activations use bfloat16:

$$93{,}211{,}392 \times 2 \text{ bytes} \approx 178 \text{ MB}$$

### 4.3 Optimizer State (AdamW)

AdamW stores two additional copies (momentum + variance) in float32:

$$93{,}211{,}392 \times 4 \times 2 = 745 \text{ MB}$$

### 4.4 Gradient Storage

One gradient copy in float32:

$$93{,}211{,}392 \times 4 = 355 \text{ MB}$$

### 4.5 Activation Memory (approximate)

Per batch, per layer (rough estimate):

$$B \times T \times d \times 2 \text{ bytes} \times \text{activation factor}$$
$$= 20 \times 384 \times 768 \times 2 \times 4 \approx 47 \text{ MB per layer}$$

For 12 layers: ~560 MB (with gradient checkpointing this can be reduced).

### 4.6 Total Estimated GPU Memory

| Component | Size |
|-----------|------|
| Weights (mixed) | ~355 MB |
| Optimizer | ~745 MB |
| Gradients | ~355 MB |
| Activations | ~560 MB |
| **Estimated Total** | **~2.0 GB** |

This fits comfortably within the 8 GB VRAM of an RTX 4060.

---

## 5. FLOP Estimates

### 5.1 Per-Token FLOPs (Forward Pass)

Using the standard approximation of $2 \times N_{params}$ FLOPs per token for a forward pass:

$$\text{FLOPs}_{fwd} = 2 \times 93{,}211{,}392 \approx 186{,}422{,}784 \text{ FLOPs/token}$$

### 5.2 Per-Token FLOPs (Training, Forward + Backward)

The backward pass is approximately $2\times$ the forward pass:

$$\text{FLOPs}_{train} = 6 \times 93{,}211{,}392 \approx 559{,}268{,}352 \text{ FLOPs/token}$$

### 5.3 Per-Step FLOPs

$$\text{FLOPs}_{step} = \text{FLOPs}_{train} \times B \times T = 559{,}268{,}352 \times 20 \times 384 \approx 4.30 \times 10^{12}$$

### 5.4 Total Training FLOPs (300K steps)

$$\text{FLOPs}_{total} = 4.30 \times 10^{12} \times 300{,}000 \approx 1.29 \times 10^{18}$$

---

## 6. Comparison With Reference Models

| Model | Parameters | Layers | $d$ | Heads | KV Heads | FFN |
|-------|-----------|--------|-----|-------|----------|-----|
| GPT-2 Small | 124M | 12 | 768 | 12 | 12 (MHA) | GELU, 2 mat |
| **Mini GPT (ours)** | **~93M** | **12** | **768** | **12** | **4 (GQA)** | **SwiGLU, 3 mat** |
| LLaMA 7B | 6.7B | 32 | 4096 | 32 | 32 (MHA) | SwiGLU |
| Mistral 7B | 7.3B | 32 | 4096 | 32 | 8 (GQA) | SwiGLU |

Our model is architecturally closer to LLaMA/Mistral than to GPT-2, despite being much smaller.

---

## 7. Model Instantiation

```python
import config
from model import GPTLanguageModel

model = GPTLanguageModel(config)
# config is used as a namespace: config.n_embd, config.n_layer, etc.
```

The model accepts the config module directly as its constructor argument, reading all architectural parameters from it.
