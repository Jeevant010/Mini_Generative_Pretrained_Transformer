# Attention Mechanisms — Theoretical Deep-Dive

## 1. Overview

Attention is the core mechanism that allows Transformer models to dynamically weight the importance of different tokens in a sequence. This project implements and explores **five** attention variants across the research notebooks and production code.

This document provides mathematical formulations, computational complexity analysis, memory trade-offs, and practical guidance for each variant.

---

## 2. Scaled Dot-Product Attention (Foundation)

All attention variants in this project build on the same fundamental operation:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

where:
- $\mathbf{Q} \in \mathbb{R}^{T_q \times d_k}$ — Query matrix
- $\mathbf{K} \in \mathbb{R}^{T_k \times d_k}$ — Key matrix  
- $\mathbf{V} \in \mathbb{R}^{T_k \times d_v}$ — Value matrix
- $d_k$ — Key/query dimension (used for scaling)
- $T_q, T_k$ — Query and key sequence lengths

### 2.1 Why Scale by $\sqrt{d_k}$?

Without scaling, the dot products $\mathbf{Q}\mathbf{K}^\top$ grow in magnitude proportionally to $d_k$, pushing the softmax into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ keeps the variance of the dot products approximately 1, ensuring healthy gradient flow.

---

## 3. Type 1: Multi-Head Self-Attention (MHA)

### 3.1 Formulation

Multi-Head Attention (Vaswani et al., 2017) projects the input into $h$ parallel "heads", each computing attention independently:

$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_i^Q, \mathbf{X}\mathbf{W}_i^K, \mathbf{X}\mathbf{W}_i^V)$$

$$\text{MHA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O$$

where:
- $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \in \mathbb{R}^{d \times d_k}$
- $\mathbf{W}^O \in \mathbb{R}^{d \times d}$
- $d_k = d / h$ (head dimension)

### 3.2 Properties

- Each head can learn different relational patterns (e.g., syntactic, semantic, positional).
- Bidirectional: every token attends to every other token.
- Suitable for encoder-style tasks (BERT, classification).

### 3.3 Complexity

| Metric | Value |
|--------|-------|
| Time complexity | $O(T^2 \cdot d)$ |
| KV memory per head | $O(T \cdot d_k)$ |
| Total KV memory | $O(h \cdot T \cdot d_k) = O(T \cdot d)$ |
| Parameters | $4 \cdot d^2$ (Q, K, V, O projections) |

---

## 4. Type 2: Masked Causal Self-Attention

### 4.1 Formulation

Identical to MHA, but with a **causal mask** $\mathbf{M}$ that prevents token $t$ from attending to any token $t' > t$:

$$\mathbf{M}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

$$\text{CausalAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$

### 4.2 Why This Is Required

In autoregressive language modeling, the training objective is to predict token $t$ from tokens $1, \ldots, t-1$. If the model could see token $t$ (or future tokens) during training, it would trivially copy the answer instead of learning to predict.

Causal masking enforces the constraint that information flows only from past to present, making training consistent with the generation procedure.

### 4.3 Implementation

PyTorch's `F.scaled_dot_product_attention(q, k, v, is_causal=True)` automatically applies an efficient causal mask using FlashAttention-compatible fused kernels when available.

---

## 5. Type 3: Multi-Query Attention (MQA)

### 5.1 Formulation (Shazeer, 2019)

MQA reduces the number of key and value heads to **one**, while keeping multiple query heads:

$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_i^Q, \mathbf{X}\mathbf{W}^K, \mathbf{X}\mathbf{W}^V)$$

All $h$ query heads share a **single** key projection $\mathbf{W}^K$ and a **single** value projection $\mathbf{W}^V$.

### 5.2 Advantages

- **KV cache reduction**: During autoregressive generation, only 1 KV pair is stored per layer instead of $h$, reducing memory by $h\times$.
- **Faster inference**: Fewer memory reads for KV during generation.
- **Acceptable quality**: MQA quality is close to MHA for many tasks, especially when the model is large.

### 5.3 Disadvantages

- Some quality degradation compared to full MHA, especially for smaller models.
- The single KV head becomes a bottleneck for learning diverse relational patterns.

### 5.4 Complexity

| Metric | MHA | MQA |
|--------|-----|-----|
| KV parameters | $2 \cdot d^2$ | $2 \cdot d \cdot d_k$ |
| KV cache (generation) | $O(h \cdot T \cdot d_k)$ | $O(T \cdot d_k)$ |
| Reduction factor | — | $h\times$ less KV memory |

---

## 6. Type 4: Grouped-Query Attention (GQA) — **Used in Production**

### 6.1 Formulation (Ainslie et al., 2023)

GQA is a generalization that sits between MHA and MQA. Query heads are divided into $g$ groups, where each group of $h/g$ query heads shares one key-value head:

$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_i^Q, \mathbf{X}\mathbf{W}_{g(i)}^K, \mathbf{X}\mathbf{W}_{g(i)}^V)$$

where $g(i) = \lfloor i \cdot n_{kv} / h \rfloor$ maps query head $i$ to its KV group.

### 6.2 This Project's Configuration

| Parameter | Value |
|-----------|-------|
| Query heads ($h$) | 12 |
| KV heads ($n_{kv}$) | 4 |
| Heads per KV group | 3 |
| Head dimension | 64 |

This means every 3 query heads share 1 key-value head.

### 6.3 Implementation Detail

The KV heads are expanded to match the query head count via `repeat_interleave`:

```python
if self.n_kv_heads != self.n_heads:
    k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
    v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
```

### 6.4 Trade-off Spectrum

```
Full MHA ←————————— GQA ——————————→ MQA
h KV heads        g KV heads        1 KV head
Max quality       Balanced           Max efficiency
Max KV memory     Moderate           Min KV memory
```

### 6.5 Why GQA Was Chosen

- Best quality-to-efficiency ratio for RTX 4060-class hardware.
- 3× KV memory reduction vs. MHA (4 KV heads vs. 12).
- Negligible quality loss compared to full MHA at this model scale.
- Used in LLaMA 2 (70B), Mistral 7B, and Gemma.

---

## 7. Bonus: Cross-Attention

### 7.1 Formulation

Cross-attention allows one sequence to attend to another:

$$\text{CrossAttention}(\mathbf{X}_{decoder}, \mathbf{X}_{encoder}) = \text{Attention}(\mathbf{X}_{decoder}\mathbf{W}^Q, \mathbf{X}_{encoder}\mathbf{W}^K, \mathbf{X}_{encoder}\mathbf{W}^V)$$

Queries come from the decoder, while keys and values come from the encoder.

### 7.2 Use Cases

- Encoder-decoder models (T5, BART).
- Image-conditioned text generation.
- Retrieval-augmented generation.

### 7.3 Status in This Project

Cross-attention is implemented in the research notebooks for completeness but is **not used** in the production decoder-only pipeline.

---

## 8. Comparative Summary

| Variant | KV Heads | KV Params | KV Cache | Quality | Used In |
|---------|----------|-----------|----------|---------|---------|
| MHA | $h$ | $2d^2$ | $O(hTd_k)$ | Highest | GPT-2, BERT |
| Causal MHA | $h$ | $2d^2$ | $O(hTd_k)$ | Highest | GPT-2 |
| MQA | 1 | $2dd_k$ | $O(Td_k)$ | Good | PaLM |
| GQA | $g$ | $2gdd_k$ | $O(gTd_k)$ | Very Good | LLaMA 2, **This project** |
| Cross-Attn | $h$ | $2d^2$ | N/A | N/A | T5 |

---

## 9. FlashAttention and Fused Kernels

This project leverages PyTorch's `F.scaled_dot_product_attention`, which automatically dispatches to the most efficient kernel available:

1. **FlashAttention-2** (Dao, 2023): Tiled, memory-efficient attention that avoids materializing the full $T \times T$ attention matrix.
2. **Math fallback**: Standard computation when FlashAttention is unavailable.

Benefits of FlashAttention:
- Memory: $O(T)$ instead of $O(T^2)$ for the attention matrix.
- Speed: 2–4× faster for long sequences.
- Automatically used when `is_causal=True` is set.

---

## 10. References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." *arXiv:1911.02150*.
3. Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *arXiv:2305.13245*.
4. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *arXiv:2307.08691*.
5. Touvron, H., et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." *arXiv:2307.09288*.
