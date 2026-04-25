# Transformer Architecture — Theoretical Foundations

## 1. Introduction

The Transformer architecture, introduced by Vaswani et al. (2017) in *"Attention Is All You Need"*, replaced recurrence-based sequence models with a fully attention-driven design. This project implements a **decoder-only** variant of the Transformer, which is the foundation of the GPT family of models (Radford et al., 2018, 2019; Brown et al., 2020).

This document describes the theoretical underpinnings of every architectural component used in this project's `model.py`, with mathematical formulations and design rationale.

---

## 2. Decoder-Only Architecture

### 2.1 Why Decoder-Only?

Encoder-decoder Transformers (e.g., T5, BART) are designed for sequence-to-sequence tasks. Decoder-only models are optimized for **autoregressive language modeling**: predicting the next token given all previous tokens.

Advantages of decoder-only for this project:

- Simpler architecture (no encoder, no cross-attention during training).
- Directly supports text generation via left-to-right sampling.
- Demonstrated to scale efficiently (GPT-3, LLaMA, Mistral).

### 2.2 High-Level Data Flow

```
Input Token IDs → Token Embedding → [Transformer Block × N] → Final Norm → LM Head → Logits
```

Each Transformer Block contains:

```
x → RMSNorm → GQA Attention → Residual Add → RMSNorm → SwiGLU FFN → Residual Add → output
```

This is a **pre-normalization** (Pre-LN) design, where normalization is applied before each sub-layer rather than after. Pre-LN has been shown to stabilize training in deep networks (Xiong et al., 2020).

---

## 3. Token Embedding and Weight Tying

### 3.1 Token Embedding

Each token ID $t \in \{0, 1, \ldots, V-1\}$ is mapped to a dense vector $\mathbf{e}_t \in \mathbb{R}^{d}$ via a learned embedding matrix:

$$\mathbf{E} \in \mathbb{R}^{V \times d}$$

where $V = 32{,}000$ is the vocabulary size and $d = 768$ is the embedding dimension.

### 3.2 Weight Tying

The output projection (LM head) shares the same weight matrix as the token embedding:

$$\mathbf{W}_{LM} = \mathbf{E}$$

This technique, proposed by Press & Wolf (2017), reduces parameter count by $V \times d$ parameters and has been shown to improve language modeling performance by coupling input and output representations.

In code: `self.token_embed.weight = self.lm_head.weight`

---

## 4. RMSNorm (Root Mean Square Layer Normalization)

### 4.1 Motivation

Standard Layer Normalization (Ba et al., 2016) computes both mean and variance. RMSNorm (Zhang & Sennrich, 2019) simplifies this by removing the mean-centering step, retaining only the root-mean-square normalization:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

where $\boldsymbol{\gamma} \in \mathbb{R}^d$ is a learned scale parameter and $\epsilon = 10^{-6}$ prevents division by zero.

### 4.2 Advantages

- **Faster**: Eliminates the mean computation and subtraction.
- **Stable**: Sufficient for preventing gradient explosion in deep networks.
- **Standard in modern LLMs**: Used in LLaMA, Mistral, and Gemma.

---

## 5. Rotary Positional Embeddings (RoPE)

### 5.1 The Position Encoding Problem

Transformers are permutation-invariant by default — without positional information, the model cannot distinguish token order. Three main approaches exist:

| Method | Type | Length Extrapolation |
|--------|------|---------------------|
| Learned Absolute | Additive | Poor |
| Sinusoidal (Vaswani) | Fixed, Additive | Moderate |
| RoPE (Su et al., 2021) | Rotary, Multiplicative | Good |

### 5.2 RoPE Formulation

RoPE encodes position by **rotating** query and key vectors in 2D subspaces. For position $m$ and dimension pair $(2i, 2i+1)$:

$$\begin{pmatrix} q_{2i}^{(m)} \\ q_{2i+1}^{(m)} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

where $\theta_i = 10000^{-2i/d_h}$ and $d_h$ is the head dimension.

### 5.3 Key Properties

1. **Relative position encoding**: The dot product $\langle \mathbf{q}_m, \mathbf{k}_n \rangle$ depends only on the relative position $m - n$.
2. **No additional parameters**: RoPE is computed, not learned.
3. **Length generalization**: Better extrapolation to unseen sequence lengths than learned absolute embeddings.

### 5.4 Implementation

In this project, RoPE is applied to **both queries and keys** after linear projection but before the attention computation:

```python
cos, sin = self.rope(seq_len, device, dtype)
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

The `rotate_half` function implements the rotation by splitting the last dimension and negating half:

$$\text{rotate\_half}(\mathbf{x}) = [-x_{d/2+1}, \ldots, -x_d, x_1, \ldots, x_{d/2}]$$

---

## 6. SwiGLU Feed-Forward Network

### 6.1 Standard FFN vs. SwiGLU

The original Transformer uses a two-layer FFN with ReLU:

$$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

SwiGLU (Shazeer, 2020) replaces this with a gated structure using the SiLU (Swish) activation:

$$\text{SwiGLU}(\mathbf{x}) = (\text{SiLU}(\mathbf{x} \mathbf{W}_1) \odot \mathbf{x} \mathbf{W}_2) \mathbf{W}_{out}$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$ and $\odot$ denotes element-wise multiplication.

### 6.2 Architecture Details

| Component | Shape |
|-----------|-------|
| $\mathbf{W}_1$ (gate) | $d \times h$ |
| $\mathbf{W}_2$ (up projection) | $d \times h$ |
| $\mathbf{W}_{out}$ (down projection) | $h \times d$ |

Hidden dimension: $h = \lfloor 3.5 \times d \rfloor = \lfloor 3.5 \times 768 \rfloor = 2688$

Note: SwiGLU uses **three** weight matrices instead of two, but the hidden dimension is typically reduced by a factor of $\frac{2}{3}$ relative to a standard FFN to keep total parameter count comparable. The `ffn_mult = 3.5` in this project accounts for this adjustment.

### 6.3 Why SwiGLU?

- Consistently outperforms ReLU and GELU FFNs at equivalent parameter budgets (Shazeer, 2020).
- Used in LLaMA, PaLM, Mistral, and most modern LLMs.
- The gating mechanism provides a learnable information filter within each layer.

---

## 7. Pre-Normalization Residual Connections

The Transformer Block uses the **Pre-LN** residual pattern:

$$\mathbf{x} = \mathbf{x} + \text{Attention}(\text{RMSNorm}(\mathbf{x}))$$
$$\mathbf{x} = \mathbf{x} + \text{SwiGLU}(\text{RMSNorm}(\mathbf{x}))$$

This differs from the original Post-LN design where normalization comes after the residual addition. Pre-LN:

- Prevents gradient magnitude issues in early layers.
- Allows stable training without careful learning rate warm-up (though warm-up is still beneficial).
- Is the standard in all recent large language models.

---

## 8. Autoregressive Training Objective

The model is trained with the standard **next-token prediction** objective (causal language modeling):

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)$$

where $T$ is the sequence length and $\theta$ are the model parameters.

This is implemented as cross-entropy loss between the model's logit predictions and the target token IDs:

```python
loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
```

### 8.1 Causal Masking

To prevent the model from attending to future tokens during training, a causal mask is applied within the attention computation. PyTorch's `F.scaled_dot_product_attention` with `is_causal=True` handles this efficiently using FlashAttention-compatible kernels.

---

## 9. Mixed-Precision Training

The training pipeline uses **bfloat16** mixed-precision via `torch.autocast`:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits, loss = model(xb, yb)
```

### 9.1 Why bfloat16?

| Property | float16 | bfloat16 |
|----------|---------|----------|
| Mantissa bits | 10 | 7 |
| Exponent bits | 5 | 8 |
| Dynamic range | Lower | Same as float32 |
| Precision | Higher | Lower |

bfloat16 matches float32's dynamic range, making it more numerically stable for training without requiring a loss scaler. This is the standard precision for modern GPU training.

---

## 10. Summary of Architectural Choices vs. Original Transformer

| Component | Original Transformer (2017) | This Project |
|-----------|-----------------------------|-------------|
| Architecture | Encoder-Decoder | Decoder-Only |
| Normalization | Post-LN LayerNorm | Pre-LN RMSNorm |
| Position Encoding | Sinusoidal (additive) | RoPE (rotary) |
| Attention | Multi-Head (full KV) | Grouped-Query (4 KV heads) |
| FFN | ReLU, 2 matrices | SwiGLU, 3 matrices |
| Bias terms | Yes | No (bias=False) |
| Weight tying | Optional | Yes (embed ↔ LM head) |
| Precision | float32 | bfloat16 mixed |

---

## 11. References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training." *OpenAI*.
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI*.
4. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS*.
5. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
6. Zhang, B. & Sennrich, R. (2019). "Root Mean Square Layer Normalization." *NeurIPS*.
7. Shazeer, N. (2020). "GLU Variants Improve Transformer." *arXiv:2002.05202*.
8. Press, O. & Wolf, L. (2017). "Using the Output Embedding to Improve Language Models." *EACL*.
9. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." *ICML*.
10. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." *arXiv:2302.13971*.
11. Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *arXiv:2305.13245*.
