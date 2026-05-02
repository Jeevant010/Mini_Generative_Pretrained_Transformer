# Transformer Architecture

## Decoder-Only Transformer

The project implements a decoder-only Transformer. Given a token sequence:

$$
x_1, x_2, ..., x_T
$$

the model estimates:

$$
P(x_1, ..., x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

This is the standard autoregressive language-modeling objective. During training, the input sequence is shifted by one position to form targets:

$$
x = [x_1, ..., x_T]
$$

$$
y = [x_2, ..., x_{T+1}]
$$

The model predicts every target token from all previous visible input tokens.

## Shape Notation

| Symbol | Meaning | Current value |
| --- | --- | ---: |
| `B` | Batch size | 20 |
| `T` | Context length | 384 |
| `V` | Vocabulary size | 32,000 |
| `d` | Embedding width | 768 |
| `L` | Number of blocks | 12 |
| `H_q` | Query heads | 12 |
| `H_kv` | Key-value heads | 4 |
| `d_h` | Head dimension | 64 |

The token batch has shape:

$$
X \in \mathbb{N}^{B \times T}
$$

The embedding output has shape:

$$
H^{(0)} \in \mathbb{R}^{B \times T \times d}
$$

## Token Embedding

Each token ID maps to a trainable vector:

$$
h_t^{(0)} = E[x_t]
$$

where:

$$
E \in \mathbb{R}^{V \times d}
$$

For this model:

$$
Vd = 32000 \times 768 = 24{,}576{,}000
$$

embedding parameters.

## Transformer Block

Each block uses pre-normalization residual structure:

$$
u^{(l)} = h^{(l)} + \operatorname{Attn}(\operatorname{RMSNorm}(h^{(l)}))
$$

$$
h^{(l+1)} = u^{(l)} + \operatorname{SwiGLU}(\operatorname{RMSNorm}(u^{(l)}))
$$

Pre-normalization makes training more stable because the residual stream remains a direct path through the network.

## RMSNorm

RMSNorm normalizes by root mean square:

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}
$$

$$
\operatorname{RMSNorm}(x)_i = g_i\frac{x_i}{\operatorname{RMS}(x)}
$$

where `g` is a learned scale vector. The implementation uses `eps = 1e-6`.

RMSNorm avoids mean subtraction, making it cheaper than LayerNorm:

$$
\operatorname{LayerNorm}(x)_i =
\gamma_i \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

## Attention

The model uses causal scaled dot-product attention:

$$
S = \frac{QK^T}{\sqrt{d_h}}
$$

The causal mask is:

$$
M_{ij} =
\begin{cases}
0, & j \leq i \\
-\infty, & j > i
\end{cases}
$$

The attention weights are:

$$
A = \operatorname{softmax}(S + M)
$$

The attention output is:

$$
O = AV
$$

The current implementation uses PyTorch `F.scaled_dot_product_attention(..., is_causal=True)` when `USE_FLASH_ATTENTION = True`.

## Feed-Forward Layer

The feed-forward network uses SwiGLU:

$$
\operatorname{SwiGLU}(x) =
W_{out}(\operatorname{SiLU}(xW_1) \odot xW_2)
$$

where:

$$
\operatorname{SiLU}(z) = z\sigma(z)
$$

and:

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

The hidden dimension is:

$$
d_{ff} = \lfloor 3.5d \rfloor = 2688
$$

## Final Projection

After all blocks, the model applies final RMSNorm:

$$
z = \operatorname{RMSNorm}(h^{(L)})
$$

and computes logits:

$$
\ell = zW_{lm}
$$

where:

$$
W_{lm} \in \mathbb{R}^{d \times V}
$$

The model ties:

$$
W_{lm} = E^T
$$

so the token embedding and output classifier share the same parameters.

## Parameter Summary

| Component | Parameters |
| --- | ---: |
| Token embedding and tied LM head | 24,576,000 |
| All Transformer blocks | 93,210,624 |
| Final RMSNorm | 768 |
| Total | 117,787,392 |

The model is therefore a 117.8M parameter decoder-only GPT-style language model.

