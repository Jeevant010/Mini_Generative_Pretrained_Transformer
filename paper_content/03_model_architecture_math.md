# Model Architecture And Mathematics

## Decoder-Only Language Model

The model is a decoder-only Transformer. Its job is to estimate the probability of the next token given all previous tokens:

$$
P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

During training, the model receives a sequence:

$$
x = [x_1, x_2, ..., x_T]
$$

and predicts:

$$
y = [x_2, x_3, ..., x_{T+1}]
$$

The architecture is causal: token position `t` can attend only to positions `1` through `t`, never to future positions.

## Token Embedding

Each token ID is mapped to a learned vector:

$$
e_t = E[x_t]
$$

where:

- `E in R^{V x d}` is the embedding matrix.
- `V = 32000` is vocabulary size.
- `d = 768` is embedding dimension.

For a batch of token IDs:

$$
X \in \mathbb{N}^{B \times T}
$$

the embedding output is:

$$
H^{(0)} = E[X] \in \mathbb{R}^{B \times T \times d}
$$

The project does not use learned absolute position embeddings. Positional information is injected inside attention using RoPE.

## Transformer Block

Each Transformer block uses a pre-normalization residual structure:

$$
\tilde{H}^{(l)} = H^{(l)} + \operatorname{Attn}(\operatorname{RMSNorm}(H^{(l)}))
$$

$$
H^{(l+1)} = \tilde{H}^{(l)} + \operatorname{SwiGLU}(\operatorname{RMSNorm}(\tilde{H}^{(l)}))
$$

where `l` is the layer index. The current model uses `L = 12` blocks.

Pre-normalization is useful because the residual path remains direct. This generally improves optimization stability compared with placing normalization after the residual addition.

## RMSNorm

RMSNorm normalizes activations by their root mean square instead of subtracting the mean as LayerNorm does. For an input vector:

$$
x \in \mathbb{R}^{d}
$$

the root mean square is:

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$

RMSNorm is:

$$
\operatorname{RMSNorm}(x)_i = g_i \frac{x_i}{\operatorname{RMS}(x)}
$$

where:

- `g in R^d` is a learned scale vector.
- `epsilon = 1e-6` prevents division by zero.

In code:

```python
rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
return self.scale * x * rms
```

Compared with LayerNorm, RMSNorm removes mean subtraction:

$$
\operatorname{LayerNorm}(x)_i = \gamma_i \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

This makes RMSNorm simpler and cheaper while still controlling activation scale.

## Query, Key, And Value Projections

Attention begins with linear projections:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

For the current model:

- `d = 768`
- `H_q = 12`
- `H_kv = 4`
- `d_h = 64`

The query projection outputs:

$$
Q \in \mathbb{R}^{B \times H_q \times T \times d_h}
$$

The key and value projections output:

$$
K, V \in \mathbb{R}^{B \times H_{kv} \times T \times d_h}
$$

## Rotary Positional Embedding

RoPE rotates query and key vectors as a function of token position. For each pair of dimensions, a rotation angle is assigned:

$$
\theta_i = 10000^{-2i / d_h}
$$

For position `m`, the angle is:

$$
m\theta_i
$$

For a two-dimensional pair `(a, b)`, RoPE applies:

$$
\begin{bmatrix}
a' \\
b'
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
a \\
b
\end{bmatrix}
$$

The implementation uses:

$$
\operatorname{RoPE}(x) = x \odot \cos(\Theta) + \operatorname{rotate\_half}(x) \odot \sin(\Theta)
$$

where:

$$
\operatorname{rotate\_half}([x_1, x_2]) = [-x_2, x_1]
$$

RoPE is applied to `Q` and `K`, not to `V`. This lets attention scores depend on relative position through the dot product:

$$
\langle \operatorname{RoPE}(q_m), \operatorname{RoPE}(k_n) \rangle
$$

which encodes information about `m - n`.

## Causal Scaled Dot-Product Attention

For each attention head, raw scores are:

$$
S = \frac{QK^T}{\sqrt{d_h}}
$$

Causality is enforced using a mask:

$$
M_{ij} =
\begin{cases}
0, & j \leq i \\
-\infty, & j > i
\end{cases}
$$

The masked attention distribution is:

$$
A = \operatorname{softmax}(S + M)
$$

The attention output is:

$$
O = AV
$$

The scaling factor:

$$
\frac{1}{\sqrt{d_h}}
$$

prevents dot products from growing too large as head dimension increases.

## Grouped-Query Attention

Standard Multi-Head Attention uses the same number of query, key, and value heads:

$$
H_q = H_k = H_v
$$

Grouped-Query Attention uses fewer key-value heads:

$$
H_{kv} < H_q
$$

In this project:

$$
H_q = 12,\quad H_{kv} = 4
$$

Each key-value head is shared by:

$$
g = \frac{H_q}{H_{kv}} = \frac{12}{4} = 3
$$

query heads.

The implementation computes keys and values with 4 heads, then repeats them to match the 12 query heads:

```python
k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
```

This reduces key and value projection parameters. Full MHA would use:

$$
2d^2
$$

parameters for key and value projections. GQA uses:

$$
2d(H_{kv}d_h)
$$

For this model:

$$
2d^2 = 2(768)(768) = 1{,}179{,}648
$$

GQA key-value projection parameters:

$$
2(768)(4 \times 64) = 393{,}216
$$

So GQA saves:

$$
1{,}179{,}648 - 393{,}216 = 786{,}432
$$

parameters per layer in the key-value projections compared with full MHA.

## Attention Output Projection

After attention, heads are concatenated:

$$
O_{concat} \in \mathbb{R}^{B \times T \times (H_q d_h)}
$$

Since:

$$
H_qd_h = 12 \times 64 = 768
$$

the concatenated attention output has the same width as the model embedding. It is projected back with:

$$
Y = O_{concat}W_O
$$

where:

$$
W_O \in \mathbb{R}^{768 \times 768}
$$

## SwiGLU Feed-Forward Network

The feed-forward block uses SwiGLU:

$$
\operatorname{SwiGLU}(x) = W_{out}(\operatorname{SiLU}(xW_1) \odot xW_2)
$$

where:

$$
\operatorname{SiLU}(z) = z \cdot \sigma(z)
$$

and:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The hidden dimension is:

$$
d_{ff} = \lfloor ffn\_mult \times d \rfloor
$$

For this project:

$$
d_{ff} = \lfloor 3.5 \times 768 \rfloor = 2688
$$

The three feed-forward matrices are:

$$
W_1 \in \mathbb{R}^{768 \times 2688}
$$

$$
W_2 \in \mathbb{R}^{768 \times 2688}
$$

$$
W_{out} \in \mathbb{R}^{2688 \times 768}
$$

The parameter count per feed-forward block is:

$$
3 \times 768 \times 2688 = 6{,}193{,}152
$$

## Final Normalization And LM Head

After all Transformer blocks:

$$
H^{(L)} = \operatorname{TransformerBlocks}(H^{(0)})
$$

the model applies final RMSNorm:

$$
Z = \operatorname{RMSNorm}(H^{(L)})
$$

The logits are:

$$
\operatorname{logits} = ZW_{lm}
$$

where:

$$
W_{lm} \in \mathbb{R}^{d \times V}
$$

## Weight Tying

The implementation ties input embedding weights and output LM-head weights:

```python
self.token_embed.weight = self.lm_head.weight
```

Mathematically:

$$
W_{lm} = E^T
$$

This reduces parameters and forces the model to use the same vector space for reading and predicting tokens.

The tied embedding parameter count is:

$$
Vd = 32000 \times 768 = 24{,}576{,}000
$$

Without tying, the model would need another:

$$
dV = 768 \times 32000 = 24{,}576{,}000
$$

parameters for the output head.

## Parameter Count

Per Transformer block:

| Component | Parameters |
| --- | ---: |
| Query projection | 589,824 |
| Key projection | 196,608 |
| Value projection | 196,608 |
| Output projection | 589,824 |
| Attention total | 1,572,864 |
| SwiGLU `W1` | 2,064,384 |
| SwiGLU `W2` | 2,064,384 |
| SwiGLU `Wout` | 2,064,384 |
| FFN total | 6,193,152 |
| Two RMSNorm scales | 1,536 |
| Total per block | 7,767,552 |

All 12 blocks:

$$
12 \times 7{,}767{,}552 = 93{,}210{,}624
$$

Embedding and tied LM head:

$$
24{,}576{,}000
$$

Final RMSNorm:

$$
768
$$

Total:

$$
93{,}210{,}624 + 24{,}576{,}000 + 768 = 117{,}787{,}392
$$

Therefore, the model has:

$$
\boxed{117{,}787{,}392}
$$

trainable parameters.

