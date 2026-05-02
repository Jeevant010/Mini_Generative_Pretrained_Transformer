# Attention Mechanisms

## Why Attention Is Needed

Self-attention lets each token build a context-aware representation by looking back at earlier tokens. For language modeling, token `t` may use positions `1...t`, but not future positions.

The causal constraint is what turns a Transformer block into a decoder-only language model.

## Query, Key, Value

For an input activation matrix:

$$
X \in \mathbb{R}^{B \times T \times d}
$$

the model computes:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

For this project:

$$
d = 768,\quad H_q = 12,\quad H_{kv}=4,\quad d_h=64
$$

Queries have shape:

$$
Q \in \mathbb{R}^{B \times H_q \times T \times d_h}
$$

Keys and values have shape:

$$
K,V \in \mathbb{R}^{B \times H_{kv} \times T \times d_h}
$$

## Scaled Dot-Product Attention

The raw attention score between query position `i` and key position `j` is:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}}
$$

The scale factor prevents score variance from growing with head dimension.

After applying the causal mask:

$$
M_{ij} =
\begin{cases}
0, & j \leq i \\
-\infty, & j > i
\end{cases}
$$

the normalized attention weight is:

$$
a_{ij} = \frac{\exp(s_{ij}+M_{ij})}{\sum_{r=1}^{T}\exp(s_{ir}+M_{ir})}
$$

The output is:

$$
o_i = \sum_{j=1}^{T} a_{ij}v_j
$$

## Rotary Positional Embeddings

Self-attention alone has no built-in token order. RoPE injects position into the query and key vectors.

For each pair of dimensions, RoPE applies a rotation:

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

where `m` is the token position and:

$$
\theta_i = 10000^{-2i/d_h}
$$

The implementation computes:

$$
\operatorname{RoPE}(x) = x \odot \cos(\Theta) + \operatorname{rotate\_half}(x)\odot \sin(\Theta)
$$

RoPE is applied to `Q` and `K`, not `V`.

## Grouped-Query Attention

Full Multi-Head Attention uses:

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
g = \frac{H_q}{H_{kv}} = 3
$$

query heads.

In code, keys and values are repeated across query groups:

```python
k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
```

## Parameter Saving From GQA

With full MHA, key and value projections each output `d` features:

$$
2d^2 = 2(768)(768) = 1{,}179{,}648
$$

With GQA, key and value projections output:

$$
H_{kv}d_h = 4 \times 64 = 256
$$

features each:

$$
2d(H_{kv}d_h) = 2(768)(256) = 393{,}216
$$

GQA saves:

$$
1{,}179{,}648 - 393{,}216 = 786{,}432
$$

key-value projection parameters per layer compared with full MHA.

## Flash Attention Path

When `USE_FLASH_ATTENTION = True`, attention is executed by:

```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

PyTorch can dispatch this to optimized kernels. The mathematical result is equivalent to standard causal attention, but the memory behavior is better because the full attention matrix does not need to be materialized in the same way as the manual path.

## Manual Attention Path

When `USE_FLASH_ATTENTION = False`, the project uses `manual_causal_attention()`:

1. Compute `Q @ K.T`
2. Apply causal mask
3. Apply softmax
4. Multiply by `V`

This is useful for ablation because it isolates Flash Attention as a hardware efficiency feature rather than a modeling feature.

