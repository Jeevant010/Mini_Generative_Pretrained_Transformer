# Ablation Studies

## Purpose

An ablation study disables one component at a time to measure what that component contributes. In this project, ablations answer:

- Does RMSNorm stabilize training?
- Does RoPE provide necessary positional information?
- Does Flash Attention improve speed and memory without changing model math?
- Does GQA reduce memory/parameters compared with full MHA?

## Current Toggles

All toggles are in `config.py` and are consumed by `model.py`.

| Toggle | Default | Enabled behavior | Disabled behavior |
| --- | --- | --- | --- |
| `USE_RMSNORM` | True | RMSNorm in blocks and final norm | Identity/no normalization |
| `USE_ROPE` | True | Apply RoPE to Q and K | No positional encoding |
| `USE_FLASH_ATTENTION` | True | PyTorch causal SDPA | Manual attention implementation |
| `USE_GQA` | True | 12 query heads, 4 KV heads | Full MHA with 12 KV heads |

## Recommended Protocol

Use the same:

- dataset
- random seed
- batch size
- block size
- model size
- number of steps
- evaluation interval
- generation prompts

Change exactly one toggle at a time.

For quick tests, use:

```python
ACTIVE_PRESET = "wizard_of_oz_smoke"
```

For paper-quality ablations, choose a fixed small budget such as 2,000 or 5,000 steps on the same data slice.

## Metrics To Report

| Metric | Why it matters |
| --- | --- |
| Final train loss | Shows optimization behavior |
| Final validation loss | Shows generalization |
| Perplexity | Interpretable language-model quality |
| Tokens/sec | Throughput |
| Peak VRAM | Memory efficiency |
| Gradient norm | Stability |
| Sample text | Qualitative behavior |
| Failure status | NaN, OOM, stable, unstable |

## RMSNorm Ablation

Disable:

```python
USE_RMSNORM = False
```

The block becomes:

$$
u^{(l)} = h^{(l)} + \operatorname{Attn}(h^{(l)})
$$

$$
h^{(l+1)} = u^{(l)} + \operatorname{FFN}(u^{(l)})
$$

Without normalization, activation scale can grow through residual additions:

$$
\|h^{(L)}\| \approx \|h^{(0)}\| + \sum_{l=0}^{L-1}\|f_l(h^{(l)})\|
$$

Expected result:

- larger gradient norms
- less stable loss
- possible NaN if learning rate is too high

## RoPE Ablation

Disable:

```python
USE_ROPE = False
```

Without positional encoding, attention is permutation-equivariant. The model loses a direct way to distinguish order. It can still learn unigram and local statistical patterns through the sequence layout of training, but the architecture no longer has explicit position-aware attention scores.

Expected result:

- worse validation loss
- weaker grammar
- poor long-range ordering
- degraded continuation quality

## Flash Attention Ablation

Disable:

```python
USE_FLASH_ATTENTION = False
```

This switches to `manual_causal_attention()`. The math remains:

$$
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_h}} + M\right)V
$$

Expected result:

- similar loss and perplexity
- lower tokens/sec
- higher VRAM

This is an efficiency ablation, not a modeling-quality ablation.

## GQA Ablation

Disable:

```python
USE_GQA = False
```

The model changes from:

$$
H_q=12,\quad H_{kv}=4
$$

to:

$$
H_q=12,\quad H_{kv}=12
$$

Key-value projection parameters increase from:

$$
2(768)(4 \times 64)=393{,}216
$$

to:

$$
2(768)(12 \times 64)=1{,}179{,}648
$$

per layer.

Expected result:

- more parameters
- more memory
- possibly slightly better quality
- slower training

## Suggested Paper Table

| Variant | Val loss | PPL | Tok/s | VRAM MB | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Full model | fill in | fill in | fill in | fill in | stable |
| No RMSNorm | fill in | fill in | fill in | fill in | expected unstable |
| No RoPE | fill in | fill in | fill in | fill in | expected worse quality |
| No Flash Attention | fill in | fill in | fill in | fill in | expected slower |
| Full MHA | fill in | fill in | fill in | fill in | expected more memory |

Do not report expected values as measured values. Run the ablation script or controlled manual runs, then fill in the table.

