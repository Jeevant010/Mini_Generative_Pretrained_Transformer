# Model Architecture

## High-Level Specification

The model class is `GPTLanguageModel` in `model.py`.

| Component | Current configuration |
| --- | --- |
| Architecture | Decoder-only Transformer |
| Layers | 12 |
| Embedding dimension | 768 |
| Query heads | 12 |
| KV heads | 4 |
| Head dimension | 64 |
| Context length | 384 |
| Vocabulary | 32,000 |
| Normalization | RMSNorm |
| Positional encoding | RoPE |
| Attention | GQA with optional Flash Attention |
| FFN | SwiGLU |
| Dropout | 0.1 |
| Weight tying | Yes |

## Module Order

The forward pass is:

```text
token IDs
  -> token embedding
  -> Transformer block x 12
       -> RMSNorm
       -> GQA causal attention with RoPE
       -> residual add
       -> RMSNorm
       -> SwiGLU FFN
       -> residual add
  -> final RMSNorm
  -> tied LM head
  -> logits
```

## Tensor Shapes

For batch size `B = 20`, sequence length `T = 384`, and embedding dimension `d = 768`:

| Tensor | Shape |
| --- | --- |
| Token IDs | `[20, 384]` |
| Embeddings | `[20, 384, 768]` |
| Query heads | `[20, 12, 384, 64]` |
| Key heads before repeat | `[20, 4, 384, 64]` |
| Value heads before repeat | `[20, 4, 384, 64]` |
| Attention output | `[20, 384, 768]` |
| Logits | `[20, 384, 32000]` |

## Per-Block Parameter Count

| Component | Formula | Parameters |
| --- | --- | ---: |
| `q_proj` | `768 x 768` | 589,824 |
| `k_proj` | `768 x 256` | 196,608 |
| `v_proj` | `768 x 256` | 196,608 |
| `o_proj` | `768 x 768` | 589,824 |
| Attention total | sum above | 1,572,864 |
| `ffn.w1` | `768 x 2688` | 2,064,384 |
| `ffn.w2` | `768 x 2688` | 2,064,384 |
| `ffn.w_out` | `2688 x 768` | 2,064,384 |
| FFN total | sum above | 6,193,152 |
| `norm1.scale` | `768` | 768 |
| `norm2.scale` | `768` | 768 |
| Per-block total | all above | 7,767,552 |

For 12 blocks:

$$
12 \times 7{,}767{,}552 = 93{,}210{,}624
$$

## Embedding And LM Head

The token embedding has:

$$
32000 \times 768 = 24{,}576{,}000
$$

parameters.

The LM head is tied to the token embedding:

```python
self.token_embed.weight = self.lm_head.weight
```

so no second independent output matrix is added.

## Total Parameters

The full model has:

$$
93{,}210{,}624 + 24{,}576{,}000 + 768 = 117{,}787{,}392
$$

trainable parameters.

## Ablation Paths

The model supports four architecture toggles:

| Toggle | Enabled behavior | Disabled behavior |
| --- | --- | --- |
| `USE_RMSNORM` | RMSNorm before attention/FFN and final norm | Identity/no normalization |
| `USE_ROPE` | RoPE applied to Q and K | No positional rotation |
| `USE_FLASH_ATTENTION` | PyTorch SDPA path | Manual attention path |
| `USE_GQA` | 4 KV heads | Full MHA with 12 KV heads |

These toggles allow controlled research experiments without rewriting the model.

