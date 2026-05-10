# Chapter 3.8 — Our Model in Numbers

## Complete Specification

| Property | Value |
|---|---|
| **Architecture** | Decoder-only Transformer |
| **Total parameters** | 117,787,392 (~118M) |
| **Layers (blocks)** | 12 |
| **Embedding dimension** | 768 |
| **Query heads** | 12 |
| **Key-Value heads** | 4 (Grouped-Query Attention) |
| **Head dimension** | 64 |
| **Feed-forward hidden dim** | 2,688 (3.5× expansion) |
| **Context window** | 384 tokens |
| **Vocabulary size** | 32,000 |
| **Normalization** | RMSNorm (pre-normalization) |
| **Position encoding** | RoPE (Rotary Positional Embeddings) |
| **Activation function** | SwiGLU |
| **Attention** | Flash Attention via PyTorch SDPA |
| **Weight tying** | Yes (embedding = LM head) |
| **Dropout** | 0.1 |

## Parameter Breakdown

| Component | Parameters | % of Total |
|---|---|---|
| Token embedding (shared with LM head) | 24,576,000 | 20.9% |
| Attention projections (Q, K, V, O) × 12 | 18,874,368 | 16.0% |
| SwiGLU feed-forward (W1, W2, Wout) × 12 | 74,317,824 | 63.1% |
| RMSNorm layers | 19,200 | 0.02% |
| **Total** | **117,787,392** | **100%** |

The feed-forward layers dominate — they contain 63% of all parameters.

## Comparison with Famous Models

| Model | Parameters | Layers | Emb Dim | Heads | Context |
|---|---|---|---|---|---|
| **Our Mini GPT** | **118M** | **12** | **768** | **12** | **384** |
| GPT-2 Small | 124M | 12 | 768 | 12 | 1024 |
| GPT-2 Medium | 355M | 24 | 1024 | 16 | 1024 |
| GPT-2 Large | 774M | 36 | 1280 | 20 | 1024 |
| LLaMA 7B | 6.7B | 32 | 4096 | 32 | 2048 |
| GPT-3 | 175B | 96 | 12288 | 96 | 2048 |

Our model is similar in size to GPT-2 Small. The main difference is the smaller context window (384 vs 1024) and the modern architecture choices (GQA, RoPE, SwiGLU instead of MHA, learned positions, GELU).

## Memory Usage

On our NVIDIA RTX 4060 Laptop GPU (8 GB VRAM):

| Component | Memory |
|---|---|
| Model weights (bfloat16) | ~236 MB |
| Optimizer states (AdamW) | ~472 MB |
| Activations (batch=20, seq=384) | ~1.5 GB |
| Gradients | ~236 MB |
| **Total during training** | **~2.5 GB** |

This fits comfortably in 8 GB VRAM with room for PyTorch overhead.

## Tokens Per Second

| Setting | Tokens/sec |
|---|---|
| Training (batch=20, seq=384) | ~7,600 tok/step |
| Generation (single sample) | ~50-100 tok/sec |

## Context Window: What 384 Tokens Means

384 tokens is roughly 250-300 words, or about half a page of text. This means:

- The model can "see" about half a page of context when predicting the next word
- Longer documents get split into overlapping 384-token windows
- The model cannot understand relationships between words that are more than 384 tokens apart

For comparison:
- GPT-2: 1,024 tokens (~2 pages)
- GPT-4: 128,000 tokens (~250 pages)
- Claude: 200,000 tokens (~400 pages)

Our context window is a known limitation, but it is sufficient for learning grammar, style, and short-range coherence. Long-range document understanding requires a larger context window.
