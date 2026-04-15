# Attention Notebook Walkthrough

Notebook analyzed: Research/Attention.ipynb

This document explains exactly what was implemented in the attention notebook, why each part exists, and how it connects to your tokenizer and embedding stages.

## 1) What was built

The notebook implements a full LLM-ready attention stack with:

- Type 1: Bidirectional Multi-Head Self-Attention (MHA)
- Type 2: Masked Causal Self-Attention
- Type 3: Multi-Query Attention (MQA)
- Type 4: Grouped-Query Attention (GQA)
- Bonus: Cross-Attention module

It also includes:

- RMSNorm pre-norm transformer blocks
- SwiGLU feed-forward blocks
- RoPE (rotary positional encoding)
- embedding warm-start from embedding_sgns_wizard.pt
- autoregressive language-model training loop
- text generation and model export

## 2) Cell-by-cell breakdown

### Cell 1: Notebook intro

Defines scope and lists all attention variants included.

### Cell 2: Imports and runtime setup

- imports PyTorch and utilities
- sets random seeds
- detects CPU/GPU and prints hardware

### Cell 3: BPE runtime tokenizer

- loads tokenizer merges from JSON
- encodes text to token IDs
- decodes generated token IDs back to text

### Cell 4: Data and artifact loading

- loads bpe_tokenizer_wizard.json
- loads wizard_of_oz.txt
- loads embedding_sgns_wizard.pt for warm-start token embedding
- splits token stream into train/validation buffers

### Cell 5: Model and training profiles

Defines profile presets for:

- cpu_safe
- cpu_quality
- rtx_4060_balanced
- rtx_4060_quality
- rtx_4060_max

Auto-selects profile based on detected device/VRAM.

### Cell 6: Helpers

- adapts pretrained embedding dimensions to model d_model
- random contiguous batch sampler for language modeling
- cosine LR scheduler with warmup

### Cell 7: Core attention modules

Implements:

- RMSNorm
- SwiGLU
- RotaryEmbedding + apply_rope
- FlexibleAttention (core attention engine)
- MultiHeadSelfAttention
- CausalSelfAttention (masked)
- MultiQueryAttention
- GroupedQueryAttention
- CrossAttention

### Cell 8: Transformer + LM

Implements:

- TransformerBlock with pre-norm residual structure
- PowerfulAttentionLM
- causal next-token cross-entropy objective
- autoregressive generation function
- evaluation helper

### Cell 9: Attention type smoke tests

Runs all attention variants and prints output shapes to verify wiring.

### Cell 10: Training loop

- initializes model and optimizer
- runs eval checkpoints during training
- uses gradient clipping
- supports AMP when CUDA is available

### Cell 11: Generation + export

- generates sample text from prompt
- decodes with BPE tokenizer
- saves attention_model_wizard.pt

### Cell 12: Scaling notes

Gives practical reminders for moving from CPU to larger GPU runs.

## 3) Attention types included and how they differ

## Type 1: Bidirectional Multi-Head Self-Attention

- each token can attend to all tokens
- common in encoder-style processing
- not strictly autoregressive by itself

## Type 2: Masked Causal Self-Attention

- each token attends only to past tokens
- required for GPT-style next-token prediction
- this is the default path used in language-model training

## Type 3: Multi-Query Attention (MQA)

- many query heads, single shared K/V head
- lower KV memory and faster inference
- useful for memory-constrained generation workloads

## Type 4: Grouped-Query Attention (GQA)

- many query heads, few grouped K/V heads
- middle ground between MHA quality and MQA efficiency
- selected as high-quality default for RTX 4060 profiles

## Bonus: Cross-Attention

- query comes from one sequence, key/value from another
- useful for encoder-decoder and conditioning tasks

## 4) Architecture quality decisions

This notebook uses modern quality choices:

1. Pre-norm residual design for stable deep training.
2. RMSNorm for efficient normalization.
3. SwiGLU for stronger FFN expressiveness.
4. RoPE for better relative position behavior.
5. Optional GQA/MQA for efficient scaling.
6. Weight tying to reduce parameters and improve LM efficiency.

## 5) Observed training behavior on current run

Observed outputs in this run:

- profile selected: cpu_safe
- trainable parameters: about 2.52M
- training loss dropped from about 8.13 to about 4.13
- validation loss dropped from about 8.13 to about 4.26
- attention_model_wizard.pt saved successfully

This confirms the end-to-end pipeline is functioning.

## 6) Artifacts produced

- attention notebook: Research/Attention.ipynb
- trained model artifact: Research/attention_model_wizard.pt
- tokenizer artifact used: Research/bpe_tokenizer_wizard.json
- embedding warm-start used: Research/embedding_sgns_wizard.pt

## 7) How to use this in the next architecture notebook

1. Load attention_model_wizard.pt and restore model config.
2. Rebuild PowerfulAttentionLM with same config.
3. Load state_dict.
4. Continue training on larger corpus and longer context.
5. Move from cpu_safe to RTX 4060 quality profile.

## 8) Suggested next upgrades

1. Train on your full dataset (not only Wizard of Oz).
2. Increase context length gradually (128 -> 256 -> 384+).
3. Keep GQA for stronger quality-efficiency tradeoff.
4. Add checkpoint save intervals and best-val checkpointing.
5. Add sampling controls (temperature schedule, repetition penalty).
