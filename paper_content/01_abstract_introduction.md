# Abstract And Introduction

## Suggested Title

Attention on a Budget: Engineering a Custom GPT-Style Language Model on Consumer Hardware

## Abstract

This project presents the design, implementation, and experimental evaluation of a compact GPT-style language model trained from scratch on consumer-grade GPU hardware. The system implements a decoder-only Transformer architecture with modern efficiency-oriented components: byte-level Byte Pair Encoding (BPE), Rotary Positional Embeddings (RoPE), Root Mean Square Layer Normalization (RMSNorm), Grouped-Query Attention (GQA), SwiGLU feed-forward layers, Flash Attention through PyTorch scaled dot-product attention, and tied input-output token embeddings. The training pipeline is designed around large local web-text corpora stored as parquet shards and converted into memory-mapped binary token files, allowing multi-gigabyte datasets to be used without loading the full corpus into RAM.

The current implementation trains a 117.8M-parameter model with 12 Transformer blocks, 768-dimensional embeddings, 12 query heads, 4 key-value heads, a 384-token context window, and a 32,000-token vocabulary. A 10 GB tokenized subset containing approximately 5.10 billion training tokens and 267.9 million validation tokens was prepared. After 60,000 optimization steps, the model reached a validation loss of 3.5171 and a perplexity of 33.69. Generated samples show that the model has learned fluent local syntax and domain-like continuation behavior, but remains a raw pretrained next-token model rather than an instruction-following chatbot. The work demonstrates that careful architectural choices and streaming data engineering can make small language-model pretraining feasible on limited hardware, while also highlighting the remaining gap between base language modeling and conversational alignment.

## Keywords

Small language model, GPT, Transformer, decoder-only architecture, Grouped-Query Attention, RoPE, RMSNorm, SwiGLU, byte-level BPE, memory-mapped training, consumer GPU.

## Introduction

Large language models are usually associated with massive compute clusters, very large datasets, and industrial-scale training infrastructure. However, the core ideas behind these models can be studied and reproduced at a smaller scale using carefully engineered systems. This project explores that middle ground: building a custom GPT-style language model from scratch while respecting the memory, throughput, and storage constraints of a consumer laptop GPU.

The main goal is not to compete with frontier-scale models, but to understand and demonstrate the complete pipeline needed to train a language model: data preparation, tokenization, autoregressive modeling, Transformer architecture design, optimization, checkpointing, evaluation, and text generation. The project is therefore both an engineering artifact and a research-style investigation into how modern Transformer components behave under constrained training conditions.

The implementation follows a decoder-only Transformer design. Given a sequence of previous tokens, the model predicts the next token. This objective is simple, scalable, and is the same fundamental pretraining objective used by GPT-family models. The system improves upon a minimal Transformer by including several modern architectural choices:

- RMSNorm for stable normalization with fewer operations than LayerNorm.
- RoPE for relative position-aware attention without learned absolute position tables.
- Grouped-Query Attention to reduce key-value projection and cache cost compared with full Multi-Head Attention.
- SwiGLU feed-forward layers for stronger nonlinear transformation capacity.
- Flash Attention through PyTorch's `scaled_dot_product_attention` for efficient causal attention.
- Weight tying between token embeddings and the language modeling head to reduce parameters and improve parameter sharing.

The project also emphasizes data engineering. The dataset preparation script reads parquet shards directly, filters low-quality or non-English documents, trains a byte-level BPE tokenizer, appends end-of-sequence tokens, and writes token IDs into compact `uint16` binary files. During training, these files are accessed with NumPy memory maps, so random batches can be sampled from multi-gigabyte files without loading the full dataset into memory.

## Problem Statement

The central research question is:

Can a modern GPT-style language model be trained from scratch on a multi-gigabyte web-text subset using consumer hardware, while preserving enough architectural rigor and instrumentation to support meaningful analysis?

This question breaks into four sub-problems:

- How should the model architecture be designed to balance quality and memory efficiency?
- How can large text data be preprocessed and sampled efficiently on a local machine?
- How should training be monitored so progress can be measured beyond subjective text samples?
- What behavior should be expected from a base pretrained model after partial training?

## Contributions

This project makes the following contributions:

- Implements a complete GPT-style decoder-only model in PyTorch with RoPE, RMSNorm, GQA, SwiGLU, Flash Attention, and tied embeddings.
- Builds a streaming data preparation pipeline for parquet-based web text that avoids large intermediate Arrow caches.
- Uses byte-level BPE tokenization with a 32k vocabulary and compact `uint16` binary storage.
- Provides a memory-mapped batch loader for large local token files.
- Implements a production training loop with cosine learning-rate decay, warmup, gradient clipping, validation loss, perplexity, checkpoint resume, CSV metrics, VRAM tracking, throughput tracking, and sample generation.
- Provides ablation toggles for core architectural features, enabling systematic experiments.
- Reports a real 10 GB subset run with 60,000 observed steps, validation loss 3.5171, and perplexity 33.69.

## Scope

The model trained here is a base language model. It learns to continue text according to the distribution of the training corpus. It is not instruction-tuned, reinforcement-learning aligned, or trained specifically as a chatbot. Therefore, a prompt such as `how can i help` should be interpreted as a text prefix rather than a command requiring assistant-like behavior. Conversational behavior would require a second fine-tuning stage using instruction-response or dialogue data.

