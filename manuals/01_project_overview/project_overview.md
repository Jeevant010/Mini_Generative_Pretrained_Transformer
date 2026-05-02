# Project Overview

## Purpose

Mini Generative Pretrained Transformer is a from-scratch small language model project built in PyTorch. The goal is to study and implement the full pipeline behind GPT-style language modeling on consumer hardware:

- large text preprocessing
- byte-level BPE tokenization
- memory-mapped dataset loading
- decoder-only Transformer training
- checkpointing and resume
- perplexity evaluation
- sample generation
- ablation studies

The project is designed as a research and engineering artifact. It is small enough to run locally, but it uses many architecture choices found in modern language models.

## Current Model

The active model is a decoder-only Transformer with:

| Feature | Current value |
| --- | --- |
| Parameters | 117,787,392 |
| Layers | 12 |
| Embedding width | 768 |
| Query heads | 12 |
| KV heads | 4 |
| Head dimension | 64 |
| Context length | 384 tokens |
| Vocabulary | 32,000 byte-level BPE tokens |
| Normalization | RMSNorm |
| Position encoding | RoPE |
| Attention | Grouped-Query Attention |
| FFN | SwiGLU |
| LM head | Tied with token embedding |

## Research Question

The main research question is:

Can a modern GPT-style model be trained from scratch on a multi-gigabyte web-text subset using consumer hardware, while retaining rigorous model design, measurement, and reproducibility?

The practical sub-questions are:

- How can the model be sized to fit an RTX 4060 Laptop GPU?
- How can 10 GB of tokenized text be sampled without loading it fully into memory?
- How much improvement is visible after partial training?
- Which architecture components are load-bearing and which are efficiency optimizations?
- Why does a base model generate continuation text rather than assistant-style answers?

## Current Data Scale

The current prepared dataset uses compact `uint16` token storage.

| File | Size | Token count |
| --- | ---: | ---: |
| `train.bin` | 9.50 GB | 5,100,766,548 |
| `val.bin` | 511.06 MB | 267,942,572 |
| `bpe_tokenizer_32k.json` | 2.16 MB | 32,000 vocabulary entries |

The total target size is exactly 10 GiB:

$$
10 \times 1024^3 = 10{,}737{,}418{,}240 \text{ bytes}
$$

Since tokens are stored as `uint16`, each token uses 2 bytes.

## Current Training Status

The latest observed experiment reached step 60,000. At `batch_size = 20` and `block_size = 384`, each step sees:

$$
20 \times 384 = 7680
$$

token positions.

At 60,000 steps:

$$
60{,}000 \times 7680 = 460{,}800{,}000
$$

token positions have been used for optimization. Relative to the training file:

$$
\frac{460{,}800{,}000}{5{,}100{,}766{,}548} \approx 0.0903
$$

So the current 60k run is roughly 9 percent of one token-equivalent pass over the training data.

## Observed Result

At step 0:

| Metric | Value |
| --- | ---: |
| Validation loss | 10.539526 |
| Perplexity | 37,779.67 |

At step 60,000:

| Metric | Value |
| --- | ---: |
| Validation loss | 3.517095 |
| Perplexity | 33.69 |

This is a strong indication that training is working. The generated samples are locally fluent, but they remain base-model continuations rather than instruction-following responses.

## What The Project Is Not

This project is not yet:

- an instruction-tuned chatbot
- a RLHF or preference-aligned assistant
- a retrieval-augmented model
- a factual QA system
- a production serving system

It is a base pretrained language model. Its objective is next-token prediction:

$$
\max_\theta \sum_t \log P_\theta(x_t \mid x_{<t})
$$

Instruction behavior requires an additional fine-tuning stage on conversational data.

