# Mini GPT Research Paper Content

This directory contains paper-ready Markdown content for the Mini Generative Pretrained Transformer project. The files are written so they can be copied into an IEEE-style paper, thesis chapter, project report, or presentation.

The PDF at `C:\Users\Deepesh\Downloads\Mini_GPT_Docs.pdf` currently appears to be an IEEE template draft with placeholder sections. These files replace that placeholder text with project-specific technical content based on the repository implementation and the available training artifacts.

## Files

- `01_abstract_introduction.md`: Title, abstract, keywords, motivation, contributions, and introduction.
- `02_system_overview.md`: End-to-end system pipeline, code modules, data flow, and design decisions.
- `03_model_architecture_math.md`: Full decoder-only Transformer architecture with equations for embeddings, RMSNorm, RoPE, GQA, SwiGLU, residual blocks, and parameter count.
- `04_training_objective_and_optimization.md`: Autoregressive objective, cross-entropy, perplexity, AdamW, warmup cosine learning-rate schedule, gradient clipping, and throughput metrics.
- `05_data_tokenization_and_preprocessing.md`: Parquet ingestion, quality filtering, BPE tokenizer training, binary token storage, train/validation split, and memory-mapped batch sampling.
- `06_experiments_results_and_discussion.md`: Current experiment configuration, 10 GB run evidence, 60k-step results, generation analysis, limitations, ablations, and future work.

## Current Project Snapshot

The content uses this snapshot from the local repository:

| Item | Value |
| --- | --- |
| Model type | Decoder-only GPT language model |
| Parameters | 117,787,392 trainable parameters |
| Layers | 12 Transformer blocks |
| Embedding width | 768 |
| Query heads | 12 |
| KV heads | 4 |
| Context length | 384 tokens |
| Vocabulary | 32,000 byte-level BPE tokens |
| Dataset target | 10 GB tokenized subset |
| Training tokens | 5,100,766,548 |
| Validation tokens | 267,942,572 |
| Latest observed checkpoint | `ckpt_step_60000.pt` |
| Latest observed validation loss | 3.517095 |
| Latest observed perplexity | 33.69 |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU |

## How To Use

Use `01_abstract_introduction.md` as the starting paper text. Then merge the methods sections from `02` through `05`, and use `06` for the experiments, analysis, and conclusion.

For LaTeX, equations inside `$$ ... $$` can be copied directly into equation environments.

