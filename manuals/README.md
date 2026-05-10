# Mini Generative Pretrained Transformer — The Book

A beginner-friendly guide to building, training, and understanding a GPT-style language model from scratch.

---

## What This Is

This is a complete educational book about building a 118-million-parameter language model on a single laptop GPU. It covers everything from "What is a language model?" to real training outputs, quality metrics, ablation studies, and lessons learned from mistakes.

**No prior machine learning experience required.** Every concept is explained in plain English first, with optional math sections for those who want the details.

---

## Table of Contents

### Chapter 1: What Is This?
| Section | Topic |
|---|---|
| [01 — What Is a Language Model?](ch01_what_is_this/01_what_is_a_language_model.md) | The simplest explanation, with real model outputs |
| [02 — What Is GPT?](ch01_what_is_this/02_what_is_gpt.md) | History of GPT, our model vs GPT-2 and GPT-4 |
| [03 — Why Build Your Own?](ch01_what_is_this/03_why_build_your_own.md) | What you learn by building vs using APIs |
| [04 — What You Will Learn](ch01_what_is_this/04_what_you_will_learn.md) | Roadmap of all chapters |

### Chapter 2: How Text Becomes Numbers
| Section | Topic |
|---|---|
| [01 — Characters to Tokens](ch02_how_text_becomes_numbers/01_characters_to_tokens.md) | Why we need tokenization |
| [02 — BPE Algorithm Explained](ch02_how_text_becomes_numbers/02_bpe_algorithm_explained.md) | Byte Pair Encoding step by step |
| [03 — Our Tokenizer](ch02_how_text_becomes_numbers/03_our_tokenizer.md) | Implementation, encoding/decoding, storage |
| [04 — Special Tokens](ch02_how_text_becomes_numbers/04_special_tokens.md) | pad, bos, eos, unk — what each does |

### Chapter 3: The Transformer
| Section | Topic |
|---|---|
| [01 — What Is a Transformer?](ch03_the_transformer/01_what_is_a_transformer.md) | Big picture overview |
| [02 — Embeddings](ch03_the_transformer/02_embeddings.md) | How tokens become vectors |
| [03 — Attention for Beginners](ch03_the_transformer/03_attention_for_beginners.md) | Q/K/V, multi-head, GQA, causal mask |
| [04 — RoPE: Positions](ch03_the_transformer/04_rope_positions.md) | How the model knows word order |
| [05 — SwiGLU Feed-Forward](ch03_the_transformer/05_feed_forward_swiglu.md) | The gated feed-forward layer |
| [06 — RMSNorm](ch03_the_transformer/06_rmsnorm.md) | Keeping numbers stable |
| [07 — Putting It Together](ch03_the_transformer/07_putting_it_together.md) | Full block, residual connections, 12-layer stack |
| [08 — Our Model in Numbers](ch03_the_transformer/08_our_model_numbers.md) | Complete spec, parameter breakdown, memory usage |

### Chapter 4: The Data Pipeline
| Section | Topic |
|---|---|
| [01 — Where Does Training Data Come From?](ch04_data_pipeline/01_where_does_training_data_come_from.md) | OpenWebText, Reddit sourcing, content types |
| [02 — Filtering for Quality](ch04_data_pipeline/02_filtering_for_quality.md) | All quality filters explained |
| [03 — Tokenizing Billions of Words](ch04_data_pipeline/03_tokenizing_billions.md) | Batch tokenization, memory mapping, random sampling |

### Chapter 5: Training
| Section | Topic |
|---|---|
| [01 — What Happens During Training](ch05_training/01_what_happens_during_training.md) | Forward/backward pass, optimizer, gradient clipping |
| [02 — Loss, Learning Rate, and Checkpoints](ch05_training/02_loss_and_learning_rate.md) | Loss/PPL explained, cosine schedule, saving progress |

### Chapter 6: Watching Training Happen ⭐
| Section | Topic |
|---|---|
| [01 — From Gibberish to English](ch06_watching_training/01_from_gibberish_to_english.md) | **Real outputs at every stage** — step 0 to 122K |
| [02 — Loss and Perplexity Timeline](ch06_watching_training/02_loss_and_perplexity_timeline.md) | Full metrics timeline with analysis |

### Chapter 7: Measuring Quality
| Section | Topic |
|---|---|
| [01 — Perplexity Explained](ch07_measuring_quality/01_perplexity_explained.md) | What PPL means, reference values |
| [02 — Diversity and Repetition Metrics](ch07_measuring_quality/02_diversity_metrics.md) | Distinct-N, Self-BLEU, Entropy, repetition detection |

### Chapter 8: Ablation Studies
| Section | Topic |
|---|---|
| [01 — What Is an Ablation Study?](ch08_ablation_studies/01_what_is_ablation.md) | Methodology, toggle system, how to read results |
| [02 — Component Ablations](ch08_ablation_studies/02_component_ablations.md) | RMSNorm, RoPE, Flash Attention, GQA details |

### Chapter 9: Mistakes and Lessons ⭐
| Section | Topic |
|---|---|
| [01 — The Non-English Gibberish Problem](ch09_mistakes_and_lessons/01_non_english_gibberish.md) | The "ibn nimy" bug — causes and fixes |
| [02 — Repetition Loops](ch09_mistakes_and_lessons/02_repetition_loops.md) | Why models repeat, how to detect and fix it |
| [03 — Key Takeaways](ch09_mistakes_and_lessons/03_key_takeaways.md) | 8 lessons learned + what we would do differently |

### Chapter 10: What Comes Next
| Section | Topic |
|---|---|
| [01 — From Prediction to Conversation](ch10_whats_next/01_from_prediction_to_conversation.md) | SFT → DPO → Chat roadmap |
| [02 — Branch Strategy](ch10_whats_next/02_branch_strategy.md) | Git plan, push checklist, what goes where |

### Appendices
| Section | Topic |
|---|---|
| [Glossary](appendices/glossary.md) | Every technical term explained in plain English |
| [Codebase Map](appendices/codebase_map.md) | Every file and its purpose |
| [Config Reference](appendices/config_reference.md) | All settings with descriptions |
| [References](appendices/references.md) | Academic papers cited |
| [Quick Start](appendices/quick_start.md) | Setup and run in 4 steps |

### Advanced (Future Work)
| Section | Topic |
|---|---|
| [README](advanced/README.md) | Overview of post-training guides |
| SFT, DPO, Chat Templates, Evaluation Harness, Data Analysis | Detailed guides for the next branch |

---

## Reading Guide

**Complete beginner?** → Read Chapters 1–6 in order. Skip math on first read.

**Know Python, not ML?** → Read everything in order. The math will make sense.

**Know ML already?** → Jump to Chapters 6, 7, 9 for results, metrics, and lessons.

**Here for the code?** → See [Codebase Map](appendices/codebase_map.md) then dive in.

---

## The Model

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 117,787,392 (~118M) |
| Components | GQA, RoPE, RMSNorm, SwiGLU, Flash Attention |
| Training data | 10 GB OpenWebText (5.1B tokens) |
| Hardware | NVIDIA RTX 4060 Laptop GPU (8 GB VRAM) |
| Best perplexity | 33.69 at step 60,000 |
