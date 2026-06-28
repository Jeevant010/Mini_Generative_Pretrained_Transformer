# Model Training Story

## The Goal
To build, train, and align a 118 Million parameter language model from scratch on a consumer laptop.

## Phase 1: Pre-training (Completed)
- **Architecture**: 118M parameters (12 layers, 12 heads, 768 embedding dimension).
- **Data**: Trained on 10GB of OpenWebText.
- **Duration**: Reached step 149,000 (roughly 5.5B tokens).
- **Status**: Absolute success. The model achieved a perplexity of 27.40 and demonstrates strong syntactical and structural understanding of the English language.

## Phase 2: Supervised Fine-Tuning (SFT) & Benchmarks (In Progress)
- Transitioning from an autocomplete text predictor to an instruction-following assistant.
- Using the Databricks Dolly 15K dataset.
- Establishing formal benchmark scores using EleutherAI's lm-evaluation-harness.
