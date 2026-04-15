# Full Architecture Notebook Guide

Notebook: Research/Full_Architecture.ipynb

This guide explains how to run the full end-to-end training system in one notebook, including tokenizer, embeddings, attention training, generation evaluation, and checkpoint resume.

## 1) What this notebook includes

The notebook unifies all major LLM pipeline stages:

1. Stage 1: Byte-level BPE tokenizer (train or load)
2. Stage 2: SGNS embeddings (train or load)
3. Stage 3: Transformer attention language model training
4. Stage 4: Generation evaluation and final model export

It also includes all requested attention variants:

- Type 1: Multi-Head Self-Attention (MHA)
- Type 2: Masked Causal Self-Attention
- Type 3: Multi-Query Attention (MQA)
- Type 4: Grouped-Query Attention (GQA)
- Bonus: Cross-Attention module

## 2) Main controls you will edit first

In the notebook data/setup section:

- FORCE_RETRAIN_TOKENIZER
- FORCE_RETRAIN_EMBEDDINGS

In the attention profile section:

- ATTENTION_STAGE_PLAN
- STAGE_STEP_SCALE
- CHECKPOINT_FILE
- CHECKPOINT_EVERY
- RESUME_FROM_CHECKPOINT

## 3) Small-to-large flexible training

The notebook supports curriculum-style progressive training.

Example on CPU:

- ATTENTION_STAGE_PLAN = ["cpu_safe", "cpu_quality"]
- STAGE_STEP_SCALE = [0.5, 1.0]

Example on RTX 4060:

- ATTENTION_STAGE_PLAN = ["rtx_4060_balanced", "rtx_4060_quality"]
- STAGE_STEP_SCALE = [0.5, 1.0]

How it works:

1. Stage 1 trains a smaller/faster config.
2. Stage 2 trains a stronger config.
3. Token embedding warm-start is carried forward through stages.

## 4) Checkpoint resume support

Checkpoint saves include:

- current stage index and stage name
- step inside current stage
- global step counter
- model, optimizer, and scaler states
- training history and config snapshot

Behavior:

- If RESUME_FROM_CHECKPOINT is True and CHECKPOINT_FILE exists, training resumes from the saved point.
- Checkpoints are saved every CHECKPOINT_EVERY steps and at the end of each stage.

## 5) Typical run modes

## Fast iteration mode

- FORCE_RETRAIN_TOKENIZER = False
- FORCE_RETRAIN_EMBEDDINGS = False
- ATTENTION_STAGE_PLAN = ["cpu_safe"] or ["rtx_4060_balanced"]

## Full pipeline retrain mode

- FORCE_RETRAIN_TOKENIZER = True
- FORCE_RETRAIN_EMBEDDINGS = True
- ATTENTION_STAGE_PLAN = ["cpu_safe", "cpu_quality"] or ["rtx_4060_balanced", "rtx_4060_quality"]

## 6) Outputs generated

The notebook outputs:

- tokenizer artifact: Research/bpe_tokenizer_wizard.json
- embedding artifact: Research/embedding_sgns_wizard.pt
- final model artifact: Research/full_architecture_model_wizard.pt
- periodic checkpoint: Research/checkpoints_full_arch/full_arch_last.pt

## 7) GPU readiness and portability

The notebook auto-selects device with:

- CUDA if available
- otherwise CPU

Mixed precision is enabled automatically on CUDA paths.

To move between devices:

1. Keep artifacts and checkpoints in the same Research paths.
2. Use map_location-aware loading (already in notebook).
3. Keep the same tokenizer artifact for model compatibility.

## 8) Troubleshooting

## If training is too slow

1. Reduce ATTENTION_STAGE_PLAN to one stage.
2. Reduce train_cfg steps via STAGE_STEP_SCALE.
3. Lower max_seq_len and batch size.

## If CUDA OOM occurs

1. Lower batch size first.
2. Lower max_seq_len.
3. Use a smaller profile.

## If resume behaves unexpectedly

1. Delete CHECKPOINT_FILE to restart fresh.
2. Re-run notebook from top.

## 9) Recommended next step

After this notebook is stable with your larger dataset:

1. Add periodic best-validation checkpoint tracking.
2. Add evaluation prompts set and save outputs per checkpoint.
3. Add final inference-only notebook that loads full_architecture_model_wizard.pt.
