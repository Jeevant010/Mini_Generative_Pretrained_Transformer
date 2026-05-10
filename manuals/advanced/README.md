# Advanced — Future Work (Post-Training Branch)

> **Note:** The files in this folder are for the **next branch** (post-training). They document Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), chat templates, evaluation harness setup, and data quality analysis. These guides will be implemented after the pre-training phase is pushed and the `post-training` branch is created.

## Contents

| File | Topic |
|---|---|
| `evaluation_harness_guide.md` | Setting up EleutherAI lm-evaluation-harness for standardized benchmarks (HellaSwag, ARC, LAMBADA, etc.) |
| `generation_quality_metrics.md` | Deep-dive on Distinct-N, Self-BLEU, Entropy metrics with full math and code |
| `01_overview_base_to_chat.md` | The 4-stage pipeline: Pre-training → SFT → DPO → Chat |
| `02_supervised_fine_tuning.md` | Complete SFT guide with Dolly 15K dataset, code templates, loss masking |
| `03_dpo_preference_alignment.md` | DPO loss derivation, preference data format, full implementation |
| `04_chat_templates_and_deployment.md` | Chat tokens, tokenizer extension, CLI and Gradio interfaces |
| `openwebtext_analysis.md` | Data composition, non-English contamination, Chinchilla scaling analysis |

## When to Use These

1. Push the current branch with the pre-training book (chapters 1-10)
2. Create a new branch: `git checkout -b post-training`
3. Follow these guides in order: SFT → DPO → Chat Templates
