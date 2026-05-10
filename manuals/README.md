# Mini GPT Project Manuals

This directory is the current technical manual set for the Mini Generative Pretrained Transformer project. It reflects the active codebase as of the current `subset_10gb` configuration:

- Decoder-only GPT language model
- 117,787,392 trainable parameters
- 12 Transformer blocks
- 768-dimensional embeddings
- 12 query heads and 4 key-value heads
- Grouped-Query Attention (GQA)
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- SwiGLU feed-forward blocks
- Flash Attention through PyTorch scaled dot-product attention
- Tied token embedding and language-model head weights
- Byte-level BPE tokenizer with 32,000 vocabulary entries
- 10 GB tokenized dataset target
- Memory-mapped `train.bin` and `val.bin`

## Current Experimental Snapshot

| Item | Value |
| --- | --- |
| Active preset | `subset_10gb` |
| Train binary | `train.bin`, 5,100,766,548 tokens |
| Validation binary | `val.bin`, 267,942,572 tokens |
| Latest observed checkpoint | `checkpoints/ckpt_step_60000.pt` |
| Latest observed validation loss | 3.517095 |
| Latest observed perplexity | 33.69 |
| Hardware observed | NVIDIA GeForce RTX 4060 Laptop GPU |

## Directory Structure

```text
manuals/
|-- 01_project_overview/
|   `-- project_overview.md
|-- 02_theoretical_foundations/
|   |-- transformer_architecture.md
|   |-- attention_mechanisms.md
|   `-- tokenization_theory.md
|-- 03_system_architecture/
|   |-- codebase_structure.md
|   |-- model_architecture.md
|   `-- data_pipeline.md
|-- 04_implementation_details/
|   |-- config_reference.md
|   |-- training_pipeline.md
|   |-- inference_pipeline.md
|   |-- tokenizer_implementation.md
|   |-- evaluation_metrics.md
|   `-- hyperparameter_presets.md
|-- 05_experimental_methodology/
|   |-- research_notebooks.md
|   |-- hardware_profiling.md
|   |-- training_progression.md
|   |-- ablation_studies.md
|   `-- verification_guide.md
|-- 06_reproducibility/
|   |-- environment_setup.md
|   |-- quick_start_guide.md
|   `-- checkpoint_management.md
|-- 07_appendices/
|   |-- glossary.md
|   |-- references.md
|   `-- license_and_attribution.md
|-- 08_evaluation_harness/          ← NEW
|   |-- evaluation_harness_guide.md
|   `-- generation_quality_metrics.md
|-- 09_post_training/               ← NEW
|   |-- 01_overview_base_to_chat.md
|   |-- 02_supervised_fine_tuning.md
|   |-- 03_dpo_preference_alignment.md
|   `-- 04_chat_templates_and_deployment.md
`-- 10_data_quality/                ← NEW
    `-- openwebtext_analysis.md
```

## Reading Order

### For Understanding the Current Model (Pre-Training)

1. `01_project_overview/project_overview.md`
2. `02_theoretical_foundations/transformer_architecture.md`
3. `02_theoretical_foundations/attention_mechanisms.md`
4. `03_system_architecture/model_architecture.md`
5. `03_system_architecture/data_pipeline.md`
6. `04_implementation_details/training_pipeline.md`
7. `05_experimental_methodology/training_progression.md`
8. `05_experimental_methodology/ablation_studies.md`

### For Evaluation and Metrics

9. `04_implementation_details/evaluation_metrics.md` — basic metrics (loss, PPL, throughput)
10. `08_evaluation_harness/generation_quality_metrics.md` — advanced metrics (Distinct-N, Self-BLEU, Entropy)
11. `08_evaluation_harness/evaluation_harness_guide.md` — standardized benchmarks (HellaSwag, ARC, etc.)

### For Post-Training (Building a Conversational Model)

12. `09_post_training/01_overview_base_to_chat.md` — the 4-stage pipeline overview
13. `09_post_training/02_supervised_fine_tuning.md` — SFT implementation guide
14. `09_post_training/03_dpo_preference_alignment.md` — DPO implementation guide
15. `09_post_training/04_chat_templates_and_deployment.md` — deployment and chat interface

### For Data Quality

16. `10_data_quality/openwebtext_analysis.md` — data composition, non-English filtering, scaling laws

## Important Interpretation

The trained model is a base language model. It learns:

$$
P(x_t \mid x_{<t})
$$

It does not automatically learn to behave like a chat assistant unless the training or fine-tuning data contains instruction-response examples. Therefore, generated text should be evaluated as text continuation, not as aligned assistant behavior.

To make the model conversational, follow the post-training pipeline documented in `09_post_training/`.
