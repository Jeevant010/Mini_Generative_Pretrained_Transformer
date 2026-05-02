# Research Notebooks

The `Research/` directory contains exploratory notebooks and component guides. These are useful for explaining the model in a paper or presentation, but the production code lives in the root Python files.

## Current Research Files

| File | Purpose |
| --- | --- |
| `Tokenizer.ipynb` | Tokenizer experiments |
| `TOKENIZER_WALKTHROUGH.md` | Tokenizer explanation |
| `Embeddings.ipynb` | Embedding experiments |
| `EMBEDDINGS_NOTEBOOK_ANALYSIS.md` | Embedding notes |
| `Attention.ipynb` | Attention experiments |
| `ATTENTION_BEGINNER_GUIDE.md` | Intro attention explanation |
| `ATTENTION_NOTEBOOK_WALKTHROUGH.md` | Notebook walkthrough |
| `ATTENTION_4060_TUNING.md` | Hardware tuning notes |
| `Full_Architecture.ipynb` | End-to-end architecture notebook |
| `FULL_ARCHITECTURE_GUIDE.md` | Architecture guide |
| `Small_Language_model.ipynb` | Early SLM experiments |
| `GPU_NOTEBOOK_SETUP.md` | GPU notebook setup |

## Relationship To Production Code

Use notebooks for:

- visualization
- teaching
- debugging individual components
- deriving equations
- early experiments

Use production files for final claims:

- `model.py`
- `training.py`
- `prepare_data.py`
- `dataset.py`
- `tokenizer.py`
- `config.py`

## Paper Usage

When writing the paper:

- cite notebook-derived intuition only if it matches current production code
- use current `project_report.py` numbers for parameter counts
- use `logs/training_metrics.csv` for quantitative results
- use `Prompt_Outputs/` and `logs/samples/` for qualitative examples

## Current Model Components To Cross-Check

Any notebook explanation should match these current components:

- RMSNorm, not LayerNorm
- RoPE, not learned absolute position embeddings
- GQA with 4 KV heads, not default full MHA
- SwiGLU, not ReLU/GELU MLP
- tied embedding and LM head
- 32k byte-level BPE tokenizer

