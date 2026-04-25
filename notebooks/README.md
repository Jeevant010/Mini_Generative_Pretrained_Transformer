# Notebooks — Ablation Study Experiments

These Jupyter notebooks provide **interactive, visual proof** that each architectural component
is mathematically necessary. Run them on `wizard_of_oz.txt` — each takes ~5-10 minutes.

## Notebooks

| # | Notebook | What It Proves | Time |
|---|----------|---------------|------|
| 1 | `01_Positional_Embedding_Check.ipynb` | RoPE is essential for word order | ~5 min |
| 2 | `02_LayerNorm_Check.ipynb` | RMSNorm prevents gradient explosion | ~5 min |
| 3 | `03_Flash_Attention_Check.ipynb` | Flash Attention saves VRAM & speed | ~5 min |

## How to Run

```bash
cd notebooks
jupyter notebook
```

Or in VS Code: Open any `.ipynb` file and click **Run All**.

## Outputs

Each notebook generates:
- A **matplotlib plot** saved as `.png` in the notebooks directory
- A **comparison table** printed in the final cell
- **Generated text samples** for qualitative comparison (notebook 1)

## Relationship to Research/ Folder

The `Research/` folder contains your **original exploration notebooks** (Tokenizer, Embeddings, Attention, Full Architecture) — those stay untouched.

These `notebooks/` are specifically designed for **ablation studies** that produce paper-ready figures and tables.

## Temporary Files

These notebooks create `_ablation_data/` inside the notebooks folder for temporary binaries. This folder is gitignored.
