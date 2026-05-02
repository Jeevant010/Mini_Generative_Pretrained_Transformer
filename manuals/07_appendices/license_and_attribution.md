# License And Attribution

## Project License

See the repository `LICENSE` file for the project license.

## Project Components

This project uses:

- PyTorch for model implementation and training
- HuggingFace `tokenizers` for byte-level BPE
- NumPy for memory-mapped binary token loading
- PyArrow for parquet ingestion
- tqdm for preprocessing progress display

## Dataset Attribution

The configured dataset path is:

```python
DATASET_PATH = r"D:\Openweb"
```

Any paper, report, or release should name the exact dataset source, version, and license used to create those parquet shards. The local path alone is not enough for reproducibility or attribution.

## Model Attribution

The model architecture is inspired by GPT-style decoder-only Transformers and modern efficient Transformer components:

- decoder-only autoregressive modeling
- RoPE positional encoding
- RMSNorm
- Grouped-Query Attention
- SwiGLU feed-forward blocks
- Flash Attention style scaled dot-product attention
- weight tying

## Reporting Checklist

When publishing results, include:

- dataset source and license
- tokenizer vocabulary size and training sample size
- model parameter count
- hardware used
- training steps
- batch size and context length
- validation loss and perplexity
- checkpoint used for generation
- sampling settings

