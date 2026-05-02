# References

These references support the current project design and terminology.

## Core Transformer And Language Modeling

- Vaswani et al., "Attention Is All You Need", 2017.
- Radford et al., "Improving Language Understanding by Generative Pre-Training", 2018.
- Radford et al., "Language Models are Unsupervised Multitask Learners", 2019.
- Brown et al., "Language Models are Few-Shot Learners", 2020.

## Positional Encoding

- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021.

## Normalization

- Zhang and Sennrich, "Root Mean Square Layer Normalization", 2019.

## Attention Efficiency

- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019.
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023.
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", 2022.

## Feed-Forward Networks

- Shazeer, "GLU Variants Improve Transformer", 2020.

## Tokenization

- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", 2016.
- HuggingFace `tokenizers` documentation.

## Implementation References

- PyTorch documentation for `torch.nn`, `torch.optim.AdamW`, `torch.autocast`, and `torch.nn.functional.scaled_dot_product_attention`.
- NumPy documentation for `numpy.memmap`.
- Apache Arrow and PyArrow documentation for parquet reading.

