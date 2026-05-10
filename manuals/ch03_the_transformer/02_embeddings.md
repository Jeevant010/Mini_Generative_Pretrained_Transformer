# Chapter 3.2 — Embeddings

## What Is an Embedding?

A token ID is just a number — like 464 for "The". But a single number does not carry enough information. The model needs a richer representation.

An embedding converts each token ID into a **vector** — a list of 768 numbers. Think of it like giving each word a fingerprint with 768 features. Words with similar meanings end up with similar fingerprints.

## How It Works

The model has a big table with 32,000 rows (one per token) and 768 columns:

```
Token 0   ("〈pad〉"):   [0.012, -0.034, 0.091, ..., 0.005]  ← 768 numbers
Token 1   ("〈bos〉"):   [0.045, 0.023, -0.067, ..., 0.018]  ← 768 numbers
Token 2   ("〈eos〉"):   [-0.031, 0.056, 0.012, ..., -0.042]  ← 768 numbers
...
Token 464 ("The"):     [0.089, -0.012, 0.055, ..., 0.033]  ← 768 numbers
...
Token 31999:           [0.001, 0.077, -0.023, ..., 0.064]  ← 768 numbers
```

When the model sees token 464, it looks up row 464 and gets a vector of 768 numbers. These 768 numbers are the "embedding" of the word "The."

## Learning Embeddings

The 768 numbers for each token are **learned during training**. At the start, they are random. As training progresses, the model adjusts them so that:

- Similar words get similar vectors (e.g., "dog" and "cat" end up close together)
- Different words get different vectors (e.g., "dog" and "economics" end up far apart)
- The vectors capture subtle relationships (e.g., the difference between "king" and "queen" is similar to the difference between "man" and "woman")

## The Numbers

Our embedding table has:

- 32,000 tokens × 768 dimensions = **24,576,000 parameters**

That is about 24.6 million learnable numbers just for the embedding table — roughly 21% of the model's total 118 million parameters.

## Weight Tying

Our model uses a trick called **weight tying**: the embedding table is shared between the input (converting tokens to vectors) and the output (converting vectors back to token probabilities).

In plain English: the same table that converts "The" → vector also converts vector → "The". This saves 24.6 million parameters because we do not need a separate output table.

In the code (`model.py`):
```python
self.token_embed.weight = self.lm_head.weight
```

This single line means the input embedding and output projection share the same 24.6M parameters.

## Why 768 Dimensions?

The embedding dimension (768) determines how much information each token's representation can carry:

- **Too few** (e.g., 64): Not enough room to represent the differences between 32,000 tokens
- **Too many** (e.g., 4096): More parameters than needed, slower training, higher memory usage
- **768**: A proven sweet spot for models of this size (GPT-2 Small also uses 768)

The dimension 768 was chosen because it divides evenly into 12 heads of 64 dimensions each, which is important for the attention mechanism (explained in the next section).
