# Glossary

## Autoregressive Model

A model that predicts each token from previous tokens:

$$
P(x_1,...,x_T)=\prod_t P(x_t \mid x_{<t})
$$

## Base Model

A model trained for next-token prediction on general text. It is not necessarily instruction-following.

## BPE

Byte Pair Encoding. A subword tokenization algorithm that repeatedly merges frequent adjacent pairs.

## Causal Mask

A mask that prevents a token from attending to future positions.

## Cross-Entropy

The negative log probability assigned to the correct target token.

## GQA

Grouped-Query Attention. An attention variant where multiple query heads share fewer key-value heads.

## Head Dimension

The size of each attention head:

$$
d_h = d / H_q
$$

For this model:

$$
d_h = 768 / 12 = 64
$$

## Perplexity

An exponentiated cross-entropy metric:

$$
\operatorname{PPL}=e^\mathcal{L}
$$

## RMSNorm

Root Mean Square Layer Normalization. It normalizes by activation RMS without subtracting the mean.

## RoPE

Rotary Positional Embedding. A method that injects position by rotating query and key vectors.

## SwiGLU

A gated feed-forward activation:

$$
\operatorname{SwiGLU}(x)=W_{out}(\operatorname{SiLU}(xW_1)\odot xW_2)
$$

## Token

An integer unit produced by the tokenizer. In this project, tokens come from a 32k byte-level BPE vocabulary.

## Weight Tying

Sharing input embedding weights with output LM-head weights:

$$
W_{lm}=E^T
$$

