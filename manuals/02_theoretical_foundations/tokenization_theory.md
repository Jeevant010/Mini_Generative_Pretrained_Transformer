# Tokenization Theory

## Why Tokenization Matters

Neural language models operate on integer token IDs, not raw strings. Tokenization defines the unit of prediction. A poor tokenizer can waste context length, inflate sequence lengths, and create many unknown tokens.

This project uses byte-level Byte Pair Encoding (BPE), implemented through the HuggingFace `tokenizers` library.

## Byte-Level BPE

The tokenizer starts from byte-level units and learns frequent merges. This gives two useful properties:

- Any text can be represented.
- Common words and subwords become compact tokens.

The configured vocabulary size is:

$$
V = 32000
$$

Special tokens are:

```text
<pad>, <bos>, <eos>, <unk>
```

## BPE Merge Objective

Given a corpus represented as token sequences, BPE counts adjacent token pairs:

$$
c(a,b) = \#((a,b) \text{ appears in the corpus})
$$

At each step, it selects the most frequent pair:

$$
(a^*, b^*) = \arg\max_{(a,b)} c(a,b)
$$

and replaces occurrences of:

$$
a, b
$$

with a new merged token:

$$
ab
$$

The process continues until the vocabulary reaches the target size.

## Tokenizer Implementation

The project uses:

```python
Tokenizer(models.BPE(unk_token="<unk>"))
pre_tokenizers.ByteLevel(add_prefix_space=False)
decoders.ByteLevel()
```

This means text is first represented at byte level, BPE merges are applied, and decoding reconstructs the original text form.

## BOS And EOS

The tokenizer supports optional beginning-of-sequence and end-of-sequence tokens:

```python
encode(text, add_bos=False, add_eos=False)
```

During data preparation, every document receives an EOS token:

$$
[t_1, t_2, ..., t_n] \rightarrow [t_1, t_2, ..., t_n, \texttt{<eos>}]
$$

During generation, the prompt is encoded with BOS:

```python
context_ids = tokenizer.encode(args.prompt, add_bos=True)
```

## Why `uint16` Works

The vocabulary size is 32,000. A `uint16` can represent values from 0 to 65,535, so every token ID fits:

$$
32000 < 65535
$$

Storage cost is:

$$
\text{bytes} = 2 \times \text{tokens}
$$

The current training file has:

$$
5{,}100{,}766{,}548
$$

tokens, so it uses approximately:

$$
2 \times 5{,}100{,}766{,}548 =
10{,}201{,}533{,}096
$$

bytes.

## Tokenization And Model Quality

A tokenizer trained on a 200 MB text sample may not perfectly represent the full 10 GB corpus, but it is large enough to learn common web-text patterns. Better tokenizer quality usually reduces average tokens per document and improves effective context usage.

For future experiments, useful tokenizer diagnostics include:

- average characters per token
- unknown token frequency
- token length distribution
- most common tokens
- examples of encoded/decoded text

