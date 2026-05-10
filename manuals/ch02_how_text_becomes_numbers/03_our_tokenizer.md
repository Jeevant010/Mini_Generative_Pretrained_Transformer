# Chapter 2.3 — Our Tokenizer

## The Implementation

Our tokenizer lives in `tokenizer.py`. It is a wrapper around the HuggingFace `tokenizers` library, which is written in Rust for speed. The Python code is simple — the heavy work happens in the Rust backend.

## Key Numbers

| Property | Value |
|---|---|
| Vocabulary size | 32,000 tokens |
| Algorithm | Byte-level BPE |
| Training sample | 200 MB of filtered English text |
| Tokenizer file | `bpe_tokenizer_32k.json` (2.16 MB) |
| Library | HuggingFace `tokenizers` (Rust backend) |

## How the Tokenizer Is Used

### During Data Preparation

When we prepare the training data (`prepare_data.py`), every document gets tokenized:

```
"The future of artificial intelligence..." → [464, 2003, 286, 11666, 4430, ...]
```

These numbers are saved to disk as binary files (`train.bin` and `val.bin`). The model never sees text — it only sees these numbers.

### During Training

The model reads batches of numbers from `train.bin`. Each number is a token ID between 0 and 31,999.

### During Generation

When you give the model a prompt, it:
1. Tokenizes your prompt into numbers
2. Generates new numbers one at a time
3. Converts the numbers back to text using the tokenizer

## Encoding and Decoding

**Encoding** converts text to numbers:
```python
tokenizer.encode("Hello, world!")  # → [15496, 11, 995, 0]
```

**Decoding** converts numbers back to text:
```python
tokenizer.decode([15496, 11, 995, 0])  # → "Hello, world!"
```

Encoding and decoding are always reversible — you can go back and forth without losing information.

## Batch Encoding

For efficiency, the tokenizer can encode many texts at once:

```python
texts = ["First document...", "Second document...", "Third document..."]
all_tokens = tokenizer.encode_batch(texts)
```

During data preparation, we encode 128 documents at a time (`TOKENIZATION_BATCH_SIZE = 128`). This is much faster than encoding one at a time because the Rust backend can parallelize the work.

## Token Storage

Tokens are stored as `uint16` (unsigned 16-bit integers). Each token ID takes exactly 2 bytes on disk.

Why `uint16` works:
- `uint16` can store values from 0 to 65,535
- Our vocabulary has 32,000 tokens (max ID = 31,999)
- 31,999 < 65,535, so every token fits

The training file has about 5.1 billion tokens:
- Storage: 5,100,766,548 × 2 bytes = about 9.5 GB
- This is much smaller than the original text (which was ~38 GB before filtering)

## What the Tokenizer Does NOT Do

The tokenizer is a simple text-to-numbers converter. It does not:
- Understand the meaning of words
- Know grammar or syntax
- Distinguish good text from bad text
- Filter languages

All of that is handled by other parts of the pipeline (the data filter and the model itself).
