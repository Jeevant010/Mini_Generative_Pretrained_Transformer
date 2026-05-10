# Chapter 4.3 — Tokenizing Billions of Words

## The Scale

Our training data contains 5.1 billion tokens. Tokenizing this much text takes hours and requires careful engineering.

## The Process

1. **Read** a parquet shard from disk
2. **Filter** documents (see Chapter 4.2)
3. **Batch** 128 filtered documents together
4. **Tokenize** the entire batch at once using the Rust-backend tokenizer
5. **Append** an `<eos>` token after each document
6. **Split** randomly: 5% goes to `val.bin`, 95% goes to `train.bin`
7. **Write** the token IDs as uint16 bytes to the binary files
8. **Repeat** until we hit the 10 GB target or run out of data

## Why Batch Tokenization?

Tokenizing one document at a time is slow because of Python overhead. By batching 128 documents, we let the Rust backend tokenize them in parallel — this is approximately 10× faster.

## The Target Size

The config specifies a target output size:

```python
DATASET_TARGET_SIZE_GB = 10  # Total for train.bin + val.bin
```

Once the combined size reaches 10 GB, data preparation stops, even if there are more parquet files to process. This gives us control over exactly how much data we use.

## Memory-Mapped Loading

After preparation, we have two large binary files:

| File | Size | Tokens |
|---|---|---|
| `train.bin` | 9.50 GB | 5,100,766,548 |
| `val.bin` | 511 MB | 267,942,572 |

Loading 9.5 GB of data into RAM would be wasteful. Instead, we use **memory mapping** (`numpy.memmap`). This tells the operating system: "This file exists on disk. When I need a piece of it, load just that piece into memory."

```python
data = np.memmap("train.bin", dtype=np.uint16, mode="r")
```

With memory mapping:
- The file stays on disk
- Only the chunks we actually read get loaded into RAM
- The operating system handles caching automatically
- We can access any position in the 5.1-billion-token file instantly

## Random Batch Sampling

During training, each batch needs a random chunk of tokens. The process is:

1. Pick a random starting position between 0 and (5.1 billion - 384)
2. Read 384 consecutive tokens starting from that position
3. The input is tokens [0:383], the target is tokens [1:384]
4. Repeat for all 20 batch elements

This means the model sees a random 384-token window from anywhere in the training data at each step. Over many steps, it gradually sees the entire dataset.
