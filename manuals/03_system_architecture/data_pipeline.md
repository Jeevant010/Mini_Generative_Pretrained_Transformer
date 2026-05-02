# Data Pipeline

## Goal

The data pipeline converts large parquet web-text shards into compact token binaries that can be sampled efficiently during training.

The output artifacts are:

```text
train.bin
val.bin
bpe_tokenizer_32k.json
```

## Input Format

The source path is configured in `config.py`:

```python
DATASET_PATH = r"D:\Openweb"
```

The script searches for:

```text
*.parquet
```

and detects one of these text columns:

```text
text, content, document, body
```

Optional columns:

```text
language, lang, language_code
quality_score, score, quality, rank, rating
```

## Filtering

The active filters bias the dataset toward cleaner English text:

| Setting | Value |
| --- | ---: |
| `FILTER_TO_ENGLISH` | True |
| `FILTER_FOR_QUALITY` | True |
| `MIN_DOC_CHARS` | 200 |
| `MAX_DOC_CHARS` | 50000 |
| `MIN_WORD_COUNT` | 50 |
| `MIN_ALPHA_CHAR_RATIO` | 0.55 |
| `MIN_ASCII_ALPHA_RATIO` | 0.85 |
| `MAX_DIGIT_CHAR_RATIO` | 0.20 |
| `MAX_NON_ASCII_CHAR_RATIO` | 0.20 |
| `MIN_ENGLISH_STOPWORD_RATIO` | 0.02 |
| `MAX_URL_COUNT` | 10 |
| `MAX_LINE_REPEAT_RATIO` | 0.30 |

For a document with character count `C`, alphabetic count `A`, and digit count `D`:

$$
r_\alpha = \frac{A}{C}
$$

$$
r_d = \frac{D}{C}
$$

The document is rejected if:

$$
r_\alpha < 0.55
$$

or:

$$
r_d > 0.20
$$

## Tokenizer Training

If `bpe_tokenizer_32k.json` does not exist, the script trains a byte-level BPE tokenizer from a 200 MB sample:

```python
TOKENIZER_SAMPLE_SIZE_MB = 200
vocab_size = 32000
```

If the tokenizer already exists, it is loaded and reused.

## Token Writing

Documents are encoded in batches:

```python
TOKENIZATION_BATCH_SIZE = 128
```

Each encoded document receives an EOS token:

$$
[t_1, ..., t_n] \rightarrow [t_1, ..., t_n, \texttt{<eos>}]
$$

Token IDs are stored as `uint16`:

```python
arr = np.asarray(tokens, dtype=np.uint16)
```

This is valid because:

$$
32000 < 65535
$$

## Train/Validation Split

The validation probability is:

```python
VAL_PERCENT = 0.05
```

For each document:

$$
u \sim \operatorname{Uniform}(0, 1)
$$

If:

$$
u < 0.05
$$

the document goes to validation; otherwise it goes to training.

## Current Data Artifacts

| File | Bytes | Tokens |
| --- | ---: | ---: |
| `train.bin` | 10,201,533,096 | 5,100,766,548 |
| `val.bin` | 535,885,144 | 267,942,572 |

The observed validation fraction is:

$$
\frac{267{,}942{,}572}{5{,}100{,}766{,}548 + 267{,}942{,}572}
\approx 0.0499
$$

which matches the intended 5 percent split.

## Batch Sampling

`dataset.py` opens token binaries with:

```python
np.memmap(path, dtype=np.uint16, mode="r")
```

For each batch element, it samples a start index:

$$
s \sim \operatorname{UniformInteger}(0, N - T - 1)
$$

and creates:

$$
x = [t_s, t_{s+1}, ..., t_{s+T-1}]
$$

$$
y = [t_{s+1}, t_{s+2}, ..., t_{s+T}]
$$

This avoids loading the full 10 GB dataset into RAM.

