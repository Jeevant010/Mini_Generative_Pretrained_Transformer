# Data, Tokenization, And Preprocessing

## Dataset Source And Format

The project expects local parquet shards under:

```python
DATASET_PATH = r"D:\Openweb"
```

Each parquet file is scanned for a text column. Supported candidates are:

```text
text, content, document, body
```

Optional language and quality columns are also detected:

```text
language, lang, language_code
quality_score, score, quality, rank, rating
```

This design allows the same preprocessing script to work with multiple web-text datasets.

## Document Filtering

The preprocessing pipeline includes quality filters before tokenization. A document can be rejected if it is too short, too long, not English-like, too numeric, too URL-heavy, too repetitive, or contains suspicious character repetition.

The main filters include:

| Filter | Current value |
| --- | ---: |
| Minimum document characters | 200 |
| Maximum document characters | 50,000 |
| Minimum word count | 50 |
| Minimum alphabetic character ratio | 0.55 |
| Minimum ASCII alphabetic ratio | 0.85 |
| Maximum digit character ratio | 0.20 |
| Maximum non-ASCII character ratio | 0.20 |
| Minimum English stopword ratio | 0.02 |
| Maximum URL count | 10 |
| Maximum repeated-line ratio | 0.30 |

These filters are heuristic, but they are important because web-text corpora often contain boilerplate, logs, menus, repeated pages, corrupted text, and non-English content.

## English Heuristic

Let:

- `C` be total character count.
- `A` be alphabetic character count.
- `D` be digit character count.
- `N` be non-ASCII character count.

The alphabetic ratio is:

$$
r_\alpha = \frac{A}{C}
$$

The digit ratio is:

$$
r_d = \frac{D}{C}
$$

The non-ASCII ratio is:

$$
r_n = \frac{N}{C}
$$

A document is rejected if:

$$
r_\alpha < 0.55
$$

or:

$$
r_d > 0.20
$$

or:

$$
r_n > 0.20
$$

For stopwords, let:

- `S` be the number of English stopword hits.
- `W` be the number of detected words.

The stopword ratio is:

$$
r_s = \frac{S}{W}
$$

A document is rejected if:

$$
r_s < 0.02
$$

This helps remove text that is unlikely to be natural English.

## Byte-Level BPE Tokenizer

The tokenizer uses HuggingFace `tokenizers` with a BPE model and byte-level pre-tokenization:

```python
self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
self.tokenizer.decoder = decoders.ByteLevel()
```

The vocabulary size is:

$$
V = 32000
$$

Special tokens are:

```text
<pad>, <bos>, <eos>, <unk>
```

## BPE Training Objective

Byte Pair Encoding starts with small units and repeatedly merges frequent adjacent pairs. Given a corpus represented as token sequences, the algorithm counts adjacent token pairs:

$$
c(a,b) = \#((a,b) \text{ appears in corpus})
$$

At each merge step, it selects:

$$
(a^*, b^*) = \arg\max_{(a,b)} c(a,b)
$$

and replaces every occurrence of:

$$
a, b
$$

with a new merged token:

$$
ab
$$

This continues until the target vocabulary size is reached.

Byte-level BPE is useful because every string can be represented without a large unknown-token problem. Even rare words can be decomposed into byte-level or subword units.

## Tokenization Output

Each document is encoded into token IDs:

$$
d_i \rightarrow [t_1, t_2, ..., t_n]
$$

The preprocessing step appends an end-of-sequence token:

$$
[t_1, t_2, ..., t_n, \texttt{<eos>}]
$$

The EOS token helps the model learn document boundaries instead of treating all documents as one continuous text.

## Binary Storage

Token IDs are written as `uint16`:

```python
arr = np.asarray(tokens, dtype=np.uint16)
```

Because:

$$
V = 32000 < 65535
$$

`uint16` is sufficient.

The storage cost is:

$$
\text{bytes} = 2 \times \text{number of tokens}
$$

The current files are:

| File | Bytes | Tokens |
| --- | ---: | ---: |
| `train.bin` | 10,201,533,096 | 5,100,766,548 |
| `val.bin` | 535,885,144 | 267,942,572 |

Total tokenized bytes:

$$
10{,}201{,}533{,}096 + 535{,}885{,}144
= 10{,}737{,}418{,}240
$$

This is almost exactly:

$$
10 \times 1024^3 = 10{,}737{,}418{,}240
$$

bytes, matching the configured 10 GB target.

## Train/Validation Split

The validation split probability is:

$$
p_{val} = 0.05
$$

For each tokenized document, the script randomly writes it to either train or validation:

$$
d_i \in
\begin{cases}
\text{validation}, & u < p_{val} \\
\text{train}, & u \geq p_{val}
\end{cases}
$$

where:

$$
u \sim \operatorname{Uniform}(0,1)
$$

This produces an approximate 95/5 split.

Observed token ratio:

$$
\frac{267{,}942{,}572}
{5{,}100{,}766{,}548 + 267{,}942{,}572}
\approx 0.0499
$$

which is approximately 5 percent.

## Batch Sampling

The dataset loader samples random start positions:

$$
s_b \sim \operatorname{UniformInteger}(0, N - T - 1)
$$

For each batch element:

$$
x_b = [t_{s_b}, t_{s_b+1}, ..., t_{s_b+T-1}]
$$

$$
y_b = [t_{s_b+1}, t_{s_b+2}, ..., t_{s_b+T}]
$$

This produces overlapping training windows. Overlap is not a problem because the objective is local next-token prediction and random windows improve data mixing.

## Effective Training Exposure

At batch size 20 and sequence length 384, each step processes:

$$
7680
$$

token positions.

At 60,000 steps:

$$
60000 \times 7680 = 460{,}800{,}000
$$

token positions have been used for optimization.

Relative to the 5.10B-token training file:

$$
\frac{460{,}800{,}000}{5{,}100{,}766{,}548}
\approx 0.0903
$$

So 60,000 steps correspond to roughly 9.0 percent of one full token-equivalent pass over the training set, assuming no repeated random windows. Because sampling is random with replacement, the actual unique-token coverage will be lower.

This explains why generated text may already be fluent but still inconsistent: the model has learned strong local statistics, but it has not fully saturated the dataset.

