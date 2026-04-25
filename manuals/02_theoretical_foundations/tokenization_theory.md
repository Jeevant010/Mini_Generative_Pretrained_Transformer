# Tokenization Theory — Byte-Pair Encoding (BPE)

## 1. Introduction

Tokenization is the first stage of any language model pipeline. It converts raw text into a sequence of integer token IDs that the model can process. The choice of tokenizer directly affects vocabulary coverage, compression efficiency, model capacity, and downstream performance.

This project uses **Byte-Pair Encoding (BPE)**, specifically a byte-level variant, which is the dominant tokenization strategy in modern LLMs (GPT-2/3/4, LLaMA, Mistral, Gemma).

---

## 2. Why Tokenization Matters

### 2.1 The Vocabulary Problem

A naive approach — one token per character or one token per word — creates problems:

| Strategy | Vocabulary Size | Sequence Length | Issues |
|----------|----------------|-----------------|--------|
| Character-level | ~100 | Very long | Slow training, poor long-range modeling |
| Word-level | ~500K+ | Short | Huge embedding table, OOV for rare/new words |
| Subword (BPE) | ~32K | Moderate | Balanced: handles rare words, compact sequences |

Subword tokenization splits common words into single tokens and rare/unknown words into smaller subword pieces, achieving a practical balance.

### 2.2 Compression Ratio

The **bytes-per-token** ratio measures how efficiently the tokenizer compresses text:

$$\text{bytes\_per\_token} = \frac{\text{total raw bytes}}{\text{total token count}}$$

Higher values indicate better compression. Typical values for BPE with 32K vocabulary on English text: **3.5–4.5 bytes/token**.

---

## 3. Byte-Pair Encoding Algorithm

### 3.1 Training Phase

BPE was originally proposed by Sennrich et al. (2016) for machine translation. The algorithm:

1. **Initialize** the vocabulary with all individual bytes (256 base tokens) plus special tokens.
2. **Count** all adjacent token pairs in the training corpus.
3. **Merge** the most frequent pair into a new token.
4. **Replace** all occurrences of that pair in the corpus with the new token.
5. **Repeat** steps 2–4 until the desired vocabulary size $V$ is reached.

Each merge produces one new token and one merge rule. The final tokenizer is defined by the ordered list of merge rules.

### 3.2 Encoding Phase

Given a text string and the trained merge rules:

1. Convert text to a sequence of byte tokens.
2. Apply merge rules in **priority order** (most common merges first).
3. At each step, find the highest-priority merge pair present in the sequence and merge it.
4. Repeat until no more applicable merges exist.
5. Return the final token ID sequence.

### 3.3 Decoding Phase

1. Map each token ID to its byte sequence using the vocabulary lookup.
2. Concatenate all byte sequences.
3. Decode the resulting bytes as UTF-8 text.

This process is **fully reversible** — no information is lost.

---

## 4. Byte-Level BPE

### 4.1 Why Byte-Level?

Traditional BPE operates on Unicode characters, which can create issues:

- Characters outside the training distribution become unknown (`<unk>`).
- Unicode normalization differences can cause encoding inconsistencies.

Byte-level BPE (used in GPT-2 and this project) starts from the 256 possible byte values. This guarantees:

- **Zero OOV tokens**: Any text in any language can be encoded.
- **Full reversibility**: Byte-level reconstruction is lossless.
- **Simplicity**: No character-level preprocessing needed.

### 4.2 Special Tokens

This project defines four special tokens:

| Token | ID | Purpose |
|-------|----|---------|
| `<pad>` | 256 | Padding for batched sequences |
| `<bos>` | 257 | Beginning of sequence marker |
| `<eos>` | 258 | End of sequence marker |
| `<unk>` | 259 | Unknown token (rarely used in byte-level BPE) |

---

## 5. Implementation in This Project

### 5.1 Two Tokenizer Implementations

This project contains two tokenizer implementations, reflecting its evolution:

#### A) Research Tokenizer (in notebooks)

- **From-scratch** Python implementation in `Research/Tokenizer.ipynb`.
- Trains on `wizard_of_oz.txt` with `vocab_size = 2000`.
- Outputs `Research/bpe_tokenizer_wizard.json` (merge rules as JSON).
- Used for rapid prototyping and understanding BPE internals.

#### B) Production Tokenizer (`tokenizer.py`)

- **HuggingFace `tokenizers`** library wrapper (Rust backend).
- Class: `BytePairTokenizer`.
- Trains on OpenWebText with `vocab_size = 32000`.
- Outputs `bpe_tokenizer_32k.json`.
- 10–100× faster than the Python implementation for large corpora.

### 5.2 BytePairTokenizer API

```python
class BytePairTokenizer:
    def train(files_or_iterator, vocab_size=32000)  # Train from text
    def encode(text, add_bos=False, add_eos=False)   # Text → token IDs
    def decode(token_ids, skip_special_tokens=False)  # Token IDs → text
    def save(path)                                     # Serialize to JSON
    def load(path) -> BytePairTokenizer               # Restore from JSON
    @property
    def vocab_size -> int                              # Current vocabulary size
```

### 5.3 Training Configuration

| Parameter | Research | Production |
|-----------|----------|------------|
| Vocabulary size | 2,000 | 32,000 |
| Min pair frequency | 2 | 2 |
| Training data | Wizard of Oz (237 KB) | OpenWebText sample (100 MB) |
| Backend | Pure Python | HuggingFace Rust |
| Pre-tokenizer | Whitespace regex | ByteLevel |

---

## 6. Vocabulary Size Trade-offs

Choosing the right vocabulary size involves balancing several factors:

### 6.1 Smaller Vocabulary (e.g., 2K–8K)

**Advantages:**
- Smaller embedding table (fewer parameters).
- Faster softmax computation.
- Better for very small models.

**Disadvantages:**
- Longer token sequences (more tokens per sentence).
- Model must learn more positional/compositional patterns.
- Slower training and inference per text unit.

### 6.2 Larger Vocabulary (e.g., 32K–64K)

**Advantages:**
- Shorter token sequences.
- More semantic information per token.
- Better compression ratio.

**Disadvantages:**
- Larger embedding table.
- Rare tokens may be under-trained.
- Diminishing returns beyond ~50K.

### 6.3 Industry Standards

| Model | Vocabulary Size |
|-------|----------------|
| GPT-2 | 50,257 |
| GPT-4 | ~100,000 |
| LLaMA | 32,000 |
| LLaMA 2 | 32,000 |
| Mistral 7B | 32,000 |
| **This project** | **32,000** |

The choice of 32,000 aligns with LLaMA/Mistral conventions and provides a good balance for the model's scale.

---

## 7. Pre-Tokenization

Before BPE merges are applied, text is split into chunks by a pre-tokenizer. This prevents merges from crossing word or whitespace boundaries.

### 7.1 This Project's Pre-Tokenizer

The HuggingFace `ByteLevel` pre-tokenizer is used:

```python
self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

This converts each byte to a single character in a special byte-level alphabet, allowing the BPE algorithm to work at the byte level while maintaining efficient internal representation.

### 7.2 GPT-2 Style Regex (Alternative)

The GPT-2 pre-tokenizer uses a regex pattern to split text:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
```

This was used in the research notebook's from-scratch tokenizer for educational purposes.

---

## 8. Token Storage Format

### 8.1 Binary Format

After tokenization, token IDs are stored as **uint16** (2 bytes per token) in binary files:

```python
arr = np.asarray(tokens, dtype=np.uint16)
```

This is valid because `vocab_size = 32,000 < 65,536 = 2^16`.

### 8.2 Storage Efficiency

For a corpus of $N$ tokens:
- Binary storage: $2N$ bytes
- Compared to storing raw text (~4 bytes/token average): ~50% compression
- Files: `train.bin` and `val.bin`

---

## 9. Tokenizer Quality Metrics

### 9.1 Compression Ratio

$$\text{compression\_ratio} = \frac{\text{original text bytes}}{\text{tokenized binary bytes}}$$

### 9.2 Fertility (tokens per word)

$$\text{fertility} = \frac{\text{total tokens}}{\text{total words}}$$

Lower fertility means the tokenizer produces more natural word-level representations.

### 9.3 Round-Trip Accuracy

$$\text{accuracy} = \begin{cases} 1 & \text{if decode(encode(text)) = text} \\ 0 & \text{otherwise} \end{cases}$$

For byte-level BPE, this should always be 1.0 (lossless).

---

## 10. References

1. Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units." *ACL*.
2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI* (introduced byte-level BPE).
3. Kudo, T. & Richardson, J. (2018). "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing." *EMNLP*.
4. HuggingFace Tokenizers Library. https://github.com/huggingface/tokenizers
