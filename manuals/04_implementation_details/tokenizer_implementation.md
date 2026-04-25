# Tokenizer Implementation — BytePairTokenizer API & Design

## 1. Class Overview

`tokenizer.py` wraps the HuggingFace `tokenizers` library (Rust backend) in a clean Python API.

```python
class BytePairTokenizer:
    """Production-ready BPE tokenizer wrapper using the HuggingFace tokenizers library."""
```

---

## 2. Constructor

```python
def __init__(self, config=None):
    self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    self.tokenizer.decoder = decoders.ByteLevel()
    self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    self.special_to_id = {}
```

- **BPE model**: Initialized with `<unk>` as the unknown token.
- **ByteLevel pre-tokenizer**: Converts bytes to alphabet characters before BPE.
- **ByteLevel decoder**: Reverses the byte-level encoding during decoding.

---

## 3. Training

```python
def train(self, files_or_iterator, vocab_size=32000, verbose=True):
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=self.special_tokens,
        min_frequency=2,
        show_progress=verbose
    )
    self.tokenizer.train_from_iterator(files_or_iterator, trainer)
    self._sync_special_ids()
```

Accepts a single string, list of strings, or any iterator. The Rust backend handles the heavy computation.

---

## 4. Encoding

```python
def encode(self, text, add_bos=False, add_eos=False) -> List[int]:
    output = self.tokenizer.encode(text)
    ids = output.ids
    if add_bos: ids = [self.special_to_id["<bos>"]] + ids
    if add_eos: ids = ids + [self.special_to_id["<eos>"]]
    return ids
```

---

## 5. Decoding

```python
def decode(self, token_ids, skip_special_tokens=False) -> str:
    return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
```

---

## 6. Persistence

```python
def save(self, path):
    self.tokenizer.save(str(path))

@classmethod
def load(cls, path) -> "BytePairTokenizer":
    instance = cls()
    instance.tokenizer = Tokenizer.from_file(str(path))
    instance._sync_special_ids()
    return instance
```

Serializes to JSON via the HuggingFace library. Fully portable and human-inspectable.

---

## 7. Special Token Management

```python
def _sync_special_ids(self):
    vocab = self.tokenizer.get_vocab()
    for tok in self.special_tokens:
        if tok in vocab:
            self.special_to_id[tok] = vocab[tok]
```

This synchronizes the `special_to_id` mapping after training or loading, enabling `encode(add_bos=True)` and `EOS_ID` lookup in the data pipeline.
