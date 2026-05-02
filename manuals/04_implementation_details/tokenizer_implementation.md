# Tokenizer Implementation

## Class

The tokenizer is implemented in:

```text
tokenizer.py
```

Class:

```python
BytePairTokenizer
```

It wraps HuggingFace `tokenizers`, which uses a fast Rust backend.

## Construction

```python
self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
self.tokenizer.decoder = decoders.ByteLevel()
```

Special tokens:

```text
<pad>, <bos>, <eos>, <unk>
```

## Training

The training method uses:

```python
train_from_iterator(files_or_iterator, trainer)
```

with:

```python
trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=self.special_tokens,
    min_frequency=2
)
```

The preprocessing script trains from a 200 MB text sample if the tokenizer file does not already exist.

## Encoding

Single text:

```python
ids = tokenizer.encode(text, add_bos=False, add_eos=False)
```

Batch:

```python
ids_batch = tokenizer.encode_batch(texts)
```

Optional BOS/EOS insertion is handled manually after the backend encode call.

## Decoding

```python
text = tokenizer.decode(token_ids, skip_special_tokens=False)
```

Generation uses `skip_special_tokens=True` so BOS/EOS/PAD tokens do not appear in the printed output.

## Special ID Sync

After training or loading, `_sync_special_ids()` builds:

```python
self.special_to_id
```

This map is needed for:

- adding BOS during generation
- adding EOS during data preparation

## Saved Artifact

The tokenizer is saved to:

```text
bpe_tokenizer_32k.json
```

Current size:

```text
2.16 MB
```

## Relationship To Binary Data

The tokenizer defines the mapping:

$$
\text{text} \leftrightarrow \text{token IDs}
$$

The binary files store only token IDs. Therefore, `train.bin` and `val.bin` are meaningful only with the exact tokenizer JSON used to create them.

