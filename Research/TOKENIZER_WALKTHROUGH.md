# Tokenizer Notebook Walkthrough

This document explains exactly what was built in `Research/Tokenizer.ipynb`, why it was built this way, and how to use it in your next LLM notebooks.

## 1) Goal of the notebook

You wanted a proper tokenizer from scratch for an LLM pipeline, trained first on `wizard_of_oz.txt`, and reusable later for:

- embeddings notebook
- attention notebook
- full advanced architecture notebook

The notebook now implements a **byte-level BPE tokenizer** with:

- training
- encoding
- decoding
- special tokens
- save/load serialization
- basic quality checks

## 2) What is implemented (high level)

The tokenizer uses a byte-level approach:

1. Start from a base vocabulary of 256 byte tokens (`0..255`).
2. Add special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`).
3. Learn frequent byte-pair merges from data using BPE.
4. Use learned merges to encode text into token IDs.
5. Decode token IDs back to UTF-8 text.

Why this is a strong baseline:

- Fully reversible at byte level.
- Works for any Unicode text through UTF-8 bytes.
- No external tokenizer dependency required.
- Practical and aligned with how many modern tokenizers are designed.

## 3) Cell-by-cell explanation

### Cell 1 (Markdown)

Introduces the notebook purpose:

- production-style tokenizer
- `wizard_of_oz.txt` as initial dataset
- compatible with later embedding and attention stages

### Cell 2 (Imports)

Imports Python tools needed for a clean implementation:

- `json` for tokenizer serialization
- `re` for pre-tokenization
- `time` for training timing
- `Counter` for pair frequency counting
- `dataclass` for configuration
- `Path` for robust file paths
- typing hints for cleaner code structure

### Cell 3 (Core tokenizer implementation)

Defines two components.

#### A) `TokenizerConfig`

Holds config values:

- `vocab_size` (default 2000)
- `min_pair_freq` (default 2)
- `special_tokens`

#### B) `BytePairTokenizer`

Main class implementing the tokenizer.

Key data structures:

- `token_to_bytes`: token ID -> byte sequence
- `merges`: pair `(a, b)` -> new merged token ID
- `merges_rank`: pair `(a, b)` -> merge order rank
- `special_to_id` and `id_to_special`

Important methods:

- `_init_special_tokens()`
  - assigns IDs after base bytes
- `_pretokenize(text)`
  - splits into whitespace and non-whitespace chunks
- `_merge_sequence(...)`
  - applies one merge pair to a sequence
- `train(text, verbose=True)`
  - learns BPE merges from corpus frequencies
- `_encode_chunk(chunk)`
  - greedily applies best-ranked merges
- `encode(text, add_bos=False, add_eos=False)`
  - encodes full text to token IDs
- `decode(token_ids, skip_special_tokens=False)`
  - reconstructs text from token IDs
- `save(path)` / `load(path)`
  - serializes and restores tokenizer

### Cell 4 (Training on Wizard of Oz)

- Reads `../wizard_of_oz.txt`.
- Creates tokenizer config (`vocab_size=2000`, `min_pair_freq=2`).
- Trains the tokenizer.
- Prints final vocabulary size.

This is where merge rules are learned from real text.

### Cell 5 (Quality and compression check)

Runs a controlled sample sentence through:

- encode
- decode
- exact-match check (after removing printed BOS/EOS markers)

Also computes compression metric:

`bytes_per_token = raw_bytes / token_count`

This helps estimate tokenization efficiency.

### Cell 6 (Save and reload stability)

- Saves tokenizer to `Research/bpe_tokenizer_wizard.json`.
- Loads it back.
- Verifies:
  - encoded sequence is identical before/after reload
  - decoded text is identical before/after reload

This confirms reproducibility and portability.

## 4) Why these design choices are good

1. Byte-level vocabulary

- No out-of-vocabulary issue for characters.
- Works robustly across punctuation and Unicode.

2. Special tokens in config

- Keeps sequence handling clean for training/inference.
- Ready for later transformer input pipelines.

3. Explicit merge ranks

- Deterministic and stable encoding behavior.
- Reproducible tokenization after reload.

4. JSON serialization

- Easy to inspect, version, and reuse.
- No dependency on framework-specific binary formats.

## 5) How to use this tokenizer in your next notebooks

### A) In embedding notebook

Typical flow:

1. Load tokenizer from JSON.
2. Encode text corpus to token IDs.
3. Build fixed-length sequences.
4. Create PyTorch tensors.
5. Feed IDs to embedding layer.

Example:

```python
from pathlib import Path
import torch

# if class is available in the notebook/session
tokenizer = BytePairTokenizer.load(Path("Research") / "bpe_tokenizer_wizard.json")

text = "Your training sample text"
ids = tokenizer.encode(text, add_bos=True, add_eos=True)

x = torch.tensor(ids, dtype=torch.long)
```

### B) For language model training

- Use BOS/EOS consistently.
- Keep PAD token ID fixed.
- Build input-target shifted pairs:
  - input: `tokens[:-1]`
  - target: `tokens[1:]`

### C) For decoding during generation

- Decode predicted token IDs with `decode(...)`.
- Optionally skip special tokens if needed.

## 6) Current baseline status

What is complete:

- from-scratch byte-level BPE implementation
- training on Wizard of Oz
- encode/decode pipeline
- save/load pipeline
- initial quality checks

What can be improved next (recommended roadmap):

1. Speed optimization for large corpus training

- more efficient pair-count updates
- chunked corpus processing

2. Better pre-tokenization policy

- optional GPT-style regex pattern

3. Production extras

- max sequence helpers
- batch encode/decode
- optional unknown token handling policy

4. Evaluation notebook

- compare vocab sizes (2k, 4k, 8k, 16k)
- monitor bytes/token and downstream perplexity

## 7) File outputs created by this notebook

- Notebook implementation: `Research/Tokenizer.ipynb`
- Tokenizer artifact: `Research/bpe_tokenizer_wizard.json`
- This explanation doc: `Research/TOKENIZER_WALKTHROUGH.md`

## 8) Summary

You now have a real, reusable tokenizer foundation for your LLM stack.

The pipeline is already ready to connect to:

- embedding layer creation
- attention blocks
- full transformer training and generation loops

If you want, the next step can be creating a dedicated `tokenizer.py` module from this notebook code so every future notebook imports the exact same tokenizer implementation.

## 9) Embeddings notebook created from scratch

The new embedding notebook is now implemented at `Research/Embeddings.ipynb`.

What it contains:

- loads `Research/bpe_tokenizer_wizard.json`
- encodes `wizard_of_oz.txt` to BPE token IDs
- trains SGNS (skip-gram with negative sampling) embeddings
- saves embedding artifact as `Research/embedding_sgns_wizard.pt`
- provides an LLM-ready `TokenAndPositionEmbedding` module

This gives you a practical embedding baseline before building attention blocks.

## 10) How far can embedding dimension go from scratch?

For your current tokenizer (`vocab_size = 2000`), embedding memory is small.

Approximate SGNS memory (float32 + Adam states):

`memory_bytes ≈ 32 * vocab_size * embedding_dim`

For `vocab_size = 2000`:

- `dim=128` -> about `7.8 MB`
- `dim=256` -> about `15.6 MB`
- `dim=384` -> about `23.4 MB`
- `dim=512` -> about `31.3 MB`
- `dim=768` -> about `46.9 MB`
- `dim=1024` -> about `62.5 MB`

So your bottleneck is usually not embedding tables. The real limit comes later from full transformer training (attention + MLP activations + sequence length + batch size).

Practical recommendation:

- best speed/quality baseline: `dim=256` or `dim=384`
- strong quality for mini-LLM: `dim=512`
- `dim>=768` is possible, but only worth it when model size, data size, and training budget also increase

## 11) RTX 4060 tuning options (with fallback)

Your notebook already includes these profiles:

- `cpu_safe`: `dim=128`, `batch=256`, `negatives=5`, `epochs=2`
- `cpu_quality`: `dim=192`, `batch=256`, `negatives=6`, `epochs=3`
- `rtx_4060_balanced`: `dim=256`, `batch=1024`, `negatives=8`, `epochs=3`
- `rtx_4060_quality`: `dim=384`, `batch=1024`, `negatives=10`, `epochs=4`
- `rtx_4060_max`: `dim=512`, `batch=768`, `negatives=12`, `epochs=4`, `grad_accum=2`

How to pick quickly:

1. Start with `rtx_4060_quality`.
2. If CUDA out-of-memory happens, reduce in this order:
  - lower `batch_size`
  - lower `negatives`
  - lower `max_pairs`
  - lower `dim`
3. If training is stable and GPU usage is low, increase `batch_size` first.

If your notebook shows `Device: cpu` even though you have a 4060:

- verify CUDA build: `torch.cuda.is_available()`
- make sure the environment is using your CUDA-enabled PyTorch install
- select a Python kernel tied to that environment in VS Code

## 12) Best embedding types and proper applications

Use this as your decision map:

1. SGNS / Word2Vec-style token embeddings
- Best for: fast semantic warm-start from raw corpus
- Use when: you want strong local token neighborhoods before transformer training

2. Transformer token embeddings (jointly trained with LM objective)
- Best for: final LLM quality
- Use when: your attention notebook is ready and you begin autoregressive training

3. Learned positional embeddings
- Best for: fixed max context training
- Use when: sequence length is known and stable (for example 256, 512, 1024)

4. RoPE/sinusoidal positional methods
- Best for: better length extrapolation and portability
- Use when: you expect longer context at inference than training

5. Subword-aware embeddings (for example FastText-like ideas)
- Best for: noisy text, typo-heavy data, morphology-rich languages
- Use when: token form variation is very high

For your current roadmap, the strongest path is:

1. SGNS warm-start now (done in `Embeddings.ipynb`)
2. initialize transformer token embedding with these weights
3. continue joint training in the attention + full architecture notebooks
