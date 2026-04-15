# Embeddings Notebook Detailed Analysis

Notebook analyzed: Research/Embeddings.ipynb

This document gives a deep analysis of the current notebook implementation, including architecture, training objective, complexity, bottlenecks, tuning behavior, and what to improve next.

## 1) What this notebook does end to end

The notebook builds token embeddings from scratch using your trained BPE tokenizer and SGNS training.

Pipeline:

1. Load Python and PyTorch stack, set seed, detect CPU or CUDA.
2. Load BPE tokenizer artifact from JSON and encode wizard_of_oz corpus.
3. Build hardware-aware training config profile.
4. Construct skip-gram pairs and negative-sampling distribution.
5. Train SkipGramNS model (input and output embedding tables).
6. Normalize and inspect nearest neighbors in embedding space.
7. Export embedding artifact and test token+position embedding module for LLM use.

## 2) Notebook structure by cell number

The notebook currently has 10 cells total (8 code + 2 markdown).

### Cell 1 (Markdown)

Purpose statement and scope of notebook.

Strength:

- Clear framing of goals before code starts.

### Cell 2 (Imports, seed, device)

Main actions:

- Imports all required dependencies.
- Sets deterministic seeds for random, numpy, torch.
- Detects device and prints GPU name/VRAM when CUDA is available.

Strength:

- Reproducibility and environment visibility are handled early.

Risk:

- Full determinism is not guaranteed unless deterministic backend flags are also set.

### Cell 3 (BPETokenizerRuntime)

Main actions:

- Reconstructs tokenizer from saved JSON merges.
- Maintains merge rank order for deterministic BPE encoding.
- Supports BOS/EOS insertion and token ID generation.

Strengths:

- Keeps embedding notebook independent from training notebook runtime state.
- Reuses exact tokenizer behavior through saved merges.

Risks / notes:

- BPE encode loop is merge-rank greedy and scans adjacent pairs each pass; this is correct but not the fastest possible implementation.

### Cell 4 (Load artifacts and corpus encode)

Main actions:

- Validates paths for tokenizer JSON and text corpus.
- Loads corpus text and encodes to token IDs.
- Computes token frequency summary.

Strength:

- Fails fast on missing files with clear error messages.

Risk:

- Path assumptions are relative to notebook location; moving notebook can break paths unless updated.

### Cell 5 (Training profile selection)

Main actions:

- Defines EmbeddingTrainingConfig dataclass.
- Builds CPU and RTX 4060 oriented profile presets.
- Estimates SGNS memory footprint and auto-selects profile by detected hardware.

Strengths:

- Practical profile system for your hardware progression.
- Good first-level guard against over-aggressive settings.

Risk:

- Memory estimate focuses on embedding table and optimizer scale; it does not fully include temporary activations and dataloader overhead.

### Cell 6 (Pairs, dataset, noise dist, model)

Main actions:

- Creates skip-gram center-context pairs.
- Creates dataset and dataloader.
- Builds negative-sampling distribution with exponent 0.75.
- Defines SkipGramNS model with separate input/output embeddings.

Strengths:

- Uses standard Word2Vec-style negative sampling behavior.
- Power 0.75 is a proven heuristic.

Design choice note:

- Pair construction chooses one random context for each center position. This reduces compute and memory versus generating all context pairs.

### Cell 7 (Training loop)

Main actions:

- Trains SGNS with AdamW optimizer.
- Uses AMP only on CUDA.
- Supports gradient accumulation.
- Logs average loss per epoch.

Strengths:

- Clean split between CPU and CUDA path.
- Updated GradScaler API usage is modern and warning-safe.

Risk:

- No validation metric is tracked (only train loss). Embedding quality is estimated later by nearest neighbors only.

### Cell 8 (Nearest-neighbor inspection)

Main actions:

- L2-normalizes learned input embeddings.
- Computes cosine-similar nearest neighbors for top frequent tokens.

Strength:

- Quick semantic sanity check with very low extra cost.

Risk:

- Frequent whitespace or formatting tokens can dominate top-neighbor reporting and make qualitative interpretation harder.

### Cell 9 (Export + token/position wrapper)

Main actions:

- Defines TokenAndPositionEmbedding module.
- Saves learned SGNS embeddings and metadata artifact.
- Runs a shape sanity check for token+position output.

Strengths:

- Immediately bridges this notebook to attention notebook stage.
- Artifact contains enough metadata for reproducibility.

Risk:

- Position embedding is learned absolute only in this wrapper; longer-context extrapolation will require architecture decision later.

### Cell 10 (Embedding families markdown)

Main actions:

- Brief decision list for embedding method selection.

Strength:

- Good conceptual bridge to the next notebooks.

## 3) Mathematical objective used

The training objective in SkipGramNS is:

- Positive term: maximize similarity of center token and true context token.
- Negative term: minimize similarity between center token and sampled negative tokens.

For one sample:

- Positive: log(sigmoid(v_c dot u_o))
- Negative: sum over k negatives of log(sigmoid(-v_c dot u_k))

Batch loss is the negative mean of positive + negative terms.

This is the standard SGNS objective used in Word2Vec style training.

## 4) Time and space complexity analysis

Let:

- N = number of token IDs in corpus
- W = max window size
- D = embedding dimension
- K = negatives per example
- B = batch size
- E = epochs

### Pair generation

Current method generates around O(N) pairs (one context sample per center).

- Time: O(N \* W) worst case due window slicing and filtering
- Memory: O(P) where P is generated pair count

### SGNS forward/backward

Per batch cost is roughly proportional to B _ (K + 1) _ D.

- Time per epoch: O((P / B) _ B _ (K + 1) _ D) = O(P _ (K + 1) \* D)
- Time full training: O(E _ P _ (K + 1) \* D)

### Model memory

Two embedding tables are stored:

- input table: vocab_size x D
- output table: vocab_size x D

Parameter count: 2 _ vocab_size _ D

With optimizer states, practical memory for SGNS parameters scales linearly with D.

## 5) Quality analysis of current design

### What is strong now

1. Full from-scratch implementation with no hidden framework shortcuts.
2. Reproducible tokenizer to embedding handoff via JSON artifact.
3. Hardware-aware profile strategy already integrated.
4. Exported artifact is directly reusable for downstream transformer.
5. Minimal but useful embedding-space diagnostics included.

### What limits quality right now

1. One-context-per-center sampling underuses available co-occurrence signal.
2. Training corpus is a single book, so semantics are narrow domain.
3. No held-out intrinsic score (for example similarity benchmark or downstream proxy).
4. Frequent formatting tokens can bias nearest-neighbor checks.
5. SGNS objective is good for warm start, but final LLM quality still depends on joint training with attention stack.

## 6) Practical tuning guidance from this implementation

### CPU-first path

Use cpu_safe or cpu_quality profiles.

When CPU is slow:

1. reduce max_pairs
2. reduce negatives
3. reduce dim
4. reduce epochs

### RTX 4060 path

Recommended order:

1. start with rtx_4060_quality
2. if OOM, reduce batch_size first
3. then reduce negatives
4. then reduce max_pairs
5. finally reduce dim

When stable and under-utilized:

1. increase batch_size first
2. then increase max_pairs
3. then consider dim increase

## 7) Output artifact analysis

Saved artifact: Research/embedding_sgns_wizard.pt

Contents include:

- token_embedding (input embeddings)
- output_embedding (context embeddings)
- loss_history
- selected profile and config
- vocab size
- tokenizer JSON path

This is enough to:

1. initialize token embedding weights in transformer model
2. compare future training checkpoints to warm-start baseline
3. keep reproducibility metadata for experiments

## 8) Recommended upgrades (priority order)

### Priority 1: Better signal per token

Change pair generation to include multiple contexts per center (or all contexts within sampled window).

Expected impact:

- Better semantic structure at same dimension.

### Priority 2: Better corpus coverage

Train on your larger dataset with the same tokenizer.

Expected impact:

- Better generalization and stronger neighborhoods.

### Priority 3: Stronger evaluation

Add quantitative checks:

- average cosine agreement for nearest-neighbor stability across seeds
- downstream proxy loss when embedding is used in a tiny language model block

Expected impact:

- More objective profile selection than visual neighbor inspection.

### Priority 4: Throughput optimizations

- increase dataloader workers on GPU setups
- precompute pair tensors once and memory-map if large
- optional fused sampling or batched multinomial improvements

Expected impact:

- Faster wall-clock training for larger corpora.

## 9) How this notebook fits your roadmap

Your roadmap was:

1. tokenizer
2. embeddings
3. attention
4. full advanced architecture

Current status:

- tokenizer stage: complete
- embedding stage: complete baseline with export
- next stage: attention notebook should load token_embedding from artifact and continue end-to-end LM training

## 10) Final assessment

This notebook is a strong practical baseline for from-scratch embeddings.

It is not yet the final embedding quality ceiling, but it is exactly the right bridge into attention and transformer training:

- correct objective
- reproducible artifacts
- hardware-aware profiles
- LLM-ready token+position interface

For best next-step gains, prioritize larger corpus training plus multi-context pair generation, then fine-tune embeddings jointly inside the attention-based model.
