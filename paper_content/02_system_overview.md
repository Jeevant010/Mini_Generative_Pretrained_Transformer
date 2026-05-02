# System Overview

## End-To-End Pipeline

The project implements an end-to-end language-model training pipeline:

```text
parquet shards
    -> document filtering
    -> BPE tokenizer training/loading
    -> tokenization with <eos>
    -> train.bin and val.bin
    -> memory-mapped random batch sampling
    -> decoder-only GPT training
    -> checkpointing and metrics logging
    -> text generation and qualitative evaluation
```

Each part is implemented as a separate Python module:

| File | Role |
| --- | --- |
| `config.py` | Hyperparameters, architecture settings, ablation toggles, data paths, presets |
| `prepare_data.py` | Parquet scanning, document filtering, tokenizer training, binary dataset writing |
| `tokenizer.py` | Byte-level BPE tokenizer wrapper using HuggingFace `tokenizers` |
| `dataset.py` | Memory-mapped token loading and random batch sampling |
| `model.py` | GPT model, RMSNorm, RoPE, GQA, SwiGLU, generation |
| `training.py` | Production training loop, metrics, evaluation, checkpointing |
| `generate.py` | Checkpoint loading and text generation from a prompt |
| `evaluation/perplexity.py` | Standalone perplexity evaluation |
| `evaluation/sample_generator.py` | Periodic sample generation during training |
| `ablation/run_ablation.py` | Component ablation experiment runner |

## Design Goals

The design is driven by four constraints:

- Limited GPU memory: the project targets an RTX 4060 Laptop GPU.
- Large local data: tokenized data can reach 10 GB or more.
- Reproducibility: training should resume from checkpoints and log measurable metrics.
- Research clarity: each architecture feature can be isolated using ablation toggles.

## Active Configuration

The active preset is `subset_10gb`, which applies the following values:

| Hyperparameter | Value |
| --- | --- |
| `batch_size` | 20 |
| `block_size` | 384 |
| `max_iters` | 150000 |
| `learning_rate` | 2.5e-4 |
| `min_lr` | 2.5e-5 |
| `warmup_iters` | 2000 |
| `eval_iters` | 25 |
| `eval_interval` | 2000 |
| `checkpoint_interval` | 1000 |
| `n_embd` | 768 |
| `n_layer` | 12 |
| `n_head` | 12 |
| `n_kv_heads` | 4 |
| `dropout` | 0.1 |
| `ffn_mult` | 3.5 |
| `vocab_size` | 32000 |

## Model Dimensions

Let:

- `B` be batch size.
- `T` be sequence length.
- `d` be embedding width.
- `L` be number of Transformer blocks.
- `H_q` be number of query heads.
- `H_kv` be number of key-value heads.
- `d_h = d / H_q` be head dimension.
- `V` be vocabulary size.

For the current model:

| Symbol | Value | Meaning |
| --- | --- | --- |
| `B` | 20 | Batch size |
| `T` | 384 | Context length |
| `d` | 768 | Embedding width |
| `L` | 12 | Transformer layers |
| `H_q` | 12 | Query heads |
| `H_kv` | 4 | Key-value heads |
| `d_h` | 64 | Per-head dimension |
| `V` | 32000 | Vocabulary size |

Each training batch contains:

$$
N_{batch} = B \times T = 20 \times 384 = 7680
$$

tokens. After 60,000 steps, the number of consumed training-token positions is approximately:

$$
N_{seen} = 60000 \times 7680 = 460{,}800{,}000
$$

This is not the same as unique dataset tokens because training samples random windows from the memory-mapped dataset.

## Data Artifacts

The current local artifacts are:

| Artifact | Size | Estimated tokens |
| --- | ---: | ---: |
| `train.bin` | 9.50 GB | 5,100,766,548 |
| `val.bin` | 511.06 MB | 267,942,572 |
| `bpe_tokenizer_32k.json` | 2.16 MB | 32,000 vocabulary entries |

Token IDs are stored as `uint16`, so each token uses 2 bytes. This is valid because the vocabulary size is 32,000, which is below the `uint16` maximum value of 65,535.

## Why Memory Mapping Matters

A 10 GB token file cannot be treated like a small Python list without wasting memory. The loader uses:

```python
np.memmap(config.TRAIN_BIN, dtype=np.uint16, mode="r")
```

This keeps the dataset on disk and lets the operating system page in only the slices needed for the current batch. A batch is formed by sampling random start indices and reading windows:

$$
x_i = [t_s, t_{s+1}, ..., t_{s+T-1}]
$$

$$
y_i = [t_{s+1}, t_{s+2}, ..., t_{s+T}]
$$

where `x_i` is the input sequence and `y_i` is the next-token target sequence.

## Training And Evaluation Loop

At each training step:

- A batch is sampled from `train.bin`.
- The model computes logits for all positions.
- Cross-entropy loss is computed against shifted next-token targets.
- Gradients are backpropagated.
- Gradients are clipped.
- AdamW updates the model weights.
- Metrics are written to the console and optionally to CSV.

At each evaluation interval:

- Multiple batches are sampled from train and validation splits.
- Mean train and validation loss are computed.
- Validation perplexity is computed as `exp(validation_loss)`.
- Text samples are generated from fixed prompts.
- The best model is saved if validation loss improves.

## Checkpointing

The project saves two checkpoint types:

- `checkpoints/ckpt_step_<N>.pt`: periodic resume checkpoints.
- `checkpoints/best_model.pt`: best validation checkpoint so far.

Each checkpoint contains:

```text
step
model_state_dict
optimizer_state_dict
loss
best_val_loss
```

This enables interrupted long runs to continue without restarting from step 0.

