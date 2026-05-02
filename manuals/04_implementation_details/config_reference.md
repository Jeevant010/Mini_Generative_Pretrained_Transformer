# Config Reference

`config.py` is the single source of truth for the project. It controls device selection, architecture, training, data preparation, logging, profiling, and ablation behavior.

## Device

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

The current observed environment uses CUDA on an NVIDIA GeForce RTX 4060 Laptop GPU.

## Active Preset

```python
ACTIVE_PRESET = "subset_10gb"
```

When `ACTIVE_PRESET` is not `None`, values from `PRESETS[ACTIVE_PRESET]` override the manual defaults above it.

## Active Training Values

| Name | Value |
| --- | ---: |
| `batch_size` | 20 |
| `block_size` | 384 |
| `max_iters` | 150000 |
| `learning_rate` | 2.5e-4 |
| `min_lr` | 2.5e-5 |
| `warmup_iters` | 2000 |
| `lr_decay_iters` | 150000 |
| `grad_clip` | 1.0 |
| `eval_iters` | 25 |
| `eval_interval` | 2000 |
| `checkpoint_interval` | 1000 |

Tokens per optimization step:

$$
B \times T = 20 \times 384 = 7680
$$

## Active Architecture Values

| Name | Value |
| --- | ---: |
| `n_embd` | 768 |
| `n_layer` | 12 |
| `n_head` | 12 |
| `n_kv_heads` | 4 |
| `dropout` | 0.1 |
| `ffn_mult` | 3.5 |
| `vocab_size` | 32000 |

The head dimension is:

$$
d_h = \frac{768}{12} = 64
$$

The SwiGLU hidden width is:

$$
d_{ff} = \lfloor 3.5 \times 768 \rfloor = 2688
$$

## Data Paths

| Name | Value |
| --- | --- |
| `TRAIN_BIN` | `train.bin` |
| `VAL_BIN` | `val.bin` |
| `TOKENIZER_PATH` | `bpe_tokenizer_32k.json` |
| `DATASET_PATH` | `D:\Openweb` |

## Data Preparation Values

| Name | Value |
| --- | ---: |
| `DATASET_TARGET_SIZE_GB` | 10.0 |
| `TOKENIZER_SAMPLE_SIZE_MB` | 200 |
| `TOKENIZATION_BATCH_SIZE` | 128 |
| `VAL_PERCENT` | 0.05 |
| `PREP_RANDOM_SEED` | 1337 |
| `SKIP_FULL_ROW_COUNT_SCAN` | True |
| `SHUFFLE_PARQUET_FILES` | True |
| `MAX_PARQUET_FILES` | None |

## Quality Filters

| Name | Value |
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

## Ablation Toggles

| Toggle | Default | Meaning |
| --- | --- | --- |
| `USE_RMSNORM` | True | Use RMSNorm before attention/FFN and final norm |
| `USE_ROPE` | True | Apply RoPE to Q and K |
| `USE_FLASH_ATTENTION` | True | Use PyTorch scaled dot-product attention |
| `USE_GQA` | True | Use 4 KV heads instead of full 12 KV heads |

Flip exactly one toggle at a time for controlled ablations.

## Logging

| Name | Value |
| --- | --- |
| `LOG_DIR` | `logs` |
| `LOG_METRICS_CSV` | True |
| `LOG_GRAD_NORM` | True |
| `LOG_VRAM` | True |
| `TRAIN_LOG_INTERVAL` | 1 |

Metrics are written to:

```text
logs/training_metrics.csv
```

## Sample Generation

| Name | Value |
| --- | --- |
| `GENERATE_SAMPLES` | True |
| `SAMPLE_MAX_TOKENS` | 80 |
| `SAMPLE_TEMPERATURE` | 0.8 |
| `SAMPLE_TOP_K` | 50 |

Sample files are written under:

```text
logs/samples/
```

