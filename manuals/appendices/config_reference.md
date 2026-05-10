# Configuration Reference

## Preset System

The project uses a preset system in `config.py` to switch between different training scales. Each preset configures all hyperparameters for a specific use case.

### Available Presets

| Preset | Purpose | Data Size | Steps |
|---|---|---|---|
| `wizard_of_oz_smoke` | Quick testing on a tiny text file | ~1 MB | 500 |
| `wizard_of_oz_full` | Full training on the Wizard of Oz book | ~1 MB | 5,000 |
| `subset_10gb` | **Active preset** — full training on 10 GB OpenWebText | 10 GB | 150,000 |
| `full_dataset_60gb` | Full OpenWebText training | 60 GB | 300,000 |

To switch presets:
```python
# In config.py:
ACTIVE_PRESET = "subset_10gb"  # Change this line
```

## Model Architecture Settings

| Parameter | Value | Description |
|---|---|---|
| `n_embd` | 768 | Embedding dimension |
| `n_heads` | 12 | Number of query attention heads |
| `n_kv_heads` | 4 | Number of key-value heads (GQA) |
| `n_layer` | 12 | Number of Transformer blocks |
| `block_size` | 384 | Maximum context length (tokens) |
| `vocab_size` | 32,000 | Vocabulary size |
| `dropout` | 0.1 | Dropout probability |
| `ffn_mult` | 3.5 | Feed-forward expansion multiplier |

## Training Settings

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 20 | Examples per training step |
| `learning_rate` | 3e-4 | Peak learning rate |
| `min_lr` | 3e-5 | Minimum learning rate (after decay) |
| `warmup_steps` | 1,000 | Steps to ramp up learning rate |
| `max_iters` | 150,000 | Total training steps |
| `weight_decay` | 0.1 | AdamW weight decay |
| `beta1` | 0.9 | AdamW first momentum |
| `beta2` | 0.95 | AdamW second momentum |
| `grad_clip` | 1.0 | Maximum gradient norm |

## Data Settings

| Parameter | Value | Description |
|---|---|---|
| `DATASET_PATH` | `D:\Openweb` | Source parquet directory |
| `DATASET_TARGET_SIZE_GB` | 10 | Target combined size for train + val |
| `VAL_PERCENT` | 0.05 | Fraction of data for validation |
| `TOKENIZER_PATH` | `bpe_tokenizer_32k.json` | Tokenizer file |
| `TOKENIZER_SAMPLE_SIZE_MB` | 200 | Training sample size for tokenizer |
| `TOKENIZATION_BATCH_SIZE` | 128 | Documents per tokenization batch |

## Quality Filter Settings

| Parameter | Value | Description |
|---|---|---|
| `FILTER_TO_ENGLISH` | True | Enable English language filtering |
| `FILTER_FOR_QUALITY` | True | Enable quality heuristics |
| `MIN_DOC_CHARS` | 200 | Minimum document length |
| `MAX_DOC_CHARS` | 50,000 | Maximum document length |
| `MIN_WORD_COUNT` | 50 | Minimum word count |
| `MIN_ALPHA_CHAR_RATIO` | 0.55 | Minimum alphabetic character ratio |
| `MIN_ASCII_ALPHA_RATIO` | 0.85 | Minimum ASCII alpha ratio |
| `MAX_DIGIT_CHAR_RATIO` | 0.20 | Maximum digit ratio |
| `MAX_NON_ASCII_CHAR_RATIO` | 0.20 | Maximum non-ASCII ratio |
| `MIN_ENGLISH_STOPWORD_RATIO` | 0.02 | Minimum English stopword frequency |
| `MAX_URL_COUNT` | 10 | Maximum URLs per document |
| `MAX_LINE_REPEAT_RATIO` | 0.30 | Maximum line repetition ratio |

## Ablation Toggle Settings

| Toggle | Default | What Changes When False |
|---|---|---|
| `USE_RMSNORM` | True | Removes all normalization — expect training instability |
| `USE_ROPE` | True | Removes positional encoding — expect grammar degradation |
| `USE_FLASH_ATTENTION` | True | Uses manual attention — slower, more memory, same quality |
| `USE_GQA` | True | Uses full MHA (12 KV heads) — more parameters and memory |

## Logging Settings

| Parameter | Value | Description |
|---|---|---|
| `eval_interval` | 2,000 | Steps between validation evaluations |
| `log_interval` | 100 | Steps between training loss logging |
| `sample_interval` | 2,000 | Steps between sample generation |
| `checkpoint_interval` | 2,000 | Steps between checkpoint saves |
