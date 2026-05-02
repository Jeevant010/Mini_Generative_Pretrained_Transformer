# Hyperparameter Presets

Presets live in `config.py` under the `PRESETS` dictionary. The active preset is:

```python
ACTIVE_PRESET = "subset_10gb"
```

## Preset Summary

| Preset | Purpose | Estimated params | Estimated time |
| --- | --- | ---: | --- |
| `wizard_of_oz_smoke` | Fast sanity check | ~15M | ~5 minutes |
| `wizard_of_oz_full` | Small-corpus deeper run | ~40M | ~30 minutes |
| `subset_1gb` | First real subset | 117.8M with current 768/12-layer architecture | ~14 hours |
| `subset_3gb` | Weekend baseline | 117.8M with current 768/12-layer architecture | ~2 days |
| `subset_10gb` | Current target run | 117.8M with current 768/12-layer architecture | ~5-6 days |
| `full_60gb` | Full larger corpus | 117.8M with current 768/12-layer architecture | ~5-6 weeks |

The latest exact parameter count from `project_report.py` is 117,787,392 for the active `subset_10gb` architecture.

## Active `subset_10gb`

```python
"subset_10gb": {
    "batch_size": 20,
    "block_size": 384,
    "max_iters": 150000,
    "learning_rate": 2.5e-4,
    "min_lr": 2.5e-5,
    "warmup_iters": 2000,
    "eval_iters": 25,
    "eval_interval": 2000,
    "checkpoint_interval": 1000,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_kv_heads": 4,
    "dropout": 0.1,
    "ffn_mult": 3.5,
    "vocab_size": 32000,
}
```

## Choosing A Preset

Use `wizard_of_oz_smoke` when testing:

- code correctness
- checkpoint save/load
- ablation runner behavior
- profiler setup
- generation script behavior

Use `subset_10gb` when producing:

- paper results
- long training curves
- checkpoint milestones
- qualitative samples

Use `full_60gb` only if:

- storage is available
- the run can continue for weeks
- checkpoint cleanup is planned
- thermal and power stability are acceptable

## Tokens Per Step

For any preset:

$$
\text{tokens per step} = \text{batch size} \times \text{block size}
$$

For `subset_10gb`:

$$
20 \times 384 = 7680
$$

## Planned Token Exposure

If `subset_10gb` runs to 150,000 steps:

$$
150{,}000 \times 7680 = 1{,}152{,}000{,}000
$$

token positions will be used for optimization.

Relative to the training file:

$$
\frac{1{,}152{,}000{,}000}{5{,}100{,}766{,}548} \approx 0.226
$$

So even the full 150k-step run is about 22.6 percent of one token-equivalent pass over the 10 GB training file.
