# Checkpoint Management — Resume, Best-Model Tracking & Artifact Inventory

## 1. Checkpoint Types

| Type | Filename Pattern | Trigger | Purpose |
|------|-----------------|---------|---------|
| Periodic | `checkpoints/ckpt_step_<N>.pt` | Every `checkpoint_interval` steps | Resume training after interruption |
| Best Validation | `checkpoints/best_model.pt` | When `val_loss` improves | Inference / deployment |

---

## 2. Checkpoint Contents

Each `.pt` file is a Python dictionary saved with `torch.save()`:

```python
{
    'step': int,                    # Training step at save time
    'model_state_dict': OrderedDict,# All model parameters
    'optimizer_state_dict': dict,   # AdamW momentum & variance buffers
    'loss': float,                  # Last training batch loss
    'best_val_loss': float,         # Best validation loss seen so far
}
```

---

## 3. Auto-Resume Mechanism

When `training.py` starts:

1. Scan `checkpoints/` for files matching `ckpt_step_*.pt`.
2. Sort by step number (extracted from filename).
3. Load the highest-step checkpoint.
4. Restore model weights, optimizer state, step counter, and best val loss.
5. Resume training from `step + 1`.

No special flag or CLI argument is needed — resume is fully automatic.

---

## 4. Best-Model Tracking

During evaluation:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(checkpoint, "checkpoints/best_model.pt")
```

Only one `best_model.pt` exists at any time — it is overwritten whenever validation loss improves.

---

## 5. Checkpoint Size

| Component | Size |
|-----------|------|
| Model state dict | ~355 MB |
| Optimizer state dict | ~745 MB |
| Metadata | < 1 KB |
| **Total per checkpoint** | **~1.1 GB** |

With `checkpoint_interval = 1,000` over 300K steps: up to 300 periodic checkpoints (~330 GB total). Consider periodically deleting old checkpoints.

---

## 6. Artifact Inventory

### Production Artifacts

| Artifact | Location | Producer |
|----------|----------|----------|
| `bpe_tokenizer_32k.json` | Project root | `prepare_data.py` |
| `train.bin` | Project root | `prepare_data.py` |
| `val.bin` | Project root | `prepare_data.py` |
| `ckpt_step_<N>.pt` | `checkpoints/` | `training.py` |
| `best_model.pt` | `checkpoints/` | `training.py` |
| `performance_trace.json` | Project root | `training.py` (profiler) |

### Research Artifacts

| Artifact | Location | Producer |
|----------|----------|----------|
| `bpe_tokenizer_wizard.json` | `Research/` | Tokenizer.ipynb |
| `embedding_sgns_wizard.pt` | `Research/` | Embeddings.ipynb |
| `attention_model_wizard.pt` | `Research/` | Attention.ipynb |
| `full_architecture_model_wizard.pt` | `Research/` | Full_Architecture.ipynb |
| `full_arch_last.pt` | `Research/checkpoints_full_arch/` | Full_Architecture.ipynb |

---

## 7. Checkpoint Compatibility

**Important**: Checkpoints are tied to the model architecture defined in `config.py`. If you change any of these values, existing checkpoints become incompatible:

- `n_embd`, `n_layer`, `n_head`, `n_kv_heads`
- `ffn_mult`, `vocab_size`, `dropout`

Always start fresh (delete `checkpoints/`) when changing architecture parameters.
