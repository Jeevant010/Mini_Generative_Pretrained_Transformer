# Inference Pipeline — Generation, Sampling Strategies & Checkpoint Loading

## 1. Overview

`generate.py` provides text generation from trained model checkpoints via CLI.

```bash
python generate.py --prompt "The future of AI is" --max-tokens 100
python generate.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time" --max-tokens 200
```

---

## 2. Inference Pipeline Steps

1. **Device setup**: Auto-detect CUDA/CPU from `config.device`.
2. **Load tokenizer**: `BytePairTokenizer.load(config.TOKENIZER_PATH)`.
3. **Load checkpoint**: Latest `ckpt_step_*.pt` or user-specified path.
4. **Initialize model**: `GPTLanguageModel(config).to(device)`, load state dict.
5. **Encode prompt**: `tokenizer.encode(prompt, add_bos=True)` → tensor.
6. **Generate**: Autoregressive token-by-token generation.
7. **Decode**: `tokenizer.decode(output_ids, skip_special_tokens=True)` → text.

---

## 3. Autoregressive Generation Algorithm

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    self.eval()
    for _ in range(max_new_tokens):
        # Crop to context window
        idx_cond = idx[:, -self.cfg.block_size:]
        # Forward pass
        logits, _ = self(idx_cond)
        # Take last position's logits
        logits = logits[:, -1, :] / temperature
        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

---

## 4. Sampling Parameters

### 4.1 Temperature

Controls randomness of predictions:

| Temperature | Effect |
|-------------|--------|
| 0.0–0.5 | Very deterministic, repetitive |
| 0.7–0.9 | Good balance of coherence and diversity |
| 1.0 | Standard (unmodified distribution) |
| 1.5+ | Very random, potentially incoherent |

Formula: `logits = logits / temperature`

Default in project: **0.8**

### 4.2 Top-k Sampling

Restricts sampling to the top $k$ most probable tokens:

1. Find the $k$-th largest logit value.
2. Set all logits below this threshold to $-\infty$.
3. Renormalize and sample.

Default in project: **k = 50**

### 4.3 Context Window Handling

When generated text exceeds `block_size`, only the last `block_size` tokens are used as context:

```python
idx_cond = idx[:, -self.cfg.block_size:]
```

This is a **sliding window** approach — the model always sees `block_size` (384) tokens of context.

---

## 5. Checkpoint Loading

### 5.1 Automatic Latest

```python
def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_step_")]
    latest = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
    return os.path.join(checkpoint_dir, latest)
```

### 5.2 Manual Selection

```bash
python generate.py --checkpoint checkpoints/best_model.pt
```

### 5.3 State Dict Handling

The loader handles both full checkpoint dicts and raw state dicts:

```python
ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
state_dict = ckpt_data['model_state_dict'] if 'model_state_dict' in ckpt_data else ckpt_data
model.load_state_dict(state_dict)
```

---

## 6. CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | `"Once upon a time"` | Seed text for generation |
| `--max-tokens` | `100` | Number of new tokens to generate |
| `--checkpoint` | Latest | Path to specific checkpoint file |
