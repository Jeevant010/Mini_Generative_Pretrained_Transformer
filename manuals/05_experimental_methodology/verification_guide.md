# Verification & Cross-Check Guide — Research Paper Readiness

This document maps **exactly** how to verify and cross-check every claim you make in your research paper. Each section corresponds to a paper claim and tells you which code, notebook, or script produces the proof.

---

## 1. Positional Embedding Verification (RoPE)

### Paper Claim
> "Rotary Positional Embeddings are essential for the model's ability to understand word order.
> Without them, the model degenerates to a bag-of-words model with severely broken grammar."

### How to Verify

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Run `notebooks/01_Positional_Embedding_Check.ipynb` | Loss comparison plot saved |
| 2 | Check loss curves | No-RoPE loss plateaus higher than baseline |
| 3 | Check generated text | No-RoPE text has random word order |
| 4 | Check perplexity numbers | No-RoPE PPL is 30-100% higher |

### Cross-Check
- Open `model.py` → `GroupedQueryAttention.forward()` → look for `if self.use_rope:`
- When `False`, the `apply_rope()` call is skipped — Q and K are used directly
- The mathematical proof: self-attention is permutation-equivariant → $\text{Attention}(Px) = P \cdot \text{Attention}(x)$

### Files That Produce This Evidence
- `notebooks/01_Positional_Embedding_Check.ipynb` — interactive notebook
- `ablation/run_ablation.py` — automated comparison table
- `model.py` lines with `self.use_rope` — code toggle
- `config.py` → `USE_ROPE = True/False` — toggle switch

---

## 2. LayerNorm (RMSNorm) Verification

### Paper Claim
> "Pre-normalization via RMSNorm is a load-bearing component. Without it, the compounding
> variance through 12 residual layers causes gradient explosion and training collapse within
> 100-500 steps."

### How to Verify

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Run `notebooks/02_LayerNorm_Check.ipynb` | Loss + gradient norm plots saved |
| 2 | Check loss curve | No-RMSNorm loss goes to NaN |
| 3 | Check gradient norm (log scale) | Spike to infinity right before NaN |
| 4 | Note the exact step of collapse | Document for the paper |

### Cross-Check
- Open `model.py` → `TransformerBlock.__init__()` → look for `RMSNorm if use_rmsnorm else Identity()`
- When `False`, `Identity()` passes inputs through unchanged
- Mathematical proof: $\|x_L\| \approx \|x_0\| + \sum_{l=0}^{L-1} \|f(x_l)\|$ → unbounded growth

### Files That Produce This Evidence
- `notebooks/02_LayerNorm_Check.ipynb` — interactive notebook with gradient tracking
- `ablation/run_ablation.py` — automated NaN detection
- `model.py` → `Identity` class and `RMSNorm` class
- `config.py` → `USE_RMSNORM = True/False`

---

## 3. Flash Attention Verification

### Paper Claim
> "Flash Attention is a hardware optimization that produces mathematically identical results
> while reducing VRAM usage by ~40-50% and increasing throughput by 2-3×."

### How to Verify

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Run `notebooks/03_Flash_Attention_Check.ipynb` | Speed + VRAM comparison bar chart |
| 2 | Compare final loss values | Should be nearly identical (< 0.01 difference) |
| 3 | Compare VRAM | Standard uses ~2× more |
| 4 | Compare tokens/sec | Flash is ~2-3× faster |

### Cross-Check
- Open `model.py` → `GroupedQueryAttention.forward()` → look for `if self.use_flash:`
- When `True`: uses `F.scaled_dot_product_attention(is_causal=True)` — PyTorch's fused kernel
- When `False`: uses `manual_causal_attention()` — explicit Q×K^T → mask → softmax → ×V
- Mathematical proof: both compute the same function, but Flash operates in O(T) memory vs O(T²)

### Files That Produce This Evidence
- `notebooks/03_Flash_Attention_Check.ipynb` — benchmark with bar charts
- `ablation/run_ablation.py` → "No Flash Attention" row in comparison table
- `model.py` → `manual_causal_attention()` function (the slow path)
- `config.py` → `USE_FLASH_ATTENTION = True/False`

---

## 4. GQA vs Full MHA Verification

### Paper Claim
> "Grouped-Query Attention reduces KV-cache memory by sharing K/V heads across query groups,
> enabling larger batch sizes on consumer GPUs with minimal quality trade-off."

### How to Verify

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Run `python -m ablation.run_ablation` | "Full MHA (no GQA)" row in table |
| 2 | Compare VRAM | Full MHA uses ~25% more |
| 3 | Compare loss | Full MHA may be slightly better |
| 4 | Compare params | Full MHA has more K/V parameters |

### Cross-Check
- `model.py` → `GroupedQueryAttention.__init__()` → `self.n_kv_heads = cfg.n_kv_heads if self.use_gqa else cfg.n_head`
- Config: `n_head=12, n_kv_heads=4` → 3:1 GQA ratio
- When `USE_GQA=False`, it becomes standard MHA with 12 KV heads

---

## 5. Full Ablation Table (The Paper-Ready Artifact)

### How to Generate
```bash
python -m ablation.run_ablation --steps 100
```

### Output Location
- Console: formatted comparison table
- File: `logs/ablation/ablation_results.json`

### What the Table Proves
| Config | What It Proves |
|--------|---------------|
| Full Baseline | Architecture works correctly |
| No RMSNorm | Normalization prevents gradient explosion |
| No RoPE | Positional encoding enables word order |
| No Flash Attention | Hardware optimization (identical math) |
| Full MHA | GQA is a memory optimization |

---

## 6. Perplexity Verification

### Paper Claim
> "The model achieves a perplexity of X on the validation set after Y training steps."

### How to Verify
```bash
python -m evaluation.perplexity --checkpoint checkpoints/best_model.pt --batches 50
```

### Cross-Check
- PPL = exp(average_cross_entropy_loss) — calculated in `evaluation/perplexity.py`
- Must be calculated on the **validation set** (data the model never saw during training)
- Standard interpretation: PPL < 100 = usable, PPL < 30 = good

---

## 7. Training Metrics History

### How to Verify Training Was Stable
```bash
python -m tools.profiling_history --plot
```

### Output
- `logs/loss_curve.png` — visual proof of convergence
- `logs/vram_curve.png` — GPU memory stability
- Console: summary statistics (total steps, time, best loss)

---

## 8. Code-to-Paper Mapping

| Paper Section | Primary Source | Verification Tool |
|---------------|---------------|-------------------|
| Abstract | `manuals/01_project_overview/` | — |
| Background / Theory | `manuals/02_theoretical_foundations/` | — |
| Architecture | `manuals/03_system_architecture/model_architecture.md` | `project_report.py` |
| Implementation | `manuals/04_implementation_details/` | `py_compile` all files |
| Ablation Studies | `manuals/05_.../ablation_studies.md` | `notebooks/` + `ablation/run_ablation.py` |
| Evaluation | `manuals/04_.../evaluation_metrics.md` | `evaluation/perplexity.py` |
| Reproducibility | `manuals/06_reproducibility/` | `quick_start_guide.md` smoke test |
| References | `manuals/07_appendices/references.md` | BibTeX entries verified |
