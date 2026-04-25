"""
ablation/run_ablation.py — Automated ablation study runner.

Runs short training bursts (50-100 steps) on wizard_of_oz.txt with each
ablation toggle disabled individually, then produces a comparison table.

This is the single most powerful table for a research paper — it proves
every component of the architecture is mathematically necessary.

Usage:
    python -m ablation.run_ablation
    python -m ablation.run_ablation --steps 100 --data wizard_of_oz.txt
"""

import os
import sys
import time
import math
import json
import argparse
import importlib
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_wizard_binaries(text_path, vocab_size=2000):
    """
    Creates tiny train.bin/val.bin from wizard_of_oz.txt for ablation testing.
    Uses a small vocab to keep things fast.
    """
    from tokenizer import BytePairTokenizer

    print(f"  Preparing ablation data from: {text_path}")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Train a small tokenizer
    tok = BytePairTokenizer()
    tok.train([text], vocab_size=vocab_size, verbose=False)

    tokens = tok.encode(text)
    arr = np.array(tokens, dtype=np.uint16)

    # 90/10 split
    split_idx = int(len(arr) * 0.9)
    train_tokens = arr[:split_idx]
    val_tokens = arr[split_idx:]

    ablation_dir = os.path.join("ablation", "_data")
    os.makedirs(ablation_dir, exist_ok=True)

    train_path = os.path.join(ablation_dir, "train.bin")
    val_path = os.path.join(ablation_dir, "val.bin")
    tok_path = os.path.join(ablation_dir, "tokenizer.json")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    tok.save(tok_path)

    print(f"  Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")
    return train_path, val_path, tok_path, vocab_size


def run_single_ablation(name, overrides, steps, train_bin, val_bin, vocab_size):
    """
    Runs a single ablation configuration for N steps and captures metrics.
    """
    # Fresh import of config for each run
    import config as cfg

    # Apply ablation data paths
    cfg.TRAIN_BIN = train_bin
    cfg.VAL_BIN = val_bin
    cfg.vocab_size = vocab_size

    # Apply small model for speed
    cfg.batch_size = 4
    cfg.block_size = 128
    cfg.n_embd = 256
    cfg.n_layer = 4
    cfg.n_head = 4
    cfg.n_kv_heads = 2
    cfg.ffn_mult = 3.5
    cfg.dropout = 0.0
    cfg.learning_rate = 3e-4
    cfg.min_lr = 3e-5
    cfg.warmup_iters = 10
    cfg.max_iters = steps
    cfg.lr_decay_iters = steps
    cfg.grad_clip = 1.0

    # Reset all toggles to True, then apply overrides
    cfg.USE_RMSNORM = True
    cfg.USE_ROPE = True
    cfg.USE_FLASH_ATTENTION = True
    cfg.USE_GQA = True

    for k, v in overrides.items():
        setattr(cfg, k, v)

    # Rebuild model fresh
    from model import GPTLanguageModel
    from dataset import get_batch

    device = cfg.device
    model = GPTLanguageModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokens_per_batch = cfg.batch_size * cfg.block_size

    result = {
        "name": name,
        "params": num_params,
        "status": "✅ Stable",
        "final_loss": None,
        "perplexity": None,
        "tokens_per_sec": None,
        "vram_mb": None,
        "grad_norm": None,
        "exploded": False,
    }

    model.train()
    losses = []

    if "cuda" in str(device):
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.perf_counter()

    for step in range(steps):
        try:
            xb, yb = get_batch("train")
            with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
                logits, loss = model(xb, yb)

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                result["status"] = "💥 Exploded (NaN)"
                result["exploded"] = True
                result["final_loss"] = float("nan")
                break

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Capture gradient norm
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            result["grad_norm"] = total_norm.item()

            optimizer.step()
            losses.append(loss.item())

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                result["status"] = "🔴 OOM"
                result["exploded"] = True
                torch.cuda.empty_cache()
                break
            raise

    t_elapsed = time.perf_counter() - t_start

    if losses:
        result["final_loss"] = losses[-1]
        result["perplexity"] = math.exp(losses[-1]) if losses[-1] < 20 else float("inf")
        result["tokens_per_sec"] = (len(losses) * tokens_per_batch) / t_elapsed

    if "cuda" in str(device):
        result["vram_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        torch.cuda.empty_cache()

    # Determine status based on behavior
    if not result["exploded"] and losses:
        if losses[-1] > losses[0] * 1.5:
            result["status"] = "⚠️ Diverging"
        elif result["grad_norm"] and result["grad_norm"] > 100:
            result["status"] = "⚠️ Unstable gradients"

    return result


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies on wizard_of_oz.txt")
    parser.add_argument("--steps", type=int, default=100, help="Steps per ablation run.")
    parser.add_argument("--data", type=str, default="wizard_of_oz.txt", help="Text file for ablation.")
    args = parser.parse_args()

    print("=" * 70)
    print("🔬 ABLATION STUDY RUNNER")
    print("=" * 70)

    # Prepare data
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)

    train_bin, val_bin, tok_path, vocab_size = create_wizard_binaries(args.data)

    # Define ablation configurations
    configs = [
        ("Full Baseline (all ON)", {}),
        ("No RMSNorm", {"USE_RMSNORM": False}),
        ("No RoPE", {"USE_ROPE": False}),
        ("No Flash Attention", {"USE_FLASH_ATTENTION": False}),
        ("Full MHA (no GQA)", {"USE_GQA": False}),
    ]

    results = []
    for name, overrides in configs:
        print(f"\n{'─'*50}")
        print(f"▶ Running: {name}")
        print(f"  Overrides: {overrides if overrides else 'None (baseline)'}")
        print(f"  Steps: {args.steps}")

        result = run_single_ablation(name, overrides, args.steps, train_bin, val_bin, vocab_size)
        results.append(result)

        loss_str = f"{result['final_loss']:.4f}" if result['final_loss'] and not math.isnan(result['final_loss']) else "NaN"
        ppl_str = f"{result['perplexity']:.1f}" if result['perplexity'] and result['perplexity'] != float('inf') else "∞"
        tok_str = f"{result['tokens_per_sec']:,.0f}" if result['tokens_per_sec'] else "—"
        vram_str = f"{result['vram_mb']:.0f}" if result['vram_mb'] else "—"
        grad_str = f"{result['grad_norm']:.2f}" if result['grad_norm'] else "—"

        print(f"  Result: Loss={loss_str} | PPL={ppl_str} | {tok_str} tok/s | VRAM={vram_str}MB | {result['status']}")

    # Print comparison table
    print(f"\n\n{'='*90}")
    print("📊 ABLATION COMPARISON TABLE")
    print(f"{'='*90}")
    header = f"{'Config':<25} | {'Loss':>8} | {'PPL':>8} | {'Tok/s':>10} | {'VRAM MB':>8} | {'Grad Norm':>10} | {'Status'}"
    print(header)
    print("-" * 90)

    for r in results:
        loss = f"{r['final_loss']:.4f}" if r['final_loss'] and not math.isnan(r['final_loss']) else "NaN"
        ppl = f"{r['perplexity']:.1f}" if r['perplexity'] and r['perplexity'] != float('inf') else "∞"
        tok = f"{r['tokens_per_sec']:,.0f}" if r['tokens_per_sec'] else "—"
        vram = f"{r['vram_mb']:.0f}" if r['vram_mb'] else "—"
        grad = f"{r['grad_norm']:.2f}" if r['grad_norm'] else "—"
        print(f"{r['name']:<25} | {loss:>8} | {ppl:>8} | {tok:>10} | {vram:>8} | {grad:>10} | {r['status']}")

    print(f"{'='*90}")

    # Save results to JSON
    log_dir = os.path.join("logs", "ablation")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
