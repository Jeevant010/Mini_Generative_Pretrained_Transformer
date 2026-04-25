"""
tools/profiling_history.py — Cumulative profiling tracker across training runs.

Reads the training metrics CSV and produces a history report showing:
- Training progress over time (across multiple resume sessions)
- Loss curves, LR schedule, tokens/sec, VRAM usage
- Comparison between different training runs

Usage:
    python -m tools.profiling_history
    python -m tools.profiling_history --csv logs/training_metrics.csv
    python -m tools.profiling_history --plot   # Generate matplotlib plots
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_metrics(csv_path):
    """Load training metrics from CSV."""
    if not os.path.exists(csv_path):
        print(f"❌ Metrics file not found: {csv_path}")
        return []

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                try:
                    if "." in str(row[key]):
                        row[key] = float(row[key])
                    else:
                        row[key] = int(row[key])
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def analyze_metrics(rows):
    """Compute summary statistics from training metrics."""
    if not rows:
        return None

    steps = [r.get("step", 0) for r in rows]
    losses = [r.get("loss", 0) for r in rows if r.get("loss")]
    val_losses = [r.get("val_loss") for r in rows if r.get("val_loss")]
    ppls = [r.get("perplexity") for r in rows if r.get("perplexity")]
    tok_sec = [r.get("tokens_per_sec", 0) for r in rows if r.get("tokens_per_sec")]
    vram = [r.get("vram_mb", 0) for r in rows if r.get("vram_mb")]

    summary = {
        "total_steps": max(steps) if steps else 0,
        "total_entries": len(rows),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "val_loss_min": min(val_losses) if val_losses else None,
        "ppl_min": min(ppls) if ppls else None,
        "avg_tok_sec": sum(tok_sec) / len(tok_sec) if tok_sec else None,
        "avg_vram_mb": sum(vram) / len(vram) if vram else None,
        "peak_vram_mb": max(vram) if vram else None,
    }
    return summary


def print_report(summary, csv_path):
    """Print a formatted profiling history report."""
    print(f"\n{'='*60}")
    print(f"📈 TRAINING PROFILING HISTORY")
    print(f"{'='*60}")
    print(f"Source: {csv_path}")

    if summary is None:
        print("No data found.")
        return

    print(f"\n{'Metric':<30} | {'Value':>15}")
    print("-" * 50)
    print(f"{'Total Steps':<30} | {summary['total_steps']:>15,}")
    print(f"{'Log Entries':<30} | {summary['total_entries']:>15,}")

    if summary["loss_start"] is not None:
        print(f"{'Loss (start → end)':<30} | {summary['loss_start']:>7.4f} → {summary['loss_end']:.4f}")
        print(f"{'Loss (best)':<30} | {summary['loss_min']:>15.4f}")

    if summary["val_loss_min"] is not None:
        print(f"{'Val Loss (best)':<30} | {summary['val_loss_min']:>15.4f}")

    if summary["ppl_min"] is not None:
        print(f"{'Perplexity (best)':<30} | {summary['ppl_min']:>15.2f}")

    if summary["avg_tok_sec"] is not None:
        print(f"{'Avg Tokens/sec':<30} | {summary['avg_tok_sec']:>15,.0f}")

    if summary["avg_vram_mb"] is not None:
        print(f"{'Avg VRAM (MB)':<30} | {summary['avg_vram_mb']:>15,.0f}")
        print(f"{'Peak VRAM (MB)':<30} | {summary['peak_vram_mb']:>15,.0f}")

    # Training time estimate
    if summary["avg_tok_sec"] and summary["total_steps"]:
        import config
        tokens_per_step = config.batch_size * config.block_size
        elapsed_tokens = summary["total_steps"] * tokens_per_step
        elapsed_hours = elapsed_tokens / summary["avg_tok_sec"] / 3600
        remaining_steps = config.max_iters - summary["total_steps"]
        remaining_hours = remaining_steps * tokens_per_step / summary["avg_tok_sec"] / 3600

        print(f"\n{'─'*50}")
        print(f"{'Est. Elapsed Time':<30} | {elapsed_hours:>12.1f} hrs")
        print(f"{'Est. Remaining Time':<30} | {remaining_hours:>12.1f} hrs")
        print(f"{'Est. Total Time':<30} | {elapsed_hours + remaining_hours:>12.1f} hrs")

    print(f"{'='*60}")


def generate_plots(rows, output_dir="logs"):
    """Generate matplotlib plots of training metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not installed. Skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    steps = [r.get("step", 0) for r in rows]
    losses = [r.get("loss") for r in rows]

    # Loss curve
    valid = [(s, l) for s, l in zip(steps, losses) if l is not None]
    if valid:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot([v[0] for v in valid], [v[1] for v in valid], linewidth=0.8, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, "loss_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"📊 Loss curve saved: {path}")

    # VRAM over time
    vram = [(r.get("step", 0), r.get("vram_mb")) for r in rows if r.get("vram_mb")]
    if vram:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot([v[0] for v in vram], [v[1] for v in vram], linewidth=0.8, color="orange")
        ax.set_xlabel("Step")
        ax.set_ylabel("VRAM (MB)")
        ax.set_title("GPU VRAM Usage Over Training")
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, "vram_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"📊 VRAM curve saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Training profiling history report.")
    parser.add_argument("--csv", type=str, default="logs/training_metrics.csv")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots.")
    args = parser.parse_args()

    rows = load_metrics(args.csv)
    summary = analyze_metrics(rows)
    print_report(summary, args.csv)

    if args.plot and rows:
        generate_plots(rows)


if __name__ == "__main__":
    main()
