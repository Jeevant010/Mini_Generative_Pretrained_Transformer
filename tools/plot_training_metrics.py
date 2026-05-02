"""
Generate paper-ready plots from logs/training_metrics.csv.

Usage:
    python tools/plot_training_metrics.py
    python tools/plot_training_metrics.py --csv logs/training_metrics.csv --out paper_content/figures
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NUMERIC_COLUMNS = [
    "step",
    "loss",
    "lr",
    "tokens_per_sec",
    "tflops",
    "grad_norm",
    "vram_mb",
    "val_loss",
    "perplexity",
]


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return PROJECT_ROOT / path


def prepare_output_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = out_dir / ".write_test"

    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return out_dir
    except OSError:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_dir = out_dir.with_name(f"{out_dir.name}_{stamp}")
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            fallback_dir = Path.cwd() / f"{out_dir.name}_{stamp}"
            fallback_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory is not writable right now; using {fallback_dir}")
        return fallback_dir


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "step" not in df.columns:
        raise ValueError("Metrics CSV must contain a 'step' column.")

    return df.sort_values("step").reset_index(drop=True)


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
    except PermissionError:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = out_path.with_name(f"{out_path.stem}_{stamp}{out_path.suffix}")
        fig.savefig(fallback_path, dpi=220, bbox_inches="tight")
        out_path = fallback_path
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_loss_curves(step_rows: pd.DataFrame, eval_rows: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))

    train = step_rows.dropna(subset=["loss"])
    eval_loss = eval_rows.dropna(subset=["val_loss"])

    ax.plot(
        train["step"],
        rolling(train["loss"], 250),
        color="#2563eb",
        linewidth=1.8,
        label="Training loss (250-step rolling mean)",
    )
    ax.plot(
        eval_loss["step"],
        eval_loss["val_loss"],
        color="#dc2626",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="Validation loss",
    )

    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    save_plot(fig, out_dir / "loss_curves.png")


def plot_perplexity(eval_rows: pd.DataFrame, out_dir: Path) -> None:
    ppl = eval_rows.dropna(subset=["perplexity"])
    if ppl.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        ppl["step"],
        ppl["perplexity"],
        color="#7c3aed",
        marker="o",
        markersize=3.5,
        linewidth=1.8,
    )
    ax.set_yscale("log")
    ax.set_title("Validation Perplexity")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Perplexity (log scale)")
    ax.grid(True, which="both", alpha=0.25)

    save_plot(fig, out_dir / "validation_perplexity.png")


def plot_learning_rate(step_rows: pd.DataFrame, out_dir: Path) -> None:
    lr_rows = step_rows.dropna(subset=["lr"])
    if lr_rows.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(lr_rows["step"], lr_rows["lr"], color="#059669", linewidth=1.8)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.25)

    save_plot(fig, out_dir / "learning_rate_schedule.png")


def plot_throughput(step_rows: pd.DataFrame, out_dir: Path) -> None:
    throughput = step_rows.dropna(subset=["tokens_per_sec"])
    if throughput.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        throughput["step"],
        rolling(throughput["tokens_per_sec"], 250),
        color="#ea580c",
        linewidth=1.8,
    )
    ax.set_title("Training Throughput")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Tokens per second (250-step rolling mean)")
    ax.grid(True, alpha=0.25)

    save_plot(fig, out_dir / "training_throughput.png")


def plot_vram(step_rows: pd.DataFrame, out_dir: Path) -> None:
    vram = step_rows.dropna(subset=["vram_mb"])
    if vram.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        vram["step"],
        rolling(vram["vram_mb"], 250),
        color="#0891b2",
        linewidth=1.8,
    )
    ax.set_title("Peak CUDA Memory Allocation")
    ax.set_xlabel("Training step")
    ax.set_ylabel("VRAM allocated (MB, 250-step rolling mean)")
    ax.grid(True, alpha=0.25)

    save_plot(fig, out_dir / "vram_usage.png")


def plot_grad_norm(step_rows: pd.DataFrame, out_dir: Path) -> None:
    grad = step_rows.dropna(subset=["grad_norm"])
    if grad.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        grad["step"],
        rolling(grad["grad_norm"], 250),
        color="#be123c",
        linewidth=1.8,
    )
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient norm (250-step rolling mean)")
    ax.grid(True, alpha=0.25)

    save_plot(fig, out_dir / "gradient_norm.png")


def plot_summary_dashboard(step_rows: pd.DataFrame, eval_rows: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    train = step_rows.dropna(subset=["loss"])
    eval_loss = eval_rows.dropna(subset=["val_loss"])
    ppl = eval_rows.dropna(subset=["perplexity"])
    throughput = step_rows.dropna(subset=["tokens_per_sec"])
    vram = step_rows.dropna(subset=["vram_mb"])

    axes[0, 0].plot(train["step"], rolling(train["loss"], 250), color="#2563eb", linewidth=1.6)
    axes[0, 0].plot(eval_loss["step"], eval_loss["val_loss"], color="#dc2626", marker="o", markersize=3, linewidth=1.4)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_ylabel("Cross-entropy")

    axes[0, 1].plot(ppl["step"], ppl["perplexity"], color="#7c3aed", marker="o", markersize=3, linewidth=1.4)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Validation Perplexity")
    axes[0, 1].set_ylabel("PPL (log)")

    axes[1, 0].plot(throughput["step"], rolling(throughput["tokens_per_sec"], 250), color="#ea580c", linewidth=1.6)
    axes[1, 0].set_title("Throughput")
    axes[1, 0].set_ylabel("Tokens/sec")

    axes[1, 1].plot(vram["step"], rolling(vram["vram_mb"], 250), color="#0891b2", linewidth=1.6)
    axes[1, 1].set_title("VRAM")
    axes[1, 1].set_ylabel("MB")

    for ax in axes.flat:
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.25)

    save_plot(fig, out_dir / "training_dashboard.png")


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    eval_rows = df.dropna(subset=["val_loss"])
    step_rows = df.dropna(subset=["loss"])

    latest_eval = eval_rows.sort_values("step").tail(1)
    latest_step = step_rows.sort_values("step").tail(1)

    lines = ["# Training Figure Summary", ""]
    lines.append(f"- Source CSV: `logs/training_metrics.csv`")
    lines.append(f"- Total CSV rows: {len(df):,}")

    if not latest_step.empty:
        row = latest_step.iloc[0]
        lines.append(f"- Latest training step row: step {int(row['step']):,}, loss {row['loss']:.6f}")

    if not latest_eval.empty:
        row = latest_eval.iloc[0]
        lines.append(
            f"- Latest eval row: step {int(row['step']):,}, "
            f"val_loss {row['val_loss']:.6f}, perplexity {row['perplexity']:.2f}"
        )

    lines.extend([
        "",
        "## Generated Figures",
        "",
        "- `loss_curves.png`",
        "- `validation_perplexity.png`",
        "- `learning_rate_schedule.png`",
        "- `training_throughput.png`",
        "- `vram_usage.png`",
        "- `gradient_norm.png`",
        "- `training_dashboard.png`",
        "",
    ])

    summary_path = out_dir / "README.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Mini GPT training metrics.")
    parser.add_argument("--csv", type=Path, default=Path("logs/training_metrics.csv"))
    parser.add_argument("--out", type=Path, default=Path("paper_content/figures"))
    args = parser.parse_args()

    csv_path = resolve_project_path(args.csv)
    out_dir = prepare_output_dir(resolve_project_path(args.out))

    df = load_metrics(csv_path)

    step_rows = df[df["loss"].notna()].copy()
    eval_rows = df[df["val_loss"].notna()].copy()

    plot_loss_curves(step_rows, eval_rows, out_dir)
    plot_perplexity(eval_rows, out_dir)
    plot_learning_rate(step_rows, out_dir)
    plot_throughput(step_rows, out_dir)
    plot_vram(step_rows, out_dir)
    plot_grad_norm(step_rows, out_dir)
    plot_summary_dashboard(step_rows, eval_rows, out_dir)
    write_summary(df, out_dir)


if __name__ == "__main__":
    main()
