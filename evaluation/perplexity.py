"""
evaluation/perplexity.py — Perplexity (PPL) calculation on the validation set.

Perplexity is the industry-standard metric for language models.
    PPL = exp(cross_entropy_loss)

Lower PPL = model is less "confused" = better language understanding.

Usage:
    python -m evaluation.perplexity
    python -m evaluation.perplexity --checkpoint checkpoints/best_model.pt
    python -m evaluation.perplexity --split val --batches 50
"""

import os
import math
import argparse
import torch

import config
from dataset import get_batch
from model import GPTLanguageModel


@torch.no_grad()
def calculate_perplexity(model, split="val", num_batches=None, device=None):
    """
    Calculate perplexity on the given data split.

    Args:
        model: GPTLanguageModel instance (already on device).
        split: "train" or "val".
        num_batches: Number of batches to average over. None = use config.eval_iters.
        device: Compute device.

    Returns:
        dict with "avg_loss", "perplexity", "num_batches", "tokens_evaluated".
    """
    if num_batches is None:
        num_batches = config.eval_iters
    if device is None:
        device = config.device

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for _ in range(num_batches):
        xb, yb = get_batch(split)
        logits, loss = model(xb, yb)
        batch_tokens = yb.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")  # Cap to avoid overflow

    return {
        "avg_loss": avg_loss,
        "perplexity": ppl,
        "num_batches": num_batches,
        "tokens_evaluated": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity on validation set.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batches", type=int, default=50, help="Number of batches to evaluate.")
    args = parser.parse_args()

    device = config.device
    print(f"Device: {device}")

    # Load model
    model = GPTLanguageModel(config).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        step = ckpt.get("step", "?")
        print(f"Loaded checkpoint: {args.checkpoint} (step {step})")
    else:
        # Find latest
        ckpt_dir = "checkpoints"
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_step_")]
            if ckpts:
                latest = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
                path = os.path.join(ckpt_dir, latest)
                ckpt = torch.load(path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"Loaded latest checkpoint: {path} (step {ckpt['step']})")

    # Calculate
    result = calculate_perplexity(model, split=args.split, num_batches=args.batches)

    print(f"\n{'='*50}")
    print(f"📊 PERPLEXITY REPORT ({args.split} split)")
    print(f"{'='*50}")
    print(f"Average Loss      : {result['avg_loss']:.4f}")
    print(f"Perplexity (PPL)   : {result['perplexity']:.2f}")
    print(f"Batches Evaluated  : {result['num_batches']}")
    print(f"Tokens Evaluated   : {result['tokens_evaluated']:,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
