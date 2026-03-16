"""
generate.py — Text generation / inference script for the SLM.

Usage:
    python generate.py
    python generate.py --prompt "Once upon a time"
    python generate.py --prompt "Hello world" --max-tokens 200
"""

import argparse

import torch

from config import device
from tokenizer import vocab_size, encode, decode
from model import GPTLanguageModel


def main():
    parser = argparse.ArgumentParser(description="Generate text with the trained SLM.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello! Where the hell are you?",
        help="Seed prompt for generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model-01.pt",
        help="Path to the model checkpoint file.",
    )
    args = parser.parse_args()

    # ─── Load Model ────────────────────────────────────
    print(f"Using device: {device}")
    model = GPTLanguageModel(vocab_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    # ─── Generate ──────────────────────────────────────
    context = torch.tensor(encode(args.prompt), dtype=torch.long, device=device)
    context = context.unsqueeze(0)  # (1, T)

    output_ids = model.generate(context, max_new_tokens=args.max_tokens)
    generated_text = decode(output_ids[0].tolist())

    print(f"\n{'─'*50}")
    print(generated_text)
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
