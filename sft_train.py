"""Supervised fine-tuning entry point for the current Mini GPT checkpoint."""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

import config
from model import GPTLanguageModel
from sft_dataset import SFTDataset
from tokenizer import BytePairTokenizer


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def estimate_sft_loss(model: GPTLanguageModel, dataset: SFTDataset, batch_size: int, batches: int):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(batches):
            xb, yb = dataset.get_batch(split, batch_size)
            logits, _ = model(xb)
            losses.append(sft_loss(logits, yb).item())
        out[split] = sum(losses) / max(len(losses), 1)
    model.train()
    return out


def load_checkpoint(model: GPTLanguageModel, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint


def save_checkpoint(path: str, model, optimizer, step: int, epoch: int, val_loss: float):
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the base model on Dolly SFT data.")
    parser.add_argument("--data", default="data/sft/dolly_15k.json")
    parser.add_argument("--checkpoint", default="checkpoints/ckpt_step_130000.pt")
    parser.add_argument("--out-dir", default="checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Using device: {config.device}")

    tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)
    dataset = SFTDataset(args.data, tokenizer, max_length=config.block_size)
    print(
        f"SFT examples: {len(dataset.train_examples)} train, "
        f"{len(dataset.val_examples)} val"
    )

    model = GPTLanguageModel(config).to(config.device)
    base = load_checkpoint(model, args.checkpoint)
    print(f"Loaded base checkpoint: {args.checkpoint} (step {base.get('step', '?')})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(dataset.train_examples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    best_val_loss = float("inf")
    step = 0
    model.train()
    print(f"Starting SFT for up to {total_steps} steps")

    for epoch in range(args.epochs):
        for _ in range(steps_per_epoch):
            if args.max_steps is not None and step >= args.max_steps:
                break

            t0 = time.perf_counter()
            xb, yb = dataset.get_batch("train", args.batch_size)

            with torch.autocast(
                device_type="cuda" if "cuda" in str(config.device) else "cpu",
                dtype=torch.bfloat16,
            ):
                logits, _ = model(xb)
                loss = sft_loss(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            step += 1
            if step % 10 == 0:
                dt = time.perf_counter() - t0
                print(
                    f"step {step:5d}/{total_steps} | epoch {epoch + 1} | "
                    f"loss {loss.item():.4f} | {dt * 1000:.0f} ms"
                )

            if step % args.eval_interval == 0 or step == total_steps:
                losses = estimate_sft_loss(model, dataset, args.batch_size, args.eval_batches)
                print(
                    f">>> eval step {step}: train {losses['train']:.4f} | "
                    f"val {losses['val']:.4f}"
                )
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    best_path = os.path.join(args.out_dir, "best_sft_model.pt")
                    save_checkpoint(best_path, model, optimizer, step, epoch, best_val_loss)
                    print(f"saved best SFT checkpoint: {best_path}")

            if step % args.save_interval == 0:
                path = os.path.join(args.out_dir, f"sft_step_{step}.pt")
                save_checkpoint(path, model, optimizer, step, epoch, best_val_loss)
                print(f"saved checkpoint: {path}")

            if step >= total_steps:
                break

        if args.max_steps is not None and step >= args.max_steps:
            break

    final_path = os.path.join(args.out_dir, "last_sft_model.pt")
    save_checkpoint(final_path, model, optimizer, step, args.epochs, best_val_loss)
    print(f"SFT complete. Last checkpoint: {final_path}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
