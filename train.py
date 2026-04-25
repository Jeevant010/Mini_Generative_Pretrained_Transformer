"""
train.py — Training script for the SLM.

Usage:
    python train.py
"""

import torch
import torch._dynamo

from config import (
    device, batch_size, block_size,
    max_iters, learning_rate, eval_iters,
    n_embd, n_head, n_layer, dropout,
)
from tokenizer import vocab_size
from dataset import get_batch
from model import GPTLanguageModel


# ─── Performance Flags ─────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True

print(f"Using device: {device}")
print(f"Vocabulary size: {vocab_size}")

# ─── Instantiate Model ─────────────────────────────────
model = GPTLanguageModel(vocab_size).to(device)

try:
    model = torch.compile(model)
    print("✅ Using torch.compile() optimized mode.")
except Exception as e:
    print(f"⚠️  torch.compile() failed ({e}); running in eager mode.")


# ─── Evaluation Helper ─────────────────────────────────
@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    """Estimate mean loss on train and val splits."""
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().detach().cpu().item()

    model.train()
    return out


# ─── Optimizer & AMP ───────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

# ─── Training Loop ─────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Training for {max_iters} iterations")
print(f"{'='*50}\n")

for iteration in range(max_iters):

    # Periodic evaluation
    if iteration % 100 == 0 or iteration == max_iters - 1:
        losses = estimate_loss()
        print(
            f"Step {iteration:5d} | "
            f"train loss {losses['train']:.3f} | "
            f"val loss {losses['val']:.3f}"
        )

    xb, yb = get_batch("train")

    # Mixed-precision forward + backward
    with torch.autocast(
        device_type=device, dtype=torch.float16, enabled=(device == "cuda")
    ):
        _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    # Gradient clipping for stability
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

print(f"\n✅ Training complete. Final loss: {loss.item():.4f}")

# ─── Save Checkpoint ───────────────────────────────────
torch.save(model.state_dict(), "model-01.pt")
print("💾 Model saved to model-01.pt")
