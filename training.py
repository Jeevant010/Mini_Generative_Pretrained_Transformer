import os
import time
import math
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from dataset import get_batch
from tokenizer import BytePairTokenizer
from model import GPTLanguageModel # <--- Importing from the new modular model.py

# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / eval_iters
    model.train()
    return out

def train():
    os.makedirs("checkpoints", exist_ok=True)
    device = config.device
    print(f"Starting production training on {device}...")

    model = GPTLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Check for latest checkpoint
    start_step = 0
    ckpts = [f for f in os.listdir("checkpoints") if f.startswith("ckpt_step_")]
    if ckpts:
        latest = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        print(f"Loading checkpoint: {latest}")
        checkpoint = torch.load(os.path.join("checkpoints", latest), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1

    model.train()
    for step in range(start_step, config.max_iters):
        # Optimization Step
        xb, yb = get_batch("train")
        
        # Production Hardware Optimization: bfloat16
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Monitoring
        if step % 100 == 0:
            print(f"Step {step:5d} | Loss: {loss.item():.4f}")

        # Evaluation
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, config.eval_iters)
            print(f">>> EVAL Step {step:5d}: train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f}")

        # Checkpointing
        if step > 0 and step % config.checkpoint_interval == 0:
            ckpt_path = os.path.join("checkpoints", f"ckpt_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    print("✅ Training Complete.")

if __name__ == "__main__":
    train()
