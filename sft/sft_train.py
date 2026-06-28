"""Supervised Fine-Tuning training loop."""

import os
import sys
import time
import math
import torch
import torch.nn.functional as F

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from model import GPTLanguageModel
from sft_dataset import SFTDataset
from tokenizer import BytePairTokenizer


def sft_loss(logits, labels):
    """
    Cross-entropy loss with label masking.
    Labels of -100 are ignored (instruction tokens).
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=0.1,
    )


def train_sft():
    device = config.device
    print(f"Starting SFT on {device}...")

    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config.TOKENIZER_PATH))
    tokenizer = BytePairTokenizer.load(tokenizer_path)

    # Load base model from best pre-training checkpoint
    model = GPTLanguageModel(config).to(device)
    base_ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pt'))
    
    print(f"Loading base model from {base_ckpt_path}")
    base_ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    state_dict = base_ckpt["model_state_dict"] if "model_state_dict" in base_ckpt else base_ckpt
    model.load_state_dict(state_dict)
    
    step_val = base_ckpt.get('step', '?') if isinstance(base_ckpt, dict) else 'Unknown'
    print(f"Loaded base model from step {step_val}")

    # SFT hyperparameters
    sft_lr = 2e-5
    sft_epochs = 3
    sft_batch_size = 4
    eval_interval = 100
    save_interval = 500

    # Load SFT dataset
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'dolly_15k.json'))
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run download_sft_data.py first!")
        sys.exit(1)
        
    dataset = SFTDataset(data_path, tokenizer, max_length=config.block_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_lr, weight_decay=0.01)

    # Directories
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'sft'))
    os.makedirs(ckpt_dir, exist_ok=True)
    
    model.train()
    step = 0
    best_val_loss = float("inf")

    steps_per_epoch = len(dataset.train_examples) // sft_batch_size

    for epoch in range(sft_epochs):
        print(f"\n--- Epoch {epoch + 1}/{sft_epochs} ---")
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx in range(steps_per_epoch):
            t0 = time.perf_counter()

            xb, yb = dataset.get_batch("train", sft_batch_size)

            with torch.autocast(
                device_type="cuda" if "cuda" in str(device) else "cpu",
                dtype=torch.bfloat16,
            ):
                logits, _ = model(xb)
                loss = sft_loss(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dt = time.perf_counter() - t0
            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1

            if step % 10 == 0:
                print(f"  Step {step:4d} | Loss: {loss.item():.4f} | {dt*1000:.0f}ms/step")

            # Evaluation
            if step % eval_interval == 0:
                model.eval()
                val_losses = []
                for _ in range(20):
                    vx, vy = dataset.get_batch("val", sft_batch_size)
                    with torch.no_grad():
                        vlogits, _ = model(vx)
                        vloss = sft_loss(vlogits, vy)
                    val_losses.append(vloss.item())
                avg_val = sum(val_losses) / len(val_losses)
                print(f"  >>> Eval: val_loss = {avg_val:.4f}")

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save({
                        "step": step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val,
                    }, os.path.join(ckpt_dir, "best_sft_model.pt"))
                    print(f"  [OK] New best SFT model saved!")

                model.train()

            # Periodic save
            if step % save_interval == 0:
                torch.save({
                    "step": step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(ckpt_dir, f"sft_step_{step}.pt"))

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1} complete | Avg loss: {avg_epoch_loss:.4f}")

    print("\nSFT Training Complete!")
    print(f"Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_sft()
