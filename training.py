"""
training.py — Production training loop with integrated evaluation suite.

Features:
    - Cosine LR with warmup
    - Mixed-precision (bfloat16)
    - Gradient clipping & norm tracking
    - Periodic evaluation with Perplexity (PPL)
    - Sample text generation at eval intervals
    - VRAM monitoring (CUDA)
    - CSV metrics logging (logs/training_metrics.csv)
    - Checkpoint resume (automatic)
    - PyTorch profiler integration
"""

import os
import csv
import time
import math
import torch
from torch.profiler import profile, ProfilerActivity

import config
from dataset import get_batch
from model import GPTLanguageModel  # <--- Importing from the new modular model.py

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation & Logging Helpers
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


def get_lr(step):
    warmup_iters = getattr(config, "warmup_iters", 0)
    lr_decay_iters = getattr(config, "lr_decay_iters", config.max_iters)
    min_lr = getattr(config, "min_lr", config.learning_rate)

    if warmup_iters > 0 and step < warmup_iters:
        return config.learning_rate * (step + 1) / warmup_iters

    if step > lr_decay_iters:
        return min_lr

    if lr_decay_iters <= warmup_iters:
        return config.learning_rate

    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (config.learning_rate - min_lr)


def validate_training_setup():
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if config.block_size <= 1:
        raise ValueError("block_size must be greater than 1.")
    if config.max_iters <= 0:
        raise ValueError("max_iters must be greater than 0.")
    if config.eval_iters <= 0:
        raise ValueError("eval_iters must be greater than 0.")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be greater than 0.")
    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be greater than 0.")
    if getattr(config, "warmup_iters", 0) < 0:
        raise ValueError("warmup_iters must be >= 0.")
    if getattr(config, "lr_decay_iters", config.max_iters) <= 0:
        raise ValueError("lr_decay_iters must be greater than 0.")
    if getattr(config, "min_lr", config.learning_rate) < 0:
        raise ValueError("min_lr must be >= 0.")
    if getattr(config, "TRAIN_LOG_INTERVAL", 100) <= 0:
        raise ValueError("TRAIN_LOG_INTERVAL must be greater than 0.")

    for split_name, path in [("train", config.TRAIN_BIN), ("val", config.VAL_BIN)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split_name} binary file not found: {path}")
        token_count = os.path.getsize(path) // 2  # uint16 tokens
        if token_count <= config.block_size + 1:
            raise ValueError(
                f"{split_name} split is too small for block_size={config.block_size}: "
                f"found {token_count} tokens in {path}."
            )


def init_csv_logger():
    """Initialize CSV metrics logger."""
    log_dir = getattr(config, "LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "training_metrics.csv")
    file_exists = os.path.exists(csv_path)

    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow([
            "timestamp", "step", "loss", "lr",
            "tokens_per_sec", "tflops", "grad_norm",
            "vram_mb", "val_loss", "perplexity",
        ])
        csv_file.flush()

    return csv_file, writer


def load_tokenizer_safe():
    """Attempt to load the tokenizer for sample generation."""
    try:
        from tokenizer import BytePairTokenizer
        tok_path = getattr(config, "TOKENIZER_PATH", "bpe_tokenizer_32k.json")
        if os.path.exists(tok_path):
            return BytePairTokenizer.load(tok_path)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    validate_training_setup()
    os.makedirs("checkpoints", exist_ok=True)
    device = config.device
    train_start_time = time.perf_counter()
    log_interval = getattr(config, "TRAIN_LOG_INTERVAL", 100)
    print(f"Starting production training on {device}...")

    # Print ablation status
    print(f"Ablation: RMSNorm={getattr(config, 'USE_RMSNORM', True)} | "
          f"RoPE={getattr(config, 'USE_ROPE', True)} | "
          f"FlashAttn={getattr(config, 'USE_FLASH_ATTENTION', True)} | "
          f"GQA={getattr(config, 'USE_GQA', True)}")

    model = GPTLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # --- Performance Metric Constants ---
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops_per_token = 6 * num_params # 2 (fwd) + 4 (bwd) estimate
    tokens_per_batch = config.batch_size * config.block_size
    flops_per_step = flops_per_token * tokens_per_batch
    
    print(f"Model Parameters: {num_params/1e6:.2f}M")
    
    # --- CSV Logger ---
    csv_file, csv_writer = None, None
    if getattr(config, "LOG_METRICS_CSV", False):
        csv_file, csv_writer = init_csv_logger()
        print(f"CSV logging: logs/training_metrics.csv")

    # --- Tokenizer for sample generation ---
    tokenizer = None
    if getattr(config, "GENERATE_SAMPLES", False):
        tokenizer = load_tokenizer_safe()
        if tokenizer:
            print(f"Sample generation: enabled ({len(config.SAMPLE_PROMPTS)} prompts)")
        else:
            print("Sample generation: disabled (tokenizer not found)")

    # Check for latest checkpoint
    start_step = 0
    best_val_loss = float("inf")
    ckpts = [f for f in os.listdir("checkpoints") if f.startswith("ckpt_step_")]
    if ckpts:
        latest = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        print(f"Loading checkpoint: {latest}")
        checkpoint = torch.load(os.path.join("checkpoints", latest), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    model.train()
    timer_target_iteration = getattr(config, "TIMER_TARGET_ITERATION", None)
    effective_timer_iteration = timer_target_iteration
    if timer_target_iteration is None:
        print("Iteration timer: disabled (TIMER_TARGET_ITERATION=None)")
    else:
        if timer_target_iteration < start_step:
            effective_timer_iteration = start_step
            print(
                f"Iteration timer: requested step {timer_target_iteration}, "
                f"but training resumes at step {start_step}. "
                f"Timer will run at step {effective_timer_iteration}."
            )
        else:
            print(f"Iteration timer: enabled for step {effective_timer_iteration}.")
    
    # --- Profiler Setup ---
    prof = None
    if config.ENABLE_PROFILING:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=8, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            with_stack=True
        )
        prof.start()

    for step in range(start_step, config.max_iters):
        t0 = time.perf_counter()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        timer_enabled_for_step = (
            effective_timer_iteration is not None and step == effective_timer_iteration
        )
        if timer_enabled_for_step:
            iteration_start_wall = time.time()
            t_data0 = time.perf_counter()
        
        # Optimization Step
        xb, yb = get_batch("train")
        if timer_enabled_for_step:
            t_data1 = time.perf_counter()
        
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
            if timer_enabled_for_step:
                t_fwd0 = time.perf_counter()
            logits, loss = model(xb, yb)
            if timer_enabled_for_step:
                t_fwd1 = time.perf_counter()
        
        if timer_enabled_for_step:
            t_bwd0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # --- Gradient Norm Tracking ---
        grad_norm = None
        if getattr(config, "grad_clip", 0.0) and config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
        elif getattr(config, "LOG_GRAD_NORM", False):
            # Calculate norm without clipping
            grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

        if timer_enabled_for_step:
            t_bwd1 = time.perf_counter()

        if timer_enabled_for_step:
            t_opt0 = time.perf_counter()
        optimizer.step()
        if timer_enabled_for_step:
            t_opt1 = time.perf_counter()
        
        if prof: prof.step()
        
        t1 = time.perf_counter()
        dt = t1 - t0 # seconds per step

        if timer_enabled_for_step:
            iteration_end_wall = time.time()
            data_ms = (t_data1 - t_data0) * 1000
            fwd_ms = (t_fwd1 - t_fwd0) * 1000
            bwd_ms = (t_bwd1 - t_bwd0) * 1000
            opt_ms = (t_opt1 - t_opt0) * 1000
            step_ms = dt * 1000
            print("\n" + "=" * 50)
            print(f"⏱ ITERATION TIMER (step {step})")
            print("=" * 50)
            print(f"\n[Step {step}] Start time: {iteration_start_wall:.4f} s")
            print(f"End time           : {iteration_end_wall:.4f} s")
            print(f"Data load          : {data_ms:.2f} ms")
            print(f"Forward pass       : {fwd_ms:.2f} ms")
            print(f"Backward pass      : {bwd_ms:.2f} ms")
            print(f"Optimizer step     : {opt_ms:.2f} ms")
            print(f"Full step          : {step_ms:.2f} ms")
            print("=" * 50 + "\n")

        # --- VRAM Tracking ---
        vram_mb = None
        if getattr(config, "LOG_VRAM", False) and "cuda" in str(device):
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        # Monitoring
        if step % log_interval == 0:
            tokens_per_sec = tokens_per_batch / dt
            tflops = flops_per_step / dt / 1e12
            steps_done = step - start_step + 1
            elapsed = time.perf_counter() - train_start_time
            avg_step_time = elapsed / max(steps_done, 1)
            remaining_steps = max(config.max_iters - step - 1, 0)
            eta_seconds = remaining_steps * avg_step_time
            progress_pct = ((step + 1) / config.max_iters) * 100.0
            eta_h = int(eta_seconds // 3600)
            eta_m = int((eta_seconds % 3600) // 60)
            elapsed_h = int(elapsed // 3600)
            elapsed_m = int((elapsed % 3600) // 60)

            # Build log line
            log_parts = [
                f"Step {step:5d}",
                f"{progress_pct:6.2f}%",
                f"Loss: {loss.item():.4f}",
                f"LR: {lr:.6e}",
                f"{tokens_per_sec:,.0f} tok/s",
                f"{tflops:.2f} TFLOPS",
                f"Elapsed: {elapsed_h:02d}h{elapsed_m:02d}m",
                f"ETA: {eta_h:02d}h{eta_m:02d}m",
            ]
            if grad_norm is not None and getattr(config, "LOG_GRAD_NORM", False):
                log_parts.append(f"GradNorm: {grad_norm:.2f}")
            if vram_mb is not None:
                log_parts.append(f"VRAM: {vram_mb:.0f}MB")

            print(" | ".join(log_parts))

            # CSV logging
            if csv_writer:
                csv_writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    step, f"{loss.item():.6f}", f"{lr:.8e}",
                    f"{tokens_per_sec:.0f}", f"{tflops:.4f}",
                    f"{grad_norm:.4f}" if grad_norm else "",
                    f"{vram_mb:.0f}" if vram_mb else "",
                    "", "",  # val_loss and perplexity filled at eval
                ])
                csv_file.flush()

        # Handle profiling results
        if config.ENABLE_PROFILING and step == config.PROFILING_WINDOW[1]:
            prof.stop()
            print("\n" + "="*50)
            print("🚀 HARDWARE PROFILING REPORT")
            print("="*50)
            
            # Detailed performance table (Simple & Readable)
            print("\n" + "-"*20 + " TOP 10 OPERATORS (by CUDA time) " + "-"*20)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print("-" * 75)
            
            print(f"{'Metric':<25} | {'Value':<15}")
            print("-" * 45)
            print(f"{'Tokens / Second':<25} | {tokens_per_batch/dt:,.0f}")
            print(f"{'TFLOPS (Peak Approx)':<25} | {flops_per_step/dt/1e12:.2f}")
            print(f"{'Step Latency':<25} | {dt*1000:.2f} ms")
            
            # Export Chrome Trace
            trace_path = "performance_trace.json"
            prof.export_chrome_trace(trace_path)
            print("="*50)
            print(f"✅ Profiling complete. Chrome Trace saved to: {trace_path}")
            print("Open chrome://tracing and upload this file to see the timeline.")
            print("="*50 + "\n")
            prof = None # Stop profiling after window

        # Evaluation
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, config.eval_iters)
            train_loss = losses["train"]
            val_loss = losses["val"]
            ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

            print(f">>> EVAL Step {step:5d}: train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | PPL {ppl:.2f}")

            # CSV log for eval
            if csv_writer:
                csv_writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    step, f"{train_loss:.6f}", f"{lr:.8e}",
                    "", "", "", "",
                    f"{val_loss:.6f}", f"{ppl:.2f}",
                ])
                csv_file.flush()

            # --- Sample Generation ---
            if tokenizer and getattr(config, "GENERATE_SAMPLES", False):
                try:
                    from evaluation.sample_generator import generate_and_log_samples
                    samples = generate_and_log_samples(model, tokenizer, step, config)
                    print(f"  📝 Samples saved to: logs/samples/step_{step}.txt")
                    # Print first sample preview
                    if samples:
                        prompt, text = samples[0]
                        preview = text[:120] + "..." if len(text) > 120 else text
                        print(f"  Preview: \"{preview}\"")
                except Exception as e:
                    print(f"  ⚠ Sample generation failed: {e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join("checkpoints", "best_model.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'best_val_loss': best_val_loss,
                }, best_ckpt_path)
                print(f"New best model saved: {best_ckpt_path} | best_val_loss: {best_val_loss:.4f}")

        # Checkpointing
        if step > 0 and step % config.checkpoint_interval == 0:
            ckpt_path = os.path.join("checkpoints", f"ckpt_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'best_val_loss': best_val_loss,
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    # Cleanup
    if csv_file:
        csv_file.close()

    print("✅ Training Complete.")

if __name__ == "__main__":
    train()
