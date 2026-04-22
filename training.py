import os
import time
import math
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from matplotlib.pyplot import step
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

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
    
    # --- Performance Metric Constants ---
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops_per_token = 6 * num_params # 2 (fwd) + 4 (bwd) estimate
    tokens_per_batch = config.batch_size * config.block_size
    flops_per_step = flops_per_token * tokens_per_batch
    
    print(f"Model Parameters: {num_params/1e6:.2f}M")
    
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
        timer_enabled_for_step = (
            effective_timer_iteration is not None and step == effective_timer_iteration
        )
        if timer_enabled_for_step:
            iteration_start_wall = time.time()
            # print(f"\n[Step {step}] Start time: {iteration_start_wall:.4f} s")

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

        # Monitoring
        if step % 100 == 0:
            tokens_per_sec = tokens_per_batch / dt
            tflops = flops_per_step / dt / 1e12
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | {tokens_per_sec:,.0f} tok/s | {tflops:.2f} TFLOPS")

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
