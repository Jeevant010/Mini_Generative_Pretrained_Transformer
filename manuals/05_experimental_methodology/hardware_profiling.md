# Hardware Profiling — TFLOPS Measurement & Trace Analysis

## 1. Overview

This project includes two profiling tools:
1. **Inline profiler** in `training.py` — PyTorch profiler integration during training.
2. **Trace analyzer** in `profiler_quickview.py` — Post-hoc Chrome trace summarizer.

---

## 2. TFLOPS Estimation

### 2.1 Formula

```python
flops_per_token = 6 * num_params  # 2 (fwd) + 4 (bwd)
flops_per_step = flops_per_token * batch_size * block_size
tflops = flops_per_step / step_time_seconds / 1e12
```

### 2.2 Expected Values (RTX 4060)

| Metric | Approximate Value |
|--------|-------------------|
| Theoretical peak (bf16) | ~15 TFLOPS |
| Typical training TFLOPS | 0.3–1.5 TFLOPS |
| Utilization | 2–10% |

Low utilization is expected for small models — the GPU is bottlenecked by memory bandwidth and kernel launch overhead rather than compute.

---

## 3. PyTorch Profiler Configuration

```python
ENABLE_PROFILING = True
PROFILING_WINDOW = (100, 110)  # Steps 100–110
```

Profiler captures CPU and CUDA activity, records tensor shapes and call stacks, and exports to TensorBoard-compatible traces.

---

## 4. Profiler Quick View Tool

`profiler_quickview.py` parses Chrome trace JSON files and produces:

- **Category time totals**: CPU ops, kernels, CUDA runtime, memcpy
- **ProfilerStep timing**: avg, min, p50, p95, max
- **Top N CPU ops** (by total time)
- **Top N GPU kernels** (by CUDA time)
- **Quick takeaways**: Automated performance tips

### Usage

```bash
python profiler_quickview.py                    # Latest trace
python profiler_quickview.py --latest           # Explicit latest
python profiler_quickview.py --file path.json   # Specific file
python profiler_quickview.py --top 15           # Top 15 entries
python profiler_quickview.py --json             # Machine-readable output
```

---

## 5. Iteration Timer

For single-step diagnostics, set `TIMER_TARGET_ITERATION` in `config.py`:

```python
TIMER_TARGET_ITERATION = 250  # Detailed timing for step 250
```

Reports: data load, forward pass, backward pass, optimizer step, and total wall-clock time in milliseconds.

---

## 6. Trace Visualization

- **Chrome**: `chrome://tracing` → Load → upload trace JSON
- **Perfetto**: `ui.perfetto.dev` → Open trace file
- **TensorBoard**: `tensorboard --logdir=log/profiler`
