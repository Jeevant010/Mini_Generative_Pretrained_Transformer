# Hardware Profiling

## Hardware Context

The observed machine uses:

| Component | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Framework | PyTorch 2.11.0+cu128 |
| Device mode | CUDA |

The project is designed around consumer-GPU constraints.

## Runtime Metrics

The training loop logs:

- tokens per second
- estimated TFLOPS
- gradient norm
- peak VRAM
- step time-derived ETA

These values are recorded in:

```text
logs/training_metrics.csv
```

## Tokens Per Second

For a step duration `dt`:

$$
\operatorname{tokens/sec} = \frac{B T}{dt}
$$

For the active preset:

$$
BT = 20 \times 384 = 7680
$$

## Estimated TFLOPS

The code estimates:

$$
F_{token} \approx 6N_{params}
$$

For 117,787,392 parameters:

$$
F_{token} \approx 706{,}724{,}352
$$

Per step:

$$
F_{step} \approx 706{,}724{,}352 \times 7680
\approx 5.43 \times 10^{12}
$$

Then:

$$
\operatorname{TFLOPS} = \frac{F_{step}}{dt \times 10^{12}}
$$

This is useful for comparing runs, but it is still an approximation.

## VRAM Measurement

Peak VRAM is measured with:

```python
torch.cuda.max_memory_allocated(device)
```

Use VRAM logs to compare:

- GQA vs full MHA
- Flash Attention vs manual attention
- different block sizes
- different batch sizes

## Profiler

Profiling is controlled in `config.py`:

```python
ENABLE_PROFILING = False
PROFILING_WINDOW = (100, 110)
```

When enabled, the script uses `torch.profiler.profile` and writes traces under:

```text
log/profiler/
```

The Chrome trace can be opened in:

```text
chrome://tracing
```

## Bottlenecks To Watch

Common bottlenecks:

- attention memory at large `block_size`
- low GPU utilization from slow data loading
- high VRAM from full MHA
- checkpoint disk pressure
- thermal throttling on laptop GPUs

## Current Observation

The latest log rows around step 60,000 show VRAM around 9.28 GB allocated. This is high for an 8 GB-class laptop GPU label and may reflect reported allocation behavior, shared memory behavior, or the exact installed GPU configuration. The practical conclusion is that the current configuration is near the memory boundary and should be changed carefully.

