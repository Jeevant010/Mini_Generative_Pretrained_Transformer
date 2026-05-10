# Chapter 5.6 — Our Hardware

## The GPU

| Property | Value |
|---|---|
| Model | NVIDIA GeForce RTX 4060 Laptop GPU |
| Architecture | Ada Lovelace (2023) |
| VRAM | 8 GB GDDR6 |
| CUDA cores | 3,072 |
| Tensor cores | 96 (4th gen) |
| bfloat16 performance | ~233 TFLOPS |
| Memory bandwidth | 256 GB/s |
| TDP | 115W (laptop) |

## Why This GPU Works

Our model needs about 2.5 GB of VRAM during training (model weights + optimizer + activations + gradients). The RTX 4060's 8 GB provides comfortable headroom.

The 4th-generation Tensor Cores provide hardware acceleration for bfloat16 matrix multiplication, which is the core operation during training.

## Training Performance

| Metric | Observed Value |
|---|---|
| Tokens per step | 7,680 |
| Steps per second | ~1-2 |
| Tokens per second | ~7,600-15,000 |
| Time per step | ~500-1000 ms |
| GPU utilization | ~85-95% |
| Peak VRAM usage | ~2.5-3.5 GB |
| Training time (150K steps) | ~2-3 days |

## Could You Use a Different GPU?

### Minimum Requirements

| GPU | VRAM | Will It Work? |
|---|---|---|
| GTX 1060 (6GB) | 6 GB | Yes, but no bfloat16 — need float16 with loss scaler |
| RTX 2060 (6GB) | 6 GB | Yes, tight on memory. Reduce batch size to 10. |
| RTX 3060 (12GB) | 12 GB | Yes, very comfortable. Can increase batch size. |
| RTX 4060 (8GB) | 8 GB | Yes — this is what we use |
| RTX 4090 (24GB) | 24 GB | Yes — could train a much larger model or bigger batch |
| Apple M1/M2 (MPS) | 8-16 GB | Partially — PyTorch MPS support is improving but not all ops work |
| CPU only | N/A | Technically possible but would take weeks instead of days |

### Scaling Up

If you had a better GPU:

| Change | Effect |
|---|---|
| More VRAM | Larger batch size → more stable gradients |
| More VRAM | Larger context window (512 or 1024 tokens) |
| More compute | Faster training → more experiments per day |
| Multiple GPUs | Data parallelism → linear speedup |

## The Full Machine

| Component | Specification |
|---|---|
| CPU | (varies — not the bottleneck for GPU training) |
| RAM | 16+ GB recommended (for data preparation) |
| Storage | SSD recommended (memmap reads are faster on SSD) |
| OS | Windows 11 |
| Python | 3.8+ |
| PyTorch | 2.0+ with CUDA |

## Key Takeaway

You do **not** need expensive cloud GPUs or a multi-GPU cluster to train a real language model. A single laptop GPU with 8 GB VRAM is sufficient for a 118M-parameter model. The important thing is using modern efficiency techniques (bfloat16, Flash Attention, GQA) to make the most of limited hardware.
