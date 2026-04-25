# Environment Setup — Python, PyTorch, CUDA & Dependencies

## 1. Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11+ |
| OS | Windows 10, Linux | Windows 11 |
| GPU | CUDA-capable NVIDIA GPU | RTX 4060 (8 GB VRAM) |
| CUDA Toolkit | 12.1 | 12.4 |
| Disk Space | 10 GB (code + data) | 50+ GB (with large corpus) |
| RAM | 8 GB | 16+ GB |

---

## 2. Step-by-Step Setup

### 2.1 Create Python Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2.2 Install Dependencies

```powershell
pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

The `--extra-index-url` flag ensures PyTorch wheels with CUDA 12.4 support are downloaded.

### 2.3 Verify Installation

```powershell
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')"
```

Expected output:
```
PyTorch 2.x.x+cu124
CUDA: True
Device: NVIDIA GeForce RTX 4060
```

---

## 3. Dependencies (`requirements.txt`)

### Core Scientific Stack
| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.26.4 | Array operations, memmap |
| `pandas` | ≥2.2.2 | Data manipulation |
| `pyarrow` | ≥17.0.0 | Parquet file reading |
| `tqdm` | ≥4.66.0 | Progress bars |
| `matplotlib` | ≥3.8.0 | Plotting |

### Deep Learning Core
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.4.0, <3.0 | Model, training, inference |
| `torchvision` | ≥0.19.0 | Vision utilities |
| `torchaudio` | ≥2.4.0 | Audio utilities |

### NLP / LLM Ecosystem
| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | ≥4.45.0 | HuggingFace model utilities |
| `tokenizers` | ≥0.20.0 | BPE tokenizer (Rust backend) |
| `datasets` | ≥3.0.0 | Dataset utilities |
| `accelerate` | ≥0.34.0 | Training acceleration |
| `safetensors` | ≥0.4.4 | Safe model serialization |
| `sentencepiece` | ≥0.2.0 | SentencePiece tokenizer |

### Notebook Support
| Package | Version | Purpose |
|---------|---------|---------|
| `jupyter` | ≥1.1.1 | Notebook server |
| `ipykernel` | ≥6.29.5 | Jupyter kernel |
| `ipywidgets` | ≥8.1.3 | Interactive widgets |

---

## 4. CUDA Troubleshooting

| Symptom | Solution |
|---------|----------|
| `torch.cuda.is_available()` returns `False` | Reinstall PyTorch with CUDA index URL |
| Wrong CUDA version | Check `nvcc --version` and match PyTorch build |
| VS Code notebook uses CPU | Select correct kernel linked to CUDA environment |
| OOM errors | Reduce `batch_size` first, then `block_size` |

---

## 5. GPU Memory Management

```powershell
# Check GPU memory usage
nvidia-smi

# Monitor continuously
nvidia-smi -l 1
```

Expected memory usage during training: ~2–4 GB (out of 8 GB on RTX 4060).
