# Environment Setup

## Python Environment

Use a Python environment with PyTorch, CUDA support, and data-processing libraries.

Install dependencies:

```powershell
pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

The observed local report shows:

| Item | Value |
| --- | --- |
| Python | 3.13.5 |
| Platform | Windows |
| PyTorch | 2.11.0+cu128 |
| CUDA available | True |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |

## Required Packages

Important packages in `requirements.txt`:

- `numpy`
- `pandas`
- `pyarrow`
- `tqdm`
- `torch`
- `torchvision`
- `torchaudio`
- `transformers`
- `tokenizers`
- `datasets`
- `accelerate`
- `safetensors`
- `sentencepiece`

## Dataset Location

Set the dataset path in `config.py`:

```python
DATASET_PATH = r"D:\Openweb"
```

The directory should contain parquet shards.

## CUDA Check

Quick check:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Encoding Note

Some project files currently print Unicode symbols. On Windows, if you see a `UnicodeEncodeError`, run:

```powershell
$env:PYTHONIOENCODING='utf-8'
```

before the Python command.

## Disk Space

Plan for:

- 10 GB dataset binaries
- 1.3 GB per checkpoint
- multiple checkpoints under `checkpoints/`
- logs and samples

The current checkpoint directory is tens of GB. Clean old checkpoints carefully only after confirming they are not needed.

