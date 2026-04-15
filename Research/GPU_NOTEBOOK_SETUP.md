# GPU Setup For All Training Notebooks

This project now has a unified dependency and runtime path for CPU and CUDA-based training.

Notebooks covered:

- Research/Tokenizer.ipynb
- Research/Embeddings.ipynb
- Research/Attention.ipynb
- Research/Full_Architecture.ipynb

## 1) Install dependencies

Use the updated requirements file and install with CUDA wheel index:

pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt

This keeps the same environment compatible with CPU fallback when CUDA is not available.

## 2) Verify environment

Quick check command:

python -c "import torch; print(torch.**version**); print('cuda', torch.cuda.is_available())"

Expected behavior:

- If CUDA is available, notebooks automatically use GPU.
- Otherwise, notebooks run on CPU without code changes.

## 3) Notebook behavior summary

Tokenizer notebook:

- Trains tokenizer with CPU-friendly logic.
- No GPU dependency required for correctness.

Embeddings notebook:

- Uses CUDA mixed precision automatically when available.
- Uses CPU-safe profile when CUDA is unavailable.

Attention notebook:

- Supports all attention variants.
- Uses profile-based scaling for CPU and RTX 4060 style settings.

Full architecture notebook:

- Unified train/load pipeline for tokenizer + embeddings + attention.
- Supports small-to-large staged curriculum.
- Supports checkpoint-resume for long runs.

## 4) Recommended device workflow

1. Start with CPU-safe profiles to validate pipeline and outputs.
2. Move to GPU profile once environment check shows CUDA available.
3. Keep checkpoint resume enabled for longer runs.

## 5) Common issues

If GPU is not used:

1. Confirm correct kernel/environment is selected in VS Code.
2. Reinstall torch stack using CUDA index command above.
3. Restart notebook kernel after reinstall.

If CUDA OOM appears:

1. Lower batch size.
2. Lower max sequence length.
3. Use smaller profile.
4. Increase gradient accumulation only if needed.
