# Mini Generative Pretrained Transformer (Production Edition)

A high-performance, production-ready implementation of a Small Language Model (SLM) using PyTorch. This project features a modern Transformer architecture optimized for local hardware (NVIDIA RTX 40-series) and can efficiently train on large datasets like OpenWebText.

## 🚀 Key Features

- **Modern Architecture**: Implements **GQA** (Grouped-Query Attention), **RoPE** (Rotary Positional Embeddings), **SwiGLU** activation, and **RMSNorm**.
- **Production Data Pipeline**: Streams local Parquet files, trains a high-speed BPE tokenizer (Rust-powered), and generates memory-mapped binary datasets for zero-latency training.
- **Hardware Optimized**: Native support for **`bfloat16`** mixed-precision training on NVIDIA Ada Lovelace GPUs (RTX 4060/4070/4080/4090).
- **Auto-Resume**: Robust checkpointing system that automatically saves progress and resumes from the latest state.
- **Modular Design**: Clean separation between model architecture, tokenizer, data loading, and training logic.

## 📁 Repository Structure

```text
Mini_Generative_Pretrained_Transformer/
├── config.py           # Centralized hyperparameters & hardware config
├── model.py            # Core Transformer architecture (GQA, RoPE, SwiGLU)
├── tokenizer.py        # High-performance BPE wrapper (HuggingFace tokenizers)
├── dataset.py          # Memory-mapped (np.memmap) data loading
├── prepare_data.py     # Data factory: Parquet -> Tokenization -> Binary Binaries
├── training.py         # Main production training loop with bfloat16 & Checkpoints
├── generate.py         # Inference script with auto-checkpoint detection
├── Research/           # Notebooks and research notes for iterative learning
└── checkpoints/        # Directory containing saved model states (.pt)
```

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
   ```

2. **Prepare the Dataset**:
   Ensure your local Parquet files are at the path specified in `config.py` (Default: `D:\Dataset`).
   ```bash
   python prepare_data.py
   ```
   *This trains the 32k tokenizer and creates `train.bin` and `val.bin`.*

## 📈 Training

To start or resume training:
```bash
python training.py
```
- Logs progress every 100 steps.
- Evaluates on Validation data every 500 steps.
- Performs a full checkpoint save every 2,500 steps.

### 🔄 Resetting / Starting From Scratch

If you want to wipe all progress and start training the model from step 0:
1. **Clear Checkpoints**: Delete the `checkpoints/` folder.
   ```bash
   rm -rf checkpoints/
   ```
2. **Clear Data (Optional)**: If you want to re-run the tokenizer training or use a different sample size, delete the binary files and the tokenizer JSON:
   ```bash
   rm train.bin val.bin bpe_tokenizer_32k.json
   ```
3. **Run Pipeline**: Run `python prepare_data.py` followed by `python training.py`.

## 💬 Text Generation

To generate text from your latest checkpoint:
```bash
python generate.py --prompt "The future of AI is" --max-tokens 100
```
- Automatically finds the highest-numbered checkpoint in `checkpoints/`.
- Uses top-k sampling and temperature control for high-quality output.

## 📝 Design Decisions

- **`uint16` Dataset**: Token IDs are stored as 16-bit integers to reduce disk footprint by 4x.
- **Memory Mapping**: Training reads directly from disk using `np.memmap`, allowing 100GB+ datasets to be trained on machines with low RAM.
- **Weight Tying**: Shares weights between the token embedding and the language modeling head to reduce parameter count.

## 📜 License
See `LICENSE` for details.
