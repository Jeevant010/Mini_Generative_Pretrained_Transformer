# Presentation Content: Mini Generative Pretrained Transformer

## Slide 1: Title

### Mini GPT: Building a Decoder-Only Language Model From Scratch

- Project: Mini Generative Pretrained Transformer
- Built with PyTorch
- Trained on local OpenWebText-style parquet data
- Target hardware: single NVIDIA RTX 4060-class GPU
- Focus: modern GPT architecture, scalable data pipeline, reproducible training

Presenter notes:

This project is an end-to-end implementation of a compact GPT-style language model. The goal was not only to train a model, but to understand and build the core systems behind modern LLMs: tokenization, data preparation, transformer architecture, training, checkpointing, evaluation, and text generation.

---

## Slide 2: Motivation

### Why Build a Mini GPT?

- Large language models are powerful, but their internals are often hidden behind proprietary systems.
- Building a smaller model from scratch makes the mechanics understandable.
- The project tests whether modern transformer ideas can be implemented and trained on consumer hardware.
- It also creates a reusable research pipeline for future experiments.

Presenter notes:

The motivation was to move beyond using existing LLM APIs and actually understand how the model works internally. A small GPT lets us study the same concepts used in larger models, but at a scale that can be debugged, profiled, and explained.

---

## Slide 3: Project Objectives

### Main Goals

- Implement a byte-level BPE tokenizer with a 32,000-token vocabulary.
- Build a decoder-only transformer from modular PyTorch components.
- Use modern architecture choices: GQA, RoPE, RMSNorm, SwiGLU, Flash Attention, and weight tying.
- Create a streaming data pipeline that works with large parquet datasets.
- Train with automatic checkpointing, resume support, evaluation, and sample generation.
- Track model quality using loss, validation loss, perplexity, and generated text.

Presenter notes:

The project is both educational and practical. Each component has a clear purpose, and the system is structured so that training can resume safely, results can be monitored, and architecture choices can be tested through ablations.

---

## Slide 4: System Overview

### End-to-End Pipeline

```text
Raw Parquet Shards
        |
        v
Streaming Text Extraction
        |
        v
BPE Tokenization
        |
        v
train.bin / val.bin
        |
        v
Memory-Mapped Batch Loader
        |
        v
GPT Model Training
        |
        v
Checkpoints + Generated Samples
```

Presenter notes:

The raw data is not loaded fully into RAM. It is streamed from parquet files, tokenized, and written into binary files. During training, batches are sampled from memory-mapped files, so the model only moves the current batch to the GPU.

---

## Slide 5: Repository Structure

### Main Files

- `config.py`: hyperparameters, architecture settings, presets, and ablation toggles
- `prepare_data.py`: parquet-to-tokenized-binary preprocessing
- `dataset.py`: memory-mapped batch sampling
- `tokenizer.py`: BPE tokenizer wrapper
- `model.py`: decoder-only GPT model
- `training.py`: training loop, evaluation, checkpointing, logging
- `generate.py`: checkpoint-based text generation
- `ablation/run_ablation.py`: architecture ablation runner
- `manuals/`: project documentation and theory notes

Presenter notes:

The codebase is intentionally modular. Instead of one large notebook or script, each file has a single responsibility. That makes the project easier to test, explain, and extend.

---

## Slide 6: Data Pipeline

### Scalable Preprocessing

- Input corpus: local OpenWebText-style parquet shards.
- Tokenizer: byte-level BPE with 32,000 vocabulary entries.
- Split: 95 percent training, 5 percent validation.
- Storage: flat `uint16` token streams in `train.bin` and `val.bin`.
- Loading: `np.memmap` keeps dataset on disk and reads only sampled windows.

### Why This Matters

- Avoids loading the full corpus into memory.
- Reduces disk pressure compared with temporary text/Arrow caches.
- Allows large training data on a normal workstation.

Presenter notes:

The key engineering decision is streaming. Token IDs are written directly to binary files, and the training loader samples windows from those files. This is what makes large-data training possible without needing server-scale memory.

---

## Slide 7: Model Architecture

### Decoder-Only Transformer

| Component | Value |
|---|---:|
| Vocabulary size | 32,000 |
| Context length | 384 tokens |
| Embedding dimension | 768 |
| Transformer layers | 12 |
| Query heads | 12 |
| KV heads | 4 |
| Feed-forward type | SwiGLU |
| Normalization | RMSNorm |
| Positional encoding | RoPE |
| Dropout | 0.1 |

Presenter notes:

The model follows a GPT-style decoder-only architecture, but it uses components that are closer to modern LLaMA/Mistral-style models than the original GPT-2: RMSNorm, RoPE, SwiGLU, and Grouped-Query Attention.

---

## Slide 8: Modern Transformer Choices

### Why These Components?

- RMSNorm stabilizes deep transformer training with lower overhead than LayerNorm.
- RoPE injects positional information directly into attention queries and keys.
- Grouped-Query Attention reduces memory use by sharing key-value heads.
- SwiGLU improves feed-forward expressiveness compared with a simple MLP.
- Flash Attention improves speed and memory efficiency for causal attention.
- Weight tying reuses token embedding weights as the language-model head.

Presenter notes:

Each component solves a specific problem. RMSNorm helps stability, RoPE gives word order, GQA saves memory, SwiGLU improves capacity, Flash Attention improves hardware efficiency, and weight tying reduces redundant parameters.

---

## Slide 9: Training Pipeline

### Training Loop Features

- Mixed precision training with `bfloat16` autocast.
- AdamW optimizer.
- Linear warmup followed by cosine learning-rate decay.
- Gradient clipping with max norm 1.0.
- Evaluation every 2,000 steps.
- Periodic checkpoints every 1,000 steps.
- Automatic resume from latest checkpoint.
- Best validation checkpoint saved as `checkpoints/best_model.pt`.

Presenter notes:

The training script is designed for long-running experiments. If the process stops, it can resume from the latest checkpoint. It also tracks the best validation model separately, which is useful for generation and evaluation.

---

## Slide 10: Current Training Configuration

### Active Training Preset: `subset_10gb`

| Setting | Value |
|---|---:|
| Batch size | 20 |
| Context length | 384 |
| Tokens per step | 7,680 |
| Max iterations | 150,000 |
| Learning rate | 2.5e-4 |
| Minimum learning rate | 2.5e-5 |
| Warmup steps | 2,000 |
| Eval interval | 2,000 |
| Checkpoint interval | 1,000 |

Presenter notes:

The active preset targets a 10 GB tokenized subset and is designed for an RTX 4060-class GPU. The configuration balances model quality with VRAM constraints.

---

## Slide 11: Results So Far

### Training Progress

| Metric | Step 0 | Step 42,000 |
|---|---:|---:|
| Validation loss | 10.5395 | 3.6122 |
| Perplexity | 37,779.67 | 37.05 |

### Observations

- The model moved from near-random prediction to meaningful language modeling.
- Validation loss has steadily decreased across checkpoints.
- The current best logged validation result is at step 42,000.
- GPU memory usage is around 9.3 GB in the current logs.
- Throughput commonly ranges from roughly 3,000 to 6,000 tokens/sec in recent steps.

Presenter notes:

Perplexity is the exponential of loss, so the drop from around 37,780 to 37 is a major improvement. The model is still small and still training, but the trend shows that the architecture and pipeline are working.

---

## Slide 12: Generated Text Example

### Prompt

```text
The future of AI
```

### Example Output at Around 40,000 Steps

```text
The future of Ai ikai, with more information, updates, commentary,
and content is open to all the same. I am excited to see you all...
```

### Interpretation

- The model has learned words, grammar fragments, and topic continuity.
- It still produces factual errors and unstable phrasing.
- This is expected for a compact model trained for a limited number of steps.

Presenter notes:

Generated text is useful because loss alone does not show qualitative behavior. At this point, the model can form recognizable sentences, but it still hallucinates and drifts. That gives a clear direction for future training and evaluation.

---

## Slide 13: Ablation Study Design

### Components That Can Be Switched Off

| Toggle | Purpose | Expected Effect When Disabled |
|---|---|---|
| `USE_RMSNORM` | Training stability | Gradients may explode to NaN |
| `USE_ROPE` | Positional information | Word order and grammar degrade |
| `USE_FLASH_ATTENTION` | Efficient attention | Slower training and higher VRAM |
| `USE_GQA` | Memory-efficient attention | Full MHA uses more VRAM |

Presenter notes:

Ablation studies make the project stronger because they test whether each design choice is actually necessary. Instead of saying the model uses modern components, the project can show what happens when each one is removed.

---

## Slide 14: Key Learnings

### What This Project Demonstrates

- A modern GPT-style model can be implemented from scratch in a readable way.
- Data engineering is as important as model architecture.
- Memory-mapped binary datasets make large-corpus training feasible on local hardware.
- Checkpointing and resume support are essential for real training runs.
- Evaluation should include both quantitative metrics and generated samples.
- Small-scale experiments are valuable before scaling to larger datasets.

Presenter notes:

The biggest lesson is that training a language model is a systems problem. The model matters, but so do data format, memory use, logging, checkpointing, and reproducibility.

---

## Slide 15: Limitations

### Current Constraints

- Single-GPU training only; no distributed training.
- Context length is limited to 384 tokens.
- The model is much smaller than production LLMs.
- No instruction tuning or RLHF.
- No deployment, quantization, or serving layer yet.
- Generated text can still contain hallucinations, repetition, and factual errors.

Presenter notes:

These limitations are expected for the project scope. The goal is not to compete with commercial LLMs, but to create a working, inspectable, and extensible GPT training system.

---

## Slide 16: Future Work

### Next Steps

- Continue training toward the full `subset_10gb` target.
- Run full ablation experiments and compare loss, perplexity, throughput, and VRAM.
- Add plots for training loss, validation loss, perplexity, and learning rate.
- Improve generation evaluation with multiple prompts and sampling settings.
- Explore longer context lengths if VRAM permits.
- Package findings into a research-style report.

Presenter notes:

The immediate next step is to finish training and produce stronger evaluation visuals. After that, ablations and comparison tables can turn this from a working engineering project into a research presentation.

---

## Slide 17: Conclusion

### Final Takeaway

- This project builds a compact GPT-style model from tokenizer to generated text.
- It uses modern transformer ideas while staying understandable and runnable locally.
- The training logs show clear learning progress.
- The codebase is structured for experimentation, reproducibility, and future scaling.

Presenter notes:

The main takeaway is that the project successfully connects theory with implementation. It demonstrates how transformer architecture, data systems, training infrastructure, and evaluation come together to create a working generative language model.

---

## Optional Demo Flow

### If You Want to Show the Project Live

1. Show `config.py` and explain the active preset and ablation toggles.
2. Show `model.py` and point to RMSNorm, RoPE, GQA, SwiGLU, and weight tying.
3. Show `logs/training_metrics.csv` and explain loss/perplexity improvement.
4. Show `Prompt_Outputs/40000_4hr/40000_02` as a generation sample.
5. Run a generation command:

```powershell
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The future of AI is" --max-tokens 100
```

Presenter notes:

For a live demo, keep it short. The strongest story is: architecture, data pipeline, training progress, then generated output.
