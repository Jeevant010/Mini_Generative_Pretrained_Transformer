# Mini Generative Pretrained Transformer

A compact GPT-style language model training project built with PyTorch. The repo uses a modern decoder-only architecture with:

- Grouped-Query Attention (GQA)
- RoPE positional encoding
- RMSNorm
- SwiGLU feed-forward blocks
- tied token embedding / LM head weights

It is designed to train from local parquet shards, write tokenized binaries to disk, and train from memory-mapped data without loading the full corpus into RAM.

## Repo Layout

```text
Mini_Generative_Pretrained_Transformer/
|-- config.py
|-- prepare_data.py
|-- dataset.py
|-- model.py
|-- training.py
|-- generate.py
|-- tokenizer.py
|-- checkpoints/
`-- Research/
```

## Current Defaults

The current default config in [config.py](config.py) is:

- `batch_size = 20`
- `block_size = 384`
- `max_iters = 300000`
- `learning_rate = 2.5e-4`
- `min_lr = 2.5e-5`
- `warmup_iters = 2000`
- `lr_decay_iters = max_iters`
- `grad_clip = 1.0`
- `eval_iters = 25`
- `eval_interval = 2000`
- `checkpoint_interval = 5000`
- `vocab_size = 32000`

Data artifacts are expected at:

- `train.bin`
- `val.bin`
- `bpe_tokenizer_32k.json`

## What The Pipeline Does

`prepare_data.py`
- scans local parquet shards from `DATASET_PATH`
- reads parquet row groups directly instead of building a temporary Arrow cache
- trains a tokenizer if one does not already exist
- tokenizes the corpus
- streams token IDs directly to `train.bin` and `val.bin`

`dataset.py`
- opens the training binaries with `np.memmap`
- samples random windows for each batch
- moves only the current batch to the GPU

`training.py`
- validates the setup before starting
- auto-resumes from the latest `ckpt_step_*.pt`
- logs loss, LR, throughput, and TFLOPS
- saves periodic checkpoints
- saves `checkpoints/best_model.pt` when validation loss improves

`generate.py`
- loads the tokenizer
- loads a checkpoint
- generates text from a prompt

## Setup

Create or activate your Python environment, then install dependencies:

```powershell
pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

## Configure The Dataset

Before preprocessing, open [prepare_data.py](prepare_data.py) and set:

```python
DATASET_PATH = r"D:\Openweb"
```

That folder should contain your parquet shards.

The preprocessing step now reads those parquet files directly, which avoids the large temporary `datasets` Arrow cache that can exhaust disk space on very large corpora.

If you want different output filenames, update these in [config.py](config.py):

```python
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"
TOKENIZER_PATH = "bpe_tokenizer_32k.json"
```

`prepare_data.py`, `dataset.py`, `training.py`, and `generate.py` now all use those config paths consistently.

## How To Run

### 1. Prepare The Data

From the project root:

```powershell
python prepare_data.py
```

This will:

- train the tokenizer if `bpe_tokenizer_32k.json` does not exist
- create `train.bin`
- create `val.bin`

### 2. Train The Model

```powershell
python training.py
```

Training behavior:

- validates that `train.bin` and `val.bin` exist and are large enough
- resumes automatically from the latest `checkpoints/ckpt_step_*.pt` if present
- prints a training log every 100 steps
- runs evaluation every `eval_interval`
- saves periodic checkpoints every `checkpoint_interval`
- saves `checkpoints/best_model.pt` whenever validation loss improves

### 3. Resume Training

No special command is needed. Just run:

```powershell
python training.py
```

If checkpoints exist, training resumes automatically.

### 4. Generate Text

Use the latest step checkpoint:

```powershell
python generate.py --prompt "The future of AI is" --max-tokens 100
```

Use a specific checkpoint, including the best validation model:

```powershell
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The future of AI is" --max-tokens 100
```

## Recommended Workflow

For a first sanity check:

1. Set smaller values in `config.py`, such as lower `max_iters`.
2. Run `python prepare_data.py`.
3. Run `python training.py`.
4. Confirm that `checkpoints/` contains both step checkpoints and `best_model.pt`.
5. Run `python generate.py` on one of those checkpoints.

After that, scale the training run gradually.

## Smoke Test Before A Long Run

Before launching the full training job, do one short validation run to confirm that:

- data preprocessing completed correctly
- `train.bin` and `val.bin` are readable
- training starts without setup errors
- checkpoints are written
- resume works
- generation works from a produced checkpoint

Temporarily change these values in [config.py](config.py):

```python
max_iters = 300
eval_iters = 10
eval_interval = 100
checkpoint_interval = 150
ENABLE_PROFILING = False
TIMER_TARGET_ITERATION = None
```

Then run:

```powershell
python training.py
```

During the smoke test, check that:

- step logs appear normally
- LR is shown in the training log
- evaluation runs at least once
- a file like `checkpoints/ckpt_step_150.pt` appears
- `checkpoints/best_model.pt` appears

Then run training one more time:

```powershell
python training.py
```

This second launch should resume from the existing checkpoint instead of starting from step 0.

Finally, test generation:

```powershell
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The future of AI is" --max-tokens 80
```

If all of that works, restore your long-run values in `config.py` and start the full run.

## Notes For Larger Datasets

The preprocessing path is now stream-based, which means token IDs are written to disk incrementally instead of being collected in Python lists first. That makes it much more suitable for large datasets.

The training path reads batches from memory-mapped binaries, so the full dataset stays on disk. Only the sampled batch is moved to the GPU each step.

## Optional Learning Rate Schedule

`training.py` supports these optional config fields:

```python
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 2.5e-5
```

If they are not present in `config.py`, training falls back to a constant learning rate.

## Profiling

Profiling is controlled in [config.py](config.py):

```python
ENABLE_PROFILING = True
PROFILING_WINDOW = (100, 110)
```

For normal long training runs, setting `ENABLE_PROFILING = False` is usually a better default.

Profiler traces are written under:

```text
log/profiler/
```

You can inspect them with TensorBoard, Chrome tracing, or Perfetto.

## Checkpoints

The project uses two checkpoint types:

- `checkpoints/ckpt_step_<N>.pt`
- `checkpoints/best_model.pt`

`ckpt_step_<N>.pt` is for periodic resume.

`best_model.pt` tracks the best validation loss seen so far.

## Useful Commands

Prepare data:

```powershell
python prepare_data.py
```

Train:

```powershell
python training.py
```

Generate:

```powershell
python generate.py --prompt "Once upon a time" --max-tokens 100
```

Syntax check:

```powershell
python -m py_compile training.py prepare_data.py dataset.py generate.py config.py tokenizer.py model.py
```

## License

See [LICENSE](LICENSE).


# Comprehensive Scaling Analysis: From 118M to 70 Billion Parameters

This document provides a highly detailed, rigorous analysis of the computational requirements, configuration specifications, and estimated training times required to scale the `Mini_Generative_Pretrained_Transformer` codebase to production-grade billion-parameter models. 

---

## 1. Current Architecture Analysis (The 118M Model)

Before scaling, it is critical to understand the foundation of the current model. 

### 1.1 Do We Use Another Company's Embedder?
**No.** We are **NOT** using OpenAI's, Google's, or HuggingFace's pre-trained embeddings. 
In your `tokenizer.py`, you trained a Byte-Pair Encoding (BPE) tokenizer from absolute scratch on your own dataset. In your `model.py`, the `nn.Embedding(config.vocab_size, config.n_embd)` layer initializes completely random weights. 

During Phase 1 (Pre-training), your model learned the mathematical relationships between words entirely on its own. Your embeddings are 100% proprietary to your project.

### 1.2 Current Model Dimensions
Your current configuration exactly mirrors the GPT-2 Small architecture (updated with modern features like RoPE and SwiGLU):
* **Parameters:** ~118 Million
* **Hidden Dimension (`n_embd`):** 768
* **Attention Heads (`n_head`):** 12 (Each head is $768 / 12 = 64$ dimensions)
* **Transformer Layers (`n_layer`):** 12
* **Vocabulary Size (`vocab_size`):** 50,257

---

## 2. Scaling Configurations (Billion-Parameter Thresholds)

To build a model capable of deep factual recall (such as a dedicated Study Assistant), you must drastically increase the number of parameters. Below are the optimal architectural configurations for various scales.

### 2.1 The 1 Billion Parameter Model (Entry Level SLM)
A 1B model is small enough to run inference on a standard laptop CPU but large enough to retain basic domain-specific facts if trained heavily on study material.
* **Hidden Dimension (`n_embd`):** 2048
* **Attention Heads (`n_head`):** 32 (Head dimension: 64)
* **Transformer Layers (`n_layer`):** 22
* **Recommended Training Tokens:** 200 Billion

### 2.2 The 3 Billion Parameter Model (The "Sweet Spot" for Local RAG)
A 3B model is currently the most popular size for on-device applications (e.g., Phi-3, Llama-3 3B). It offers near-human conversational fluency and excellent retrieval-augmented generation (RAG) capabilities.
* **Hidden Dimension (`n_embd`):** 3200
* **Attention Heads (`n_head`):** 32 (Head dimension: 100)
* **Transformer Layers (`n_layer`):** 32
* **Recommended Training Tokens:** 600 Billion

### 2.3 The 7 Billion Parameter Model (Production Grade)
This is the architecture of Llama-1 7B and Llama-2 7B. It possesses significant world knowledge, deep reasoning capabilities, and can write complex code.
* **Hidden Dimension (`n_embd`):** 4096
* **Attention Heads (`n_head`):** 32 (Head dimension: 128)
* **Transformer Layers (`n_layer`):** 32
* **Recommended Training Tokens:** 1.4 Trillion

### 2.4 The 70 Billion Parameter Model (State-of-the-Art)
This scale requires a supercomputer to train and run. It is capable of PhD-level reasoning.
* **Hidden Dimension (`n_embd`):** 8192
* **Attention Heads (`n_head`):** 64 (Head dimension: 128)
* **Transformer Layers (`n_layer`):** 80
* **Recommended Training Tokens:** 3 to 15 Trillion

---

## 3. The Mathematics of Training Compute (FLOPs)

To estimate how long training will take, we must calculate the total Floating Point Operations (FLOPs) required. The universally accepted approximation for training a Transformer is:

$$ \text{Total FLOPs} = 6 \times N \times D $$

Where:
* $N$ = Number of parameters in the model
* $D$ = Number of tokens in the training dataset
* *The multiplier 6 accounts for 2 FLOPs per parameter for the forward pass, and 4 FLOPs per parameter for the backward pass.*

### 3.1 Compute Requirements by Model Size
| Model Size | Target Tokens ($D$) | Total FLOPs Required |
|------------|---------------------|----------------------|
| **118M** | 10 Billion (10GB) | $7.08 \times 10^{18}$ |
| **1B** | 200 Billion | $1.20 \times 10^{21}$ |
| **3B** | 600 Billion | $1.08 \times 10^{22}$ |
| **7B** | 1.4 Trillion | $5.88 \times 10^{22}$ |
| **70B** | 10 Trillion | $4.20 \times 10^{24}$ |

---

## 4. Hardware and Training Time Estimates

Training a Large Language Model is heavily constrained by **Model FLOPs Utilization (MFU)**. An NVIDIA GPU has a theoretical maximum speed, but due to memory bandwidth bottlenecks and communication overhead between GPUs, a highly optimized codebase typically achieves only **40% to 50% MFU**.

### GPU Baselines (bfloat16 / Tensor Core FLOPs)
* **NVIDIA A100 (80GB):** ~312 TFLOPS Theoretical $\rightarrow$ **~140 TFLOPS Effective**
* **NVIDIA H100 (80GB):** ~989 TFLOPS Theoretical $\rightarrow$ **~450 TFLOPS Effective**

### 4.1 Time to Train: 1 Billion Parameter Model
*Total Compute: $1.20 \times 10^{21}$ FLOPs*
* **1x A100:** ~99 days
* **8x A100s (1 Node):** ~12.5 days 
* **8x H100s (1 Node):** ~4 days
* *Estimated Cost (RunPod @ $2/hr per A100):* **~$4,800**

### 4.2 Time to Train: 3 Billion Parameter Model
*Total Compute: $1.08 \times 10^{22}$ FLOPs*
* **8x A100s (1 Node):** ~111 days
* **32x A100s (4 Nodes):** ~28 days 
* **32x H100s (4 Nodes):** ~8.5 days
* *Estimated Cost:* **~$43,000**

### 4.3 Time to Train: 7 Billion Parameter Model
*Total Compute: $5.88 \times 10^{22}$ FLOPs*
* **8x A100s:** ~607 days (Not recommended)
* **128x A100s (16 Nodes):** ~38 days
* **64x H100s (8 Nodes):** ~23 days
* *Estimated Cost:* **~$235,000**

---

## 5. Summary and Strategy for the Study Assistant

Based on the calculations above, building a domain-specific personalized study assistant requires a careful balance of budget and capabilities.

### Recommended Strategy: The 1.5B "Study Specialist"
If you wish to train a model strictly focused on study content, you do not need 7 Billion parameters. A **1.5 Billion parameter model** trained on 300 Billion tokens of highly curated textbooks, math formulas, and study guides is the optimal target.

* **Configuration:** `n_embd=2048`, `n_head=16`, `n_layer=30`
* **Hardware Needed:** One node of 8x H100 GPUs.
* **Time Required:** Approximately 7 to 9 days of continuous training.
* **Pipeline:** 
  1. Pre-train entirely on your custom study text dataset using your `train.py`.
  2. Synthesize 100,000 instruction-following pairs (e.g., "Explain calculus..."). Run your `sft_train.py`.
  3. Synthesize 10,000 preference pairs where the "chosen" response uses your preferred teaching style. Run your `dpo_train.py`.

Your current codebase is fully capable of executing this strategy; the only variables you need to change are the `config.py` dimensions and the renting of a cloud GPU cluster.

---

## 6. Analysis of Local Hardware & Project Configs

You requested an analysis of the exact hardware you currently own (RTX 3050 + RTX 4060) and the specific "high preset" (`full_60gb` with 300,000 iterations) configured inside your `config.py`.

### 6.1 Hardware Capabilities: RTX 3050 vs RTX 4060
Training neural networks locally is constrained by Consumer GPU limits (specifically memory bandwidth and tensor core performance). 

* **RTX 4060 (8GB VRAM):** Can process approximately **~10,000 tokens per second** on a ~117M parameter model.
* **RTX 3050 (4GB/8GB VRAM):** Has roughly half the tensor-core throughput and memory bandwidth of the 4060. It processes approximately **~4,000 to ~5,000 tokens per second**.

**Can you train 1 Billion+ parameter models on these cards?**
* **No.** An RTX 4060 has 8GB of VRAM. A 1 Billion parameter model requires at least 16GB of VRAM just to store the optimizer states (AdamW takes 8 bytes per parameter) and gradients during training. You are strictly limited to models under ~350 Million parameters on an 8GB card.

### 6.2 The "High Preset" (`full_60gb`) Analysis
In your `config.py`, the highest preset is `full_60gb`, which is configured as follows:
* **Batch Size:** 20
* **Block Size:** 384
* **Max Iterations:** 300,000

**Mathematical Reality Check on this Config:**
1. **True Parameter Count:** The config comments say `~85M` parameters, but if we calculate the actual math for 12 layers, 768 hidden size, SwiGLU, and a 32,000 vocabulary size... the model is actually **~117 Million parameters**.
## Checkpoints

The project uses two checkpoint types:

- `checkpoints/ckpt_step_<N>.pt`
- `checkpoints/best_model.pt`

`ckpt_step_<N>.pt` is for periodic resume.

`best_model.pt` tracks the best validation loss seen so far.

## Useful Commands

Prepare data:

```powershell
python prepare_data.py
```

Train:

```powershell
python training.py
```

Generate:

```powershell
python generate.py --prompt "Once upon a time" --max-tokens 100
```

Syntax check:

```powershell
python -m py_compile training.py prepare_data.py dataset.py generate.py config.py tokenizer.py model.py
```

## License

See [LICENSE](LICENSE).


# Comprehensive Scaling Analysis: From 118M to 70 Billion Parameters

This document provides a highly detailed, rigorous analysis of the computational requirements, configuration specifications, and estimated training times required to scale the `Mini_Generative_Pretrained_Transformer` codebase to production-grade billion-parameter models. 

---

## 1. Current Architecture Analysis (The 118M Model)

Before scaling, it is critical to understand the foundation of the current model. 

### 1.1 Do We Use Another Company's Embedder?
**No.** We are **NOT** using OpenAI's, Google's, or HuggingFace's pre-trained embeddings. 
In your `tokenizer.py`, you trained a Byte-Pair Encoding (BPE) tokenizer from absolute scratch on your own dataset. In your `model.py`, the `nn.Embedding(config.vocab_size, config.n_embd)` layer initializes completely random weights. 

During Phase 1 (Pre-training), your model learned the mathematical relationships between words entirely on its own. Your embeddings are 100% proprietary to your project.

### 1.2 Current Model Dimensions
Your current configuration exactly mirrors the GPT-2 Small architecture (updated with modern features like RoPE and SwiGLU):
* **Parameters:** ~118 Million
* **Hidden Dimension (`n_embd`):** 768
* **Attention Heads (`n_head`):** 12 (Each head is $768 / 12 = 64$ dimensions)
* **Transformer Layers (`n_layer`):** 12
* **Vocabulary Size (`vocab_size`):** 50,257

---

## 2. Scaling Configurations (Billion-Parameter Thresholds)

To build a model capable of deep factual recall (such as a dedicated Study Assistant), you must drastically increase the number of parameters. Below are the optimal architectural configurations for various scales.

### 2.1 The 1 Billion Parameter Model (Entry Level SLM)
A 1B model is small enough to run inference on a standard laptop CPU but large enough to retain basic domain-specific facts if trained heavily on study material.
* **Hidden Dimension (`n_embd`):** 2048
* **Attention Heads (`n_head`):** 32 (Head dimension: 64)
* **Transformer Layers (`n_layer`):** 22
* **Recommended Training Tokens:** 200 Billion

### 2.2 The 3 Billion Parameter Model (The "Sweet Spot" for Local RAG)
A 3B model is currently the most popular size for on-device applications (e.g., Phi-3, Llama-3 3B). It offers near-human conversational fluency and excellent retrieval-augmented generation (RAG) capabilities.
* **Hidden Dimension (`n_embd`):** 3200
* **Attention Heads (`n_head`):** 32 (Head dimension: 100)
* **Transformer Layers (`n_layer`):** 32
* **Recommended Training Tokens:** 600 Billion

### 2.3 The 7 Billion Parameter Model (Production Grade)
This is the architecture of Llama-1 7B and Llama-2 7B. It possesses significant world knowledge, deep reasoning capabilities, and can write complex code.
* **Hidden Dimension (`n_embd`):** 4096
* **Attention Heads (`n_head`):** 32 (Head dimension: 128)
* **Transformer Layers (`n_layer`):** 32
* **Recommended Training Tokens:** 1.4 Trillion

### 2.4 The 70 Billion Parameter Model (State-of-the-Art)
This scale requires a supercomputer to train and run. It is capable of PhD-level reasoning.
* **Hidden Dimension (`n_embd`):** 8192
* **Attention Heads (`n_head`):** 64 (Head dimension: 128)
* **Transformer Layers (`n_layer`):** 80
* **Recommended Training Tokens:** 3 to 15 Trillion

---

## 3. The Mathematics of Training Compute (FLOPs)

To estimate how long training will take, we must calculate the total Floating Point Operations (FLOPs) required. The universally accepted approximation for training a Transformer is:

$$ \text{Total FLOPs} = 6 \times N \times D $$

Where:
* $N$ = Number of parameters in the model
* $D$ = Number of tokens in the training dataset
* *The multiplier 6 accounts for 2 FLOPs per parameter for the forward pass, and 4 FLOPs per parameter for the backward pass.*

### 3.1 Compute Requirements by Model Size
| Model Size | Target Tokens ($D$) | Total FLOPs Required |
|------------|---------------------|----------------------|
| **118M** | 10 Billion (10GB) | $7.08 \times 10^{18}$ |
| **1B** | 200 Billion | $1.20 \times 10^{21}$ |
| **3B** | 600 Billion | $1.08 \times 10^{22}$ |
| **7B** | 1.4 Trillion | $5.88 \times 10^{22}$ |
| **70B** | 10 Trillion | $4.20 \times 10^{24}$ |

---

## 4. Hardware and Training Time Estimates

Training a Large Language Model is heavily constrained by **Model FLOPs Utilization (MFU)**. An NVIDIA GPU has a theoretical maximum speed, but due to memory bandwidth bottlenecks and communication overhead between GPUs, a highly optimized codebase typically achieves only **40% to 50% MFU**.

### GPU Baselines (bfloat16 / Tensor Core FLOPs)
* **NVIDIA A100 (80GB):** ~312 TFLOPS Theoretical $\rightarrow$ **~140 TFLOPS Effective**
* **NVIDIA H100 (80GB):** ~989 TFLOPS Theoretical $\rightarrow$ **~450 TFLOPS Effective**

### 4.1 Time to Train: 1 Billion Parameter Model
*Total Compute: $1.20 \times 10^{21}$ FLOPs*
* **1x A100:** ~99 days
* **8x A100s (1 Node):** ~12.5 days 
* **8x H100s (1 Node):** ~4 days
* *Estimated Cost (RunPod @ $2/hr per A100):* **~$4,800**

### 4.2 Time to Train: 3 Billion Parameter Model
*Total Compute: $1.08 \times 10^{22}$ FLOPs*
* **8x A100s (1 Node):** ~111 days
* **32x A100s (4 Nodes):** ~28 days 
* **32x H100s (4 Nodes):** ~8.5 days
* *Estimated Cost:* **~$43,000**

### 4.3 Time to Train: 7 Billion Parameter Model
*Total Compute: $5.88 \times 10^{22}$ FLOPs*
* **8x A100s:** ~607 days (Not recommended)
* **128x A100s (16 Nodes):** ~38 days
* **64x H100s (8 Nodes):** ~23 days
* *Estimated Cost:* **~$235,000**

---

## 5. Summary and Strategy for the Study Assistant

Based on the calculations above, building a domain-specific personalized study assistant requires a careful balance of budget and capabilities.

### Recommended Strategy: The 1.5B "Study Specialist"
If you wish to train a model strictly focused on study content, you do not need 7 Billion parameters. A **1.5 Billion parameter model** trained on 300 Billion tokens of highly curated textbooks, math formulas, and study guides is the optimal target.

* **Configuration:** `n_embd=2048`, `n_head=16`, `n_layer=30`
* **Hardware Needed:** One node of 8x H100 GPUs.
* **Time Required:** Approximately 7 to 9 days of continuous training.
* **Pipeline:** 
  1. Pre-train entirely on your custom study text dataset using your `train.py`.
  2. Synthesize 100,000 instruction-following pairs (e.g., "Explain calculus..."). Run your `sft_train.py`.
  3. Synthesize 10,000 preference pairs where the "chosen" response uses your preferred teaching style. Run your `dpo_train.py`.

Your current codebase is fully capable of executing this strategy; the only variables you need to change are the `config.py` dimensions and the renting of a cloud GPU cluster.

---

## 6. Analysis of Local Hardware & Project Configs

You requested an analysis of the exact hardware you currently own (RTX 3050 + RTX 4060) and the specific "high preset" (`full_60gb` with 300,000 iterations) configured inside your `config.py`.

### 6.1 Hardware Capabilities: RTX 3050 vs RTX 4060
Training neural networks locally is constrained by Consumer GPU limits (specifically memory bandwidth and tensor core performance). 

* **RTX 4060 (8GB VRAM):** Can process approximately **~10,000 tokens per second** on a ~117M parameter model.
* **RTX 3050 (4GB/8GB VRAM):** Has roughly half the tensor-core throughput and memory bandwidth of the 4060. It processes approximately **~4,000 to ~5,000 tokens per second**.

**Can you train 1 Billion+ parameter models on these cards?**
* **No.** An RTX 4060 has 8GB of VRAM. A 1 Billion parameter model requires at least 16GB of VRAM just to store the optimizer states (AdamW takes 8 bytes per parameter) and gradients during training. You are strictly limited to models under ~350 Million parameters on an 8GB card.

### 6.2 The "High Preset" (`full_60gb`) Analysis
In your `config.py`, the highest preset is `full_60gb`, which is configured as follows:
* **Batch Size:** 20
* **Block Size:** 384
* **Max Iterations:** 300,000

**Mathematical Reality Check on this Config:**
1. **True Parameter Count:** The config comments say `~85M` parameters, but if we calculate the actual math for 12 layers, 768 hidden size, SwiGLU, and a 32,000 vocabulary size... the model is actually **~117 Million parameters**.
2. **Tokens Processed:** 20 (batch size) $\times$ 384 (block size) = **7,680 tokens per step**.
3. **Total Training Tokens:** 7,680 tokens/step $\times$ 300,000 iterations = **~2.3 Billion tokens**. *(Note: The comment in your config file says ~30B tokens, but with a cap of 300,000 iterations, it will mathematically stop at 2.3 Billion tokens).*
4. **Time to Train on RTX 4060:** 2.3 Billion tokens $\div$ 10,000 tokens/second = 230,000 seconds = **~2.66 Days** of continuous 24/7 training.
5. **Time to Train on RTX 3050:** 2.3 Billion tokens $\div$ 4,500 tokens/second = 511,111 seconds = **~5.9 Days** of continuous 24/7 training.

### Summary of Local Config
If you run the `full_60gb` preset locally on your RTX 4060, it will take about **2.5 to 3 days** to complete the 300,000 iterations. It will produce a **117M parameter model** trained on 2.3 Billion tokens. This is a very solid "small" model, but it will still hallucinate facts because it is not in the "Billion Parameter" tier required for deep factual recall.

---

## 7. Data Size vs. Token Math (How Much Data Do You Need?)
You asked how dataset size (in Gigabytes) translates to token counts (e.g., 10 Billion or 20 Billion tokens).

### The "4 Bytes per Token" Rule
When using a standard Byte-Pair Encoding (BPE) tokenizer on English text, one token is approximately equal to **4 characters (or 4 bytes)**.
Since 1 Gigabyte (GB) is roughly 1 Billion bytes, we can use the following standard industry conversion:
**1 GB of plain text = ~250 Million tokens.**

### Conversions for your Training:
* **10 GB of text** $\approx$ **2.5 Billion tokens.**
* **40 GB of text** $\approx$ **10 Billion tokens.**
* **80 GB of text** $\approx$ **20 Billion tokens.**
* **120 GB of text** $\approx$ **30 Billion tokens.**

If you want to train a model on **20 Billion tokens**, you will need to scrape, download, and clean approximately **80 Gigabytes of raw `.txt` files** (PDFs and HTML must be converted to plain text first, which strips away a lot of their original file size).

---

## 8. Real-World Case Studies: Educational & Indian AI Models
Your goal to create a personalized study/educational model is exactly what the top tech companies in India are currently doing. The strategy of taking an LLM and making it hyper-focused on a specific domain (like education or regional context) is the most profitable sector of AI right now.

Here are the exact models you mentioned and others working in this space:

### 1. PhysicsWallah (PW) - "Alakh AI"
You mentioned PW; they recently launched **Alakh AI**. It is an educational AI suite launched by the EdTech unicorn to serve as a 24/7 personalized tutor.
* **What it does:** It uses RAG (Retrieval-Augmented Generation) mixed with fine-tuned educational models to solve student doubts, create personalized study plans, and explain complex math/physics concepts.
* **Relevance to you:** Your project is essentially the foundational blueprint of Alakh AI. By training a model strictly on textbooks and study guides, you are building the same technology PW is using to scale personalized tutoring.

### 2. Sarvam AI - "OpenHathi" (Southern India)
You mentioned an industry in the southern part of India. This is likely **Sarvam AI**, a high-profile startup based in Bangalore. 
* **What they built:** They created **OpenHathi**, the first open-source Hindi LLM. Instead of trying to beat GPT-4 in English, they took a 7-Billion parameter Llama model and continuously pre-trained it on massive amounts of Hindi text. 
* **Relevance to you:** They proved that you don't need a Trillion-parameter model if you restrict the "domain". By restricting the domain to the Hindi language, a 7B model can outperform much larger models. You are applying this exact same philosophy, but restricting your domain to *Study Material*.

### 3. Ola - "Krutrim"
* **What they built:** Claimed as India's own foundational AI model, Krutrim was trained from absolute scratch on over 2 Trillion tokens with a heavy emphasis on Indic languages and Indian cultural nuances.
* **Relevance to you:** Krutrim proves that training foundational models (what you did in Phase 1 of this project) is still highly valuable when you want the model's "brain" to inherently understand a specific culture or context without relying on Western models like ChatGPT.

### 4. Tech Mahindra - "Project Indus"
* **What they built:** A foundational model designed to speak Hindi and its 37 dialects. They crowd-sourced data collection to gather audio and text from across rural India to train the model.

### The Takeaway
You are on the right track. Attempting to build a general-purpose AI to beat ChatGPT is nearly impossible for a solo developer. But building a **Domain-Specific AI** (like Alakh AI for studying, or OpenHathi for Hindi) is exactly where the industry is heading, and your codebase is fully equipped to do it.
