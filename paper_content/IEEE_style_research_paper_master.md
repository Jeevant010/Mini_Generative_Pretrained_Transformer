# Attention on a Budget: Engineering a Custom GPT-Style Language Model on Consumer Hardware

Jeevant  
Department of Computer Science and Engineering  
IIIT Surat  
Surat, India  
email or ORCID

Deepesh Dangi  
Department of Computer Science and Engineering  
IIIT Surat  
Surat, India  
deepeshdangi700@gmail.com

Bhupendra Kumar  
Department of Computer Science and Engineering  
IIIT Surat  
Surat, India  
email or ORCID

## Abstract

This paper presents the design, implementation, and experimental analysis of a compact GPT-style language model trained from scratch on consumer-grade GPU hardware. The project implements a decoder-only Transformer architecture with modern efficiency-oriented components: byte-level Byte Pair Encoding (BPE), Rotary Positional Embeddings (RoPE), Root Mean Square Layer Normalization (RMSNorm), Grouped-Query Attention (GQA), SwiGLU feed-forward layers, Flash Attention through PyTorch scaled dot-product attention, and tied input-output token embeddings. The training pipeline is designed for local multi-gigabyte web-text corpora stored as parquet shards and converted into memory-mapped binary token files, allowing large datasets to be sampled without loading the full corpus into RAM.

The current implementation trains a 117,787,392-parameter model with 12 Transformer blocks, 768-dimensional embeddings, 12 query heads, 4 key-value heads, a 384-token context length, and a 32,000-token byte-level BPE vocabulary. A 10 GiB tokenized dataset was prepared, containing 5,100,766,548 training tokens and 267,942,572 validation tokens. After 60,000 optimization steps, the model reached validation loss 3.517095 and perplexity 33.69, down from initial validation loss 10.539526 and perplexity 37,779.67. Generated samples show that the model has learned fluent local syntax and text-continuation behavior, while also demonstrating the expected limitation of a base pretrained model: it does not behave as an instruction-following assistant without additional instruction tuning. The work demonstrates that careful architecture selection, memory-mapped data loading, and systematic measurement can make small-scale language-model pretraining feasible on limited hardware.

**Index Terms**: small language model, GPT, Transformer, decoder-only model, Grouped-Query Attention, RoPE, RMSNorm, SwiGLU, byte-level BPE, memory-mapped training, consumer GPU.

## I. Introduction

Large language models are typically associated with large-scale compute clusters, industrial data pipelines, and extensive training budgets. However, the core ideas behind generative language modeling can be studied at a smaller scale when the system is engineered carefully. This project investigates that setting by implementing and training a GPT-style language model from scratch on consumer hardware.

The aim is not to compete with frontier language models, but to build a complete, measurable, and reproducible language-model training pipeline. The project covers all major stages of language-model development: raw text ingestion, document filtering, tokenizer training, binary dataset creation, memory-mapped batch sampling, Transformer architecture implementation, optimization, checkpointing, validation, perplexity evaluation, qualitative generation, and ablation planning.

The central research question is:

Can a modern GPT-style model be trained from scratch on a multi-gigabyte web-text subset using consumer hardware, while retaining rigorous model design, measurement, and reproducibility?

This question is divided into five practical sub-questions:

- How can the model be sized to fit a consumer laptop GPU?
- How can 10 GiB of tokenized text be sampled efficiently without loading it fully into RAM?
- Which modern Transformer components are essential for stability and which primarily improve efficiency?
- What measurable improvement is visible after partial training?
- Why does a base pretrained model generate continuation text rather than assistant-style answers?

The current model is a decoder-only Transformer with 117,787,392 trainable parameters. It uses 12 blocks, 768-dimensional embeddings, 12 query heads, 4 key-value heads, RoPE, RMSNorm, GQA, SwiGLU, Flash Attention, and tied input-output embeddings. The data pipeline produces compact `uint16` token binaries from parquet shards and reads them through NumPy memory maps during training.

The current experiment uses the `subset_10gb` configuration. The prepared dataset contains 5.10B training tokens and 267.9M validation tokens. At the latest observed checkpoint, step 60,000, the model reached validation loss 3.517095 and perplexity 33.69. This indicates strong learning progress from initialization, while generated samples still show topic drift and base-model continuation behavior.

## II. Contributions

This work makes the following contributions:

- A complete GPT-style decoder-only Transformer implementation in PyTorch.
- Integration of modern architecture components: RoPE, RMSNorm, GQA, SwiGLU, Flash Attention, and weight tying.
- A streaming parquet-to-binary data pipeline for multi-gigabyte local datasets.
- A byte-level BPE tokenizer trained for a 32,000-token vocabulary.
- A memory-mapped dataset loader that samples random training windows from large token files.
- A production training loop with warmup cosine learning rate, AdamW, gradient clipping, checkpoint resume, validation loss, perplexity, throughput logging, VRAM logging, and sample generation.
- A current 10 GiB experiment with 60,000-step results and quantitative evidence of training progress.
- An ablation framework for studying RMSNorm, RoPE, Flash Attention, and GQA.

## III. Background

### A. Autoregressive Language Modeling

The model is trained using an autoregressive next-token prediction objective. For a token sequence:

$$
x_1, x_2, ..., x_T
$$

the joint probability is factorized as:

$$
P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

The model therefore learns to estimate the probability of each token given previous tokens. During training, the input sequence is shifted by one position to form targets:

$$
x = [x_1, x_2, ..., x_T]
$$

$$
y = [x_2, x_3, ..., x_{T+1}]
$$

This makes every position in the sequence a supervised classification problem over the vocabulary.

### B. Decoder-Only Transformer

A decoder-only Transformer is appropriate for autoregressive generation because it uses causal attention. At position `t`, the model may attend to positions `1` through `t`, but not to future positions. This is implemented with a causal mask.

The project uses the following notation:

| Symbol | Meaning | Current value |
| --- | --- | ---: |
| `B` | Batch size | 20 |
| `T` | Context length | 384 |
| `V` | Vocabulary size | 32,000 |
| `d` | Embedding width | 768 |
| `L` | Number of Transformer blocks | 12 |
| `H_q` | Query heads | 12 |
| `H_kv` | Key-value heads | 4 |
| `d_h` | Head dimension | 64 |

The token batch has shape:

$$
X \in \mathbb{N}^{B \times T}
$$

The embedding output has shape:

$$
H^{(0)} \in \mathbb{R}^{B \times T \times d}
$$

## IV. System Overview

The complete training system follows this pipeline:

```text
parquet shards
    -> document filtering
    -> byte-level BPE tokenizer training/loading
    -> batched tokenization with <eos>
    -> train.bin and val.bin
    -> memory-mapped random batch sampling
    -> decoder-only GPT training
    -> checkpointing and metrics logging
    -> text generation and qualitative evaluation
```

The core files are:

| File | Purpose |
| --- | --- |
| `config.py` | Architecture, training, data, logging, profiling, and ablation settings |
| `prepare_data.py` | Parquet reading, filtering, tokenizer training/loading, binary writing |
| `tokenizer.py` | Byte-level BPE tokenizer wrapper |
| `dataset.py` | Memory-mapped token loading and random batch sampling |
| `model.py` | Decoder-only GPT model |
| `training.py` | Production training loop |
| `generate.py` | Text generation from checkpoints |
| `evaluation/perplexity.py` | Standalone perplexity evaluation |
| `ablation/run_ablation.py` | Ablation runner |

The current production training entry point is `training.py`. The older `train.py` script is not used for current results.

## V. Data Preparation

### A. Dataset Format

The project expects local parquet shards under:

```python
DATASET_PATH = r"D:\Openweb"
```

The preprocessing script searches for parquet files and detects one of the following text columns:

```text
text, content, document, body
```

It can also detect optional language and quality columns:

```text
language, lang, language_code
quality_score, score, quality, rank, rating
```

### B. Document Filtering

The data pipeline filters documents before tokenization to reduce low-quality web text. The current filters include:

| Filter | Value |
| --- | ---: |
| Minimum document characters | 200 |
| Maximum document characters | 50,000 |
| Minimum word count | 50 |
| Minimum alphabetic character ratio | 0.55 |
| Minimum ASCII alphabetic ratio | 0.85 |
| Maximum digit character ratio | 0.20 |
| Maximum non-ASCII character ratio | 0.20 |
| Minimum English stopword ratio | 0.02 |
| Maximum URL count | 10 |
| Maximum repeated-line ratio | 0.30 |

For a document with character count `C`, alphabetic count `A`, digit count `D`, and non-ASCII count `N`, the filter computes:

$$
r_\alpha = \frac{A}{C}
$$

$$
r_d = \frac{D}{C}
$$

$$
r_n = \frac{N}{C}
$$

Documents are rejected if:

$$
r_\alpha < 0.55
$$

or:

$$
r_d > 0.20
$$

or:

$$
r_n > 0.20
$$

The stopword ratio is:

$$
r_s = \frac{S}{W}
$$

where `S` is the number of stopword hits and `W` is the number of detected words. Documents are rejected if:

$$
r_s < 0.02
$$

### C. Byte-Level BPE Tokenization

The tokenizer is implemented with the HuggingFace `tokenizers` library:

```python
Tokenizer(models.BPE(unk_token="<unk>"))
pre_tokenizers.ByteLevel(add_prefix_space=False)
decoders.ByteLevel()
```

The vocabulary size is:

$$
V = 32000
$$

The special tokens are:

```text
<pad>, <bos>, <eos>, <unk>
```

BPE repeatedly merges the most frequent adjacent token pair. If:

$$
c(a,b)
$$

is the count of adjacent pair `(a,b)`, BPE selects:

$$
(a^*, b^*) = \arg\max_{(a,b)} c(a,b)
$$

and replaces occurrences of `a,b` with a merged token `ab`. This process continues until the target vocabulary size is reached.

### D. Binary Token Storage

Each document is encoded and receives an end-of-sequence token:

$$
[t_1, t_2, ..., t_n] \rightarrow [t_1, t_2, ..., t_n, \texttt{<eos>}]
$$

Token IDs are stored as `uint16`:

```python
arr = np.asarray(tokens, dtype=np.uint16)
```

This is valid because:

$$
32000 < 65535
$$

Each token therefore uses 2 bytes. The current data artifacts are:

| File | Size | Token count |
| --- | ---: | ---: |
| `train.bin` | 9.50 GiB | 5,100,766,548 |
| `val.bin` | 511.06 MiB | 267,942,572 |
| `bpe_tokenizer_32k.json` | 2.16 MiB | 32,000 vocabulary entries |

The total tokenized bytes are:

$$
10{,}201{,}533{,}096 + 535{,}885{,}144 = 10{,}737{,}418{,}240
$$

which equals:

$$
10 \times 1024^3
$$

bytes.

### E. Train-Validation Split

The validation probability is:

$$
p_{val} = 0.05
$$

For each document:

$$
d_i \in
\begin{cases}
\text{validation}, & u < p_{val} \\
\text{train}, & u \geq p_{val}
\end{cases}
$$

where:

$$
u \sim \operatorname{Uniform}(0,1)
$$

The observed validation fraction is:

$$
\frac{267{,}942{,}572}{5{,}100{,}766{,}548 + 267{,}942{,}572}
\approx 0.0499
$$

which matches the intended 5 percent split.

## VI. Model Architecture

### A. High-Level Architecture

The model class is `GPTLanguageModel` in `model.py`. Its forward pass is:

```text
token IDs
  -> token embedding
  -> Transformer block x 12
       -> RMSNorm
       -> GQA causal attention with RoPE
       -> residual add
       -> RMSNorm
       -> SwiGLU FFN
       -> residual add
  -> final RMSNorm
  -> tied LM head
  -> logits
```

The active architecture is:

| Component | Value |
| --- | --- |
| Architecture | Decoder-only Transformer |
| Layers | 12 |
| Embedding dimension | 768 |
| Query heads | 12 |
| KV heads | 4 |
| Head dimension | 64 |
| Context length | 384 |
| Vocabulary | 32,000 |
| Normalization | RMSNorm |
| Position encoding | RoPE |
| Attention | GQA with Flash Attention |
| FFN | SwiGLU |
| Dropout | 0.1 |
| Weight tying | Yes |

### B. Token Embedding

Each token ID is mapped to a learned vector:

$$
h_t^{(0)} = E[x_t]
$$

where:

$$
E \in \mathbb{R}^{V \times d}
$$

For this model:

$$
Vd = 32000 \times 768 = 24{,}576{,}000
$$

embedding parameters.

### C. Transformer Block

Each block uses pre-normalization:

$$
u^{(l)} = h^{(l)} + \operatorname{Attn}(\operatorname{RMSNorm}(h^{(l)}))
$$

$$
h^{(l+1)} = u^{(l)} + \operatorname{SwiGLU}(\operatorname{RMSNorm}(u^{(l)}))
$$

Pre-normalization keeps the residual stream stable and improves optimization behavior.

### D. RMSNorm

RMSNorm normalizes by root mean square:

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}
$$

$$
\operatorname{RMSNorm}(x)_i = g_i\frac{x_i}{\operatorname{RMS}(x)}
$$

where `g` is a learned scale vector and `epsilon = 1e-6`.

Unlike LayerNorm, RMSNorm does not subtract the mean:

$$
\operatorname{LayerNorm}(x)_i =
\gamma_i \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

This makes RMSNorm computationally simpler while retaining scale stabilization.

### E. Query, Key, And Value Projections

The model computes:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

For the current model:

$$
d = 768,\quad H_q = 12,\quad H_{kv}=4,\quad d_h=64
$$

The query shape is:

$$
Q \in \mathbb{R}^{B \times H_q \times T \times d_h}
$$

The key and value shapes before repetition are:

$$
K,V \in \mathbb{R}^{B \times H_{kv} \times T \times d_h}
$$

### F. Causal Attention

The raw attention scores are:

$$
S = \frac{QK^T}{\sqrt{d_h}}
$$

The causal mask is:

$$
M_{ij} =
\begin{cases}
0, & j \leq i \\
-\infty, & j > i
\end{cases}
$$

The attention distribution is:

$$
A = \operatorname{softmax}(S + M)
$$

The output is:

$$
O = AV
$$

### G. Rotary Positional Embedding

Self-attention has no inherent order information. RoPE injects token position by rotating query and key vectors.

For a two-dimensional pair `(a,b)` at position `m`:

$$
\begin{bmatrix}
a' \\
b'
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
a \\
b
\end{bmatrix}
$$

where:

$$
\theta_i = 10000^{-2i/d_h}
$$

The implementation computes:

$$
\operatorname{RoPE}(x) = x \odot \cos(\Theta) + \operatorname{rotate\_half}(x)\odot \sin(\Theta)
$$

RoPE is applied to `Q` and `K`, not `V`.

### H. Grouped-Query Attention

Grouped-Query Attention uses fewer key-value heads than query heads:

$$
H_{kv} < H_q
$$

In this project:

$$
H_q = 12,\quad H_{kv}=4
$$

Each key-value head is shared by:

$$
g = \frac{H_q}{H_{kv}} = 3
$$

query heads.

Full MHA key-value projections would require:

$$
2d^2 = 2(768)(768) = 1{,}179{,}648
$$

parameters per layer. GQA uses:

$$
2d(H_{kv}d_h) = 2(768)(4 \times 64) = 393{,}216
$$

parameters per layer. The saving is:

$$
1{,}179{,}648 - 393{,}216 = 786{,}432
$$

parameters per layer.

### I. SwiGLU Feed-Forward Network

The feed-forward network uses SwiGLU:

$$
\operatorname{SwiGLU}(x)=W_{out}(\operatorname{SiLU}(xW_1)\odot xW_2)
$$

where:

$$
\operatorname{SiLU}(z)=z\sigma(z)
$$

and:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The hidden width is:

$$
d_{ff}=\lfloor 3.5 \times 768 \rfloor = 2688
$$

The feed-forward parameter count per layer is:

$$
3 \times 768 \times 2688 = 6{,}193{,}152
$$

### J. Weight Tying

The token embedding and output language-model head share parameters:

```python
self.token_embed.weight = self.lm_head.weight
```

Mathematically:

$$
W_{lm}=E^T
$$

This reduces the parameter count by avoiding a second independent `d x V` output matrix.

### K. Parameter Count

Per Transformer block:

| Component | Parameters |
| --- | ---: |
| Query projection | 589,824 |
| Key projection | 196,608 |
| Value projection | 196,608 |
| Output projection | 589,824 |
| Attention total | 1,572,864 |
| SwiGLU `W1` | 2,064,384 |
| SwiGLU `W2` | 2,064,384 |
| SwiGLU `Wout` | 2,064,384 |
| FFN total | 6,193,152 |
| Two RMSNorm scales | 1,536 |
| Total per block | 7,767,552 |

For 12 blocks:

$$
12 \times 7{,}767{,}552 = 93{,}210{,}624
$$

Embedding and tied LM head:

$$
32000 \times 768 = 24{,}576{,}000
$$

Final RMSNorm:

$$
768
$$

Total:

$$
93{,}210{,}624 + 24{,}576{,}000 + 768 = 117{,}787{,}392
$$

Therefore, the model has:

$$
\boxed{117{,}787{,}392}
$$

trainable parameters.

## VII. Training Methodology

### A. Batch Sampling

The dataset loader opens `train.bin` and `val.bin` with:

```python
np.memmap(path, dtype=np.uint16, mode="r")
```

For each batch element, it samples a random start position:

$$
s \sim \operatorname{UniformInteger}(0, N-T-1)
$$

and creates:

$$
x = [t_s,t_{s+1},...,t_{s+T-1}]
$$

$$
y = [t_{s+1},t_{s+2},...,t_{s+T}]
$$

This enables random window sampling from large token files without loading the full dataset.

### B. Loss Function

The model produces logits:

$$
z_t \in \mathbb{R}^{V}
$$

The predicted probability of token `i` is:

$$
P_\theta(x_{t+1}=i \mid x_{\leq t})
=
\frac{\exp(z_{t,i})}{\sum_{j=1}^{V}\exp(z_{t,j})}
$$

The per-token loss is:

$$
\ell_t=-\log P_\theta(y_t \mid x_{\leq t})
$$

The batch loss is:

$$
\mathcal{L}(\theta)=
-\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log P_\theta(y_{b,t}\mid x_{b,\leq t})
$$

### C. Optimizer

The optimizer is AdamW. Adam maintains first and second moment estimates:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

Bias-corrected estimates are:

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

AdamW applies decoupled weight decay:

$$
\theta_t =
\theta_{t-1}
- \eta_t
\left(
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
+ \lambda \theta_{t-1}
\right)
$$

### D. Learning-Rate Schedule

The current maximum learning rate is:

$$
\eta_{max}=2.5 \times 10^{-4}
$$

The minimum learning rate is:

$$
\eta_{min}=2.5 \times 10^{-5}
$$

Warmup lasts:

$$
T_{warmup}=2000
$$

steps. During warmup:

$$
\eta_t=\eta_{max}\frac{t+1}{T_{warmup}}
$$

After warmup, cosine decay is used:

$$
r_t=\frac{t-T_{warmup}}{T_{decay}-T_{warmup}}
$$

$$
\eta_t=\eta_{min}+\frac{1}{2}(1+\cos(\pi r_t))(\eta_{max}-\eta_{min})
$$

where:

$$
T_{decay}=150000
$$

### E. Gradient Clipping

The global gradient norm is:

$$
\|g\|_2=\sqrt{\sum_i\|g_i\|_2^2}
$$

If:

$$
\|g\|_2 > 1.0
$$

the gradients are rescaled:

$$
g_i \leftarrow g_i \frac{1.0}{\|g\|_2}
$$

This reduces instability from unusually large gradient updates.

### F. Mixed Precision

The training loop uses bfloat16 autocast:

```python
torch.autocast(device_type="cuda", dtype=torch.bfloat16)
```

bfloat16 reduces memory bandwidth and can improve throughput while preserving a wider exponent range than float16.

### G. Checkpointing

The training loop saves periodic checkpoints:

```text
checkpoints/ckpt_step_<N>.pt
```

and a best validation checkpoint:

```text
checkpoints/best_model.pt
```

Each checkpoint contains:

```text
step
model_state_dict
optimizer_state_dict
loss
best_val_loss
```

This enables interrupted training to resume from the latest checkpoint.

## VIII. Evaluation Metrics

### A. Perplexity

Perplexity is the exponentiated cross-entropy:

$$
\operatorname{PPL}=e^{\mathcal{L}}
$$

At the latest observed checkpoint:

$$
\mathcal{L}_{val}=3.517095
$$

Therefore:

$$
\operatorname{PPL}=e^{3.517095}\approx 33.69
$$

### B. Throughput

Tokens per second are computed as:

$$
\operatorname{tokens/sec} = \frac{BT}{\Delta t}
$$

For the active setting:

$$
BT=20 \times 384=7680
$$

### C. FLOP Estimate

The training script estimates FLOPs per token as:

$$
F_{token}\approx 6N_{params}
$$

For:

$$
N_{params}=117{,}787{,}392
$$

we get:

$$
F_{token}\approx 706{,}724{,}352
$$

Per step:

$$
F_{step}\approx 706{,}724{,}352 \times 7680
\approx 5.43 \times 10^{12}
$$

Estimated TFLOPS are:

$$
\operatorname{TFLOPS}=
\frac{F_{step}}{\Delta t \times 10^{12}}
$$

### D. Qualitative Samples

Generated samples are used to inspect:

- local grammar
- repetition
- topic drift
- base-model continuation behavior
- prompt sensitivity

They are not treated as a replacement for validation loss and perplexity.

## IX. Experimental Setup

The active experiment uses `subset_10gb`:

| Setting | Value |
| --- | ---: |
| Model parameters | 117,787,392 |
| Layers | 12 |
| Embedding dimension | 768 |
| Query heads | 12 |
| KV heads | 4 |
| Context length | 384 |
| Vocabulary size | 32,000 |
| Batch size | 20 |
| Tokens per step | 7,680 |
| Planned steps | 150,000 |
| Latest observed step | 60,000 |
| Optimizer | AdamW |
| Peak learning rate | 2.5e-4 |
| Minimum learning rate | 2.5e-5 |
| Warmup | 2,000 steps |
| LR decay | Cosine |
| Gradient clipping | 1.0 |
| Dropout | 0.1 |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU |

The dataset contains:

| Split | Tokens |
| --- | ---: |
| Train | 5,100,766,548 |
| Validation | 267,942,572 |

## X. Results

### A. Loss And Perplexity Improvement

At initialization:

| Step | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 0 | 10.539526 | 37,779.67 |

At the latest observed checkpoint:

| Step | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 60,000 | 3.517095 | 33.69 |

The validation-loss reduction is:

$$
10.539526 - 3.517095 = 7.022431
$$

The perplexity reduction factor is:

$$
\frac{37779.67}{33.69}\approx 1121.4
$$

This shows that the model learned substantial next-token structure from the training data.

The corresponding training and validation loss curves are generated at:

```text
paper_content/figures/loss_curves.png
```

In the final IEEE paper, this can be inserted as:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/loss_curves.png}
    \caption{Training loss and validation loss over optimization steps. Training loss is shown with a rolling mean to reduce step-level noise.}
    \label{fig:loss-curves}
\end{figure}
```

The validation perplexity curve is generated at:

```text
paper_content/figures/validation_perplexity.png
```

### B. Recent Validation Trend

| Step | Train loss | Validation loss | PPL |
| ---: | ---: | ---: | ---: |
| 42,000 | 3.618999 | 3.612168 | 37.05 |
| 44,000 | 3.625918 | 3.600169 | 36.60 |
| 46,000 | 3.583047 | 3.624296 | 37.50 |
| 48,000 | 3.561008 | 3.583526 | 36.00 |
| 50,000 | 3.592763 | 3.598916 | 36.56 |
| 52,000 | 3.580003 | 3.557480 | 35.07 |
| 54,000 | 3.527637 | 3.568154 | 35.45 |
| 56,000 | 3.528134 | 3.560501 | 35.18 |
| 58,000 | 3.537106 | 3.587802 | 36.15 |
| 60,000 | 3.504482 | 3.517095 | 33.69 |

Although individual evaluations are noisy because each evaluation uses sampled batches, the overall trend remains downward.

### C. Training Efficiency Figures

The plotting script also produces hardware and optimization figures:

| Figure | File |
| --- | --- |
| Learning-rate schedule | `paper_content/figures/learning_rate_schedule.png` |
| Training throughput | `paper_content/figures/training_throughput.png` |
| VRAM usage | `paper_content/figures/vram_usage.png` |
| Gradient norm | `paper_content/figures/gradient_norm.png` |
| Combined dashboard | `paper_content/figures/training_dashboard.png` |

These figures support the systems-engineering claims of the paper: the project is not only a model implementation, but also a measured training pipeline with learning-rate control, memory tracking, gradient monitoring, and throughput logging.

### D. Token Exposure

At 60,000 steps:

$$
60{,}000 \times 7680 = 460{,}800{,}000
$$

token positions have been processed. Relative to the training set:

$$
\frac{460{,}800{,}000}{5{,}100{,}766{,}548}
\approx 0.0903
$$

Thus, the 60k-step run corresponds to about 9 percent of one token-equivalent pass through the training file. The model is still early relative to the dataset size.

### E. Qualitative Generation

For the prompt:

```text
how can i help
```

the model produced fluent continuation-style text by 40k and 60k steps. This behavior is expected for a base language model. It means the model has learned local English structure and web-text continuation patterns, but it does not mean the model is an instruction-following chatbot.

The base model learns:

$$
P(\text{next token} \mid \text{previous tokens})
$$

It does not learn:

$$
P(\text{assistant response} \mid \text{user request})
$$

unless the training data contains instruction-response examples.

## XI. Ablation Methodology

The model contains four ablation toggles:

| Toggle | Default | Enabled behavior | Disabled behavior |
| --- | --- | --- | --- |
| `USE_RMSNORM` | True | RMSNorm in blocks and final norm | Identity/no normalization |
| `USE_ROPE` | True | RoPE applied to Q and K | No positional encoding |
| `USE_FLASH_ATTENTION` | True | PyTorch causal SDPA | Manual attention |
| `USE_GQA` | True | 12 query heads, 4 KV heads | Full MHA with 12 KV heads |

A controlled ablation should use the same dataset, random seed, batch size, context length, model size, number of steps, evaluation interval, and generation prompts. Only one variable should change at a time.

### A. RMSNorm Ablation

With `USE_RMSNORM = False`, the block becomes:

$$
u^{(l)}=h^{(l)}+\operatorname{Attn}(h^{(l)})
$$

$$
h^{(l+1)}=u^{(l)}+\operatorname{FFN}(u^{(l)})
$$

Without normalization, activation scale can grow through residual additions:

$$
\|h^{(L)}\| \approx \|h^{(0)}\|+\sum_{l=0}^{L-1}\|f_l(h^{(l)})\|
$$

Expected effects include larger gradient norms, less stable loss, and possible NaN loss if the learning rate is too high.

### B. RoPE Ablation

With `USE_ROPE = False`, attention no longer receives explicit positional rotation. Since self-attention is permutation-equivariant by default, the model loses a direct way to distinguish token order. Expected effects include degraded grammar, poorer long-range ordering, and worse validation loss.

### C. Flash Attention Ablation

With `USE_FLASH_ATTENTION = False`, the model uses manual causal attention:

$$
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_h}}+M\right)V
$$

The mathematical operation is the same, so quality should be similar, but throughput and VRAM usage are expected to worsen.

### D. GQA Ablation

With `USE_GQA = False`, the model changes from:

$$
H_q=12,\quad H_{kv}=4
$$

to:

$$
H_q=12,\quad H_{kv}=12
$$

Key-value projection parameters increase from:

$$
393{,}216
$$

to:

$$
1{,}179{,}648
$$

per layer. This may improve capacity slightly but increases memory and compute.

### E. Ablation Table Template

| Variant | Val loss | PPL | Tok/s | VRAM MB | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Full model | fill after run | fill after run | fill after run | fill after run | stable |
| No RMSNorm | fill after run | fill after run | fill after run | fill after run | expected unstable |
| No RoPE | fill after run | fill after run | fill after run | fill after run | expected worse quality |
| No Flash Attention | fill after run | fill after run | fill after run | fill after run | expected slower |
| Full MHA | fill after run | fill after run | fill after run | fill after run | expected more memory |

Expected values should not be reported as measured values. The table should be filled only after controlled experiments.

## XII. Discussion

The experiment demonstrates that a modern small language model can be trained from scratch on consumer hardware when the system is designed around memory and throughput constraints. The 60,000-step checkpoint shows a large reduction in validation loss and perplexity, indicating that the model has learned meaningful statistical structure from the dataset.

The result also clarifies the difference between base pretraining and assistant behavior. A raw GPT-style model learns to continue text. It does not automatically learn to answer user requests. Therefore, outputs from prompts such as `how can i help` should be interpreted as continuations from the web-text distribution, not failures of instruction following. To obtain assistant-like behavior, a supervised instruction-tuning stage is required.

The current architecture reflects several efficiency decisions. GQA reduces key-value projection parameters and memory relative to full MHA. Flash Attention improves memory behavior for causal attention. RMSNorm stabilizes the residual stream. RoPE provides order information without learned absolute position embeddings. SwiGLU increases feed-forward expressiveness.

At 60,000 steps, the model has processed only about 9 percent of one token-equivalent pass over the 10 GiB training file. This suggests that further training may continue improving validation loss and sample quality.

## XIII. Limitations

The current system has several limitations:

- The model is not instruction-tuned.
- The context length is only 384 tokens.
- The latest observed run covers about 9 percent of one token-equivalent dataset pass.
- Generated text can be locally fluent but globally inconsistent.
- The model may hallucinate facts because it has no retrieval or grounding mechanism.
- Dataset source metadata and license information must be documented before publication.
- Ablation results are planned but not yet fully measured in the master table.
- The model is trained on a filtered web-text distribution, which may contain bias, boilerplate, and factual noise.

## XIV. Future Work

Future work should include:

- Continue training toward the planned 150,000 steps.
- Plot training loss, validation loss, perplexity, throughput, VRAM, and gradient norm from `logs/training_metrics.csv`.
- Run controlled ablations for RMSNorm, RoPE, Flash Attention, and GQA.
- Add instruction tuning using supervised dialogue or instruction-response examples.
- Evaluate generation using fixed prompts and human scoring.
- Experiment with larger context length if memory permits.
- Add top-p sampling and repetition controls to generation.
- Document the exact dataset source, version, and license.
- Export the final paper to IEEE LaTeX format.

## XV. Conclusion

This project presents a complete small language-model training pipeline built around a modern GPT-style decoder-only Transformer. The system includes efficient data preprocessing, byte-level BPE tokenization, memory-mapped training data, RoPE, RMSNorm, GQA, SwiGLU, Flash Attention, weight tying, checkpointing, validation metrics, and generation.

The active model has 117,787,392 trainable parameters and was trained on a 10 GiB tokenized subset containing more than 5.10B training tokens. At step 60,000, the model reached validation loss 3.517095 and perplexity 33.69, showing substantial improvement from initialization. Generated samples indicate successful learning of local syntax and continuation behavior, while also showing that instruction-following ability requires additional fine-tuning.

The work demonstrates that careful architecture and systems engineering can make language-model pretraining feasible on consumer hardware, and it provides a foundation for further research into efficient small language models.

## References

[1] A. Vaswani et al., "Attention Is All You Need," 2017.

[2] A. Radford et al., "Improving Language Understanding by Generative Pre-Training," 2018.

[3] A. Radford et al., "Language Models are Unsupervised Multitask Learners," 2019.

[4] T. Brown et al., "Language Models are Few-Shot Learners," 2020.

[5] J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021.

[6] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," 2019.

[7] N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019.

[8] J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," 2023.

[9] T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," 2022.

[10] N. Shazeer, "GLU Variants Improve Transformer," 2020.

[11] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," 2016.

[12] PyTorch documentation for `torch.nn`, `torch.optim.AdamW`, `torch.autocast`, and `torch.nn.functional.scaled_dot_product_attention`.

[13] HuggingFace `tokenizers` documentation.

[14] NumPy documentation for `numpy.memmap`.

[15] Apache Arrow and PyArrow documentation for parquet reading.

## Appendix A: Reproducibility Commands

Install dependencies:

```powershell
pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

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
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The future of AI is" --max-tokens 100
```

Evaluate perplexity:

```powershell
python -m evaluation.perplexity --checkpoint checkpoints/best_model.pt --split val --batches 50
```

Generate project report:

```powershell
$env:PYTHONIOENCODING='utf-8'
python project_report.py
```

## Appendix B: Current Artifact Inventory

| Artifact | Meaning |
| --- | --- |
| `train.bin` | Training token IDs stored as `uint16` |
| `val.bin` | Validation token IDs stored as `uint16` |
| `bpe_tokenizer_32k.json` | Byte-level BPE tokenizer |
| `checkpoints/ckpt_step_60000.pt` | Latest observed periodic checkpoint |
| `checkpoints/best_model.pt` | Best validation checkpoint |
| `logs/training_metrics.csv` | Step and evaluation metrics |
| `logs/samples/` | Periodic generated samples |
| `Prompt_Outputs/` | Saved prompt outputs by milestone |

## Appendix C: Base Model Versus Assistant Model

The current model is a base language model. Its objective is:

$$
\max_\theta \sum_t \log P_\theta(x_t \mid x_{<t})
$$

This objective teaches continuation. It does not teach the model to identify a user request and produce a helpful assistant response.

To train assistant behavior, a second-stage dataset should contain examples such as:

```text
User: how can i help?
Assistant: You can ask me to explain a topic, debug code, summarize text, or generate ideas.
```

The objective remains next-token prediction, but the distribution changes from general web text to instruction-response text:

$$
P(\text{assistant response} \mid \text{user instruction})
$$
