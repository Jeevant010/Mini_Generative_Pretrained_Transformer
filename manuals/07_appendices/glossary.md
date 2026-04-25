# Glossary — Key Terms & Abbreviations

## Architecture Terms

| Term | Definition |
|------|-----------|
| **Attention** | Mechanism allowing each token to dynamically weight the importance of all other tokens in the sequence. |
| **BPE** | Byte-Pair Encoding. Subword tokenization algorithm that iteratively merges the most frequent character/byte pairs. |
| **Causal Mask** | A lower-triangular mask applied in attention to prevent tokens from attending to future positions. |
| **Cross-Attention** | Attention where queries come from one sequence and keys/values from another. Used in encoder-decoder models. |
| **Decoder-Only** | Transformer architecture that uses only the decoder stack (no encoder). Standard for GPT-style models. |
| **Dropout** | Regularization technique that randomly zeros activations during training. |
| **Embedding** | Dense vector representation of a discrete token. |
| **FFN** | Feed-Forward Network. Two or three linear layers with a nonlinearity, applied per-position in Transformer blocks. |
| **GQA** | Grouped-Query Attention. Multiple query heads share a reduced number of key-value heads. |
| **LM Head** | Linear layer projecting hidden states to vocabulary logits for next-token prediction. |
| **MHA** | Multi-Head Attention. Standard attention with independent Q/K/V projections per head. |
| **MQA** | Multi-Query Attention. All query heads share a single key-value head. |
| **Pre-LN** | Pre-normalization. Applying layer normalization before (not after) each sub-layer. |
| **Residual Connection** | Adding the input of a sub-layer to its output: $x + f(x)$. Enables gradient flow in deep networks. |
| **RMSNorm** | Root Mean Square Normalization. Simplified LayerNorm that omits mean-centering. |
| **RoPE** | Rotary Positional Embedding. Encodes position by rotating Q/K vectors in 2D subspaces. |
| **SwiGLU** | Gated Linear Unit with SiLU activation. FFN variant using three weight matrices. |
| **Weight Tying** | Sharing the same weight matrix between token embedding and LM head. |

## Training Terms

| Term | Definition |
|------|-----------|
| **AdamW** | Adam optimizer with decoupled weight decay. |
| **bfloat16** | Brain float 16. 16-bit floating point with float32's dynamic range but reduced precision. |
| **Checkpoint** | Saved snapshot of model weights, optimizer state, and training step. |
| **Cosine Schedule** | Learning rate that decays following a cosine curve from peak to minimum. |
| **Cross-Entropy** | Loss function measuring the divergence between predicted token probabilities and true labels. |
| **Gradient Clipping** | Capping the global gradient norm to prevent training instability. |
| **Mixed Precision** | Using lower-precision (bf16) for forward/backward and full-precision (f32) for optimizer updates. |
| **Perplexity** | $e^{\text{cross-entropy loss}}$. Measures how "surprised" the model is by the data. Lower = better. |
| **Warmup** | Gradually increasing the learning rate from 0 to peak over the first N steps. |

## Data Terms

| Term | Definition |
|------|-----------|
| **memmap** | Memory-mapped file I/O. Accesses file contents as if they were in RAM without loading the full file. |
| **Parquet** | Columnar storage format. Efficient for large tabular datasets with compression. |
| **Token** | Atomic unit processed by the model. Can represent a word, subword, or byte sequence. |
| **uint16** | Unsigned 16-bit integer. Stores values 0–65,535. Used for token IDs (vocab ≤ 65K). |

## Abbreviations

| Abbreviation | Full Form |
|-------------|-----------|
| BPE | Byte-Pair Encoding |
| CUDA | Compute Unified Device Architecture |
| EOS | End of Sequence |
| BOS | Beginning of Sequence |
| FFN | Feed-Forward Network |
| FLOP(s) | Floating-Point Operation(s) |
| GQA | Grouped-Query Attention |
| GPU | Graphics Processing Unit |
| LLM | Large Language Model |
| LM | Language Model |
| LR | Learning Rate |
| MHA | Multi-Head Attention |
| MQA | Multi-Query Attention |
| OOM | Out of Memory |
| SGNS | Skip-Gram with Negative Sampling |
| TFLOPS | Tera Floating-Point Operations Per Second |
| VRAM | Video Random Access Memory |
