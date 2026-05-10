# References

## Foundational Papers

### The Transformer
- Vaswani, A., et al. (2017). **"Attention Is All You Need."** NeurIPS.
  - The original paper that introduced the Transformer architecture.
  - Key idea: Replace recurrence with self-attention for sequence processing.

### GPT Series
- Radford, A., et al. (2018). **"Improving Language Understanding by Generative Pre-Training."** (GPT-1)
  - First demonstration that pre-training a Transformer on text and then fine-tuning works well.

- Radford, A., et al. (2019). **"Language Models are Unsupervised Multitask Learners."** (GPT-2)
  - Scaled up GPT-1 and showed that larger models learn to perform tasks without explicit fine-tuning.
  - Our model architecture is closest to GPT-2 Small.

## Architecture Components

### Rotary Positional Embeddings (RoPE)
- Su, J., et al. (2021). **"RoFormer: Enhanced Transformer with Rotary Position Embedding."**
  - Encodes position through rotation of query and key vectors.
  - Used in LLaMA, PaLM, and our model.

### Grouped-Query Attention (GQA)
- Ainslie, J., et al. (2023). **"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints."**
  - Shares key-value heads across query groups to reduce memory.
  - Used in LLaMA 2 and our model.

### SwiGLU
- Shazeer, N. (2020). **"GLU Variants Improve Transformer."**
  - Introduces gated feed-forward variants including SwiGLU.
  - Used in PaLM, LLaMA, and our model.

### RMSNorm
- Zhang, B. & Sennrich, R. (2019). **"Root Mean Square Layer Normalization."**
  - A simpler normalization that skips mean subtraction.
  - Used in LLaMA and our model.

## Training Techniques

### AdamW Optimizer
- Loshchilov, I. & Hutter, F. (2019). **"Decoupled Weight Decay Regularization."**
  - Fixes weight decay interaction with Adam optimizer.

### Cosine Learning Rate Schedule
- Loshchilov, I. & Hutter, F. (2017). **"SGDR: Stochastic Gradient Descent with Warm Restarts."**
  - Cosine annealing schedule for learning rate.

### Flash Attention
- Dao, T., et al. (2022). **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."**
  - Memory-efficient attention that avoids materializing the full attention matrix.

## Scaling Laws

### Chinchilla Scaling
- Hoffmann, J., et al. (2022). **"Training Compute-Optimal Large Language Models."** (Chinchilla)
  - Establishes that optimal training uses ~20 tokens per parameter.

## Post-Training (For Next Branch)

### Supervised Fine-Tuning
- Conover, M., et al. (2023). **"Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM."** (Dolly 15K dataset)

### DPO
- Rafailov, R., et al. (2023). **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model."**
  - Alignment without a separate reward model.

### Text Degeneration
- Holtzman, A., et al. (2020). **"The Curious Case of Neural Text Degeneration."**
  - Analysis of repetition loops and nucleus sampling solution.

## Datasets

### OpenWebText
- Gokaslan, A. & Cohen, V. (2019). **"OpenWebText Corpus."**
  - Open-source recreation of GPT-2's WebText dataset.

## Evaluation

### Distinct-N
- Li, J., et al. (2016). **"A Diversity-Promoting Objective Function for Neural Conversation Models."**

### Self-BLEU
- Zhu, Y., et al. (2018). **"Texygen: A Benchmarking Platform for Text Generation Models."**

### lm-evaluation-harness
- Gao, L., et al. (2021). **"A Framework for Few-Shot Language Model Evaluation."**
  - Standardized evaluation benchmarks (HellaSwag, ARC, LAMBADA, etc.)
