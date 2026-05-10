# Glossary

A plain-English dictionary of every technical term used in this project.

---

### A

**Ablation Study** — An experiment where you remove one component from a model to measure its contribution. Like removing one ingredient from a recipe to see how important it is.

**Activation Function** — A mathematical function applied inside the model that introduces non-linearity. Without it, the model could only learn straight-line relationships. Our model uses SiLU (SwiGLU).

**AdamW** — The optimizer we use. A smart version of gradient descent that remembers the history of updates and uses momentum. The "W" stands for weight decay.

**Attention** — The mechanism that lets each token "look at" previous tokens to decide what information is relevant. The core innovation of Transformers.

**Autoregressive** — Generating one token at a time, left to right, where each new token depends on all previous tokens. Like writing a sentence one word at a time.

### B

**Backpropagation** — The algorithm that calculates how each parameter in the model contributed to the error. It works backward from the output to the input.

**Batch** — A group of training examples processed together. Our batch size is 20, meaning 20 text chunks are processed simultaneously.

**Batch Size** — The number of examples in each batch. Larger batches give more stable gradient estimates but use more memory.

**BPE (Byte Pair Encoding)** — The tokenization algorithm that learns which character sequences to merge into single tokens.

**bfloat16** — A 16-bit floating-point format used during training for speed. Uses half the memory of 32-bit floats.

### C

**Causal Mask** — A mask that prevents the model from looking at future tokens. Essential for language modeling — you cannot peek at the answer.

**Checkpoint** — A saved snapshot of the model's state (parameters, optimizer, step number). Used to resume training after interruption.

**Context Window** — The maximum number of tokens the model can process at once. Ours is 384 tokens (~250 words).

**Cross-Entropy Loss** — The loss function used to train the model. Measures how different the model's predicted probability distribution is from the correct answer.

### D

**Decoder-Only** — A Transformer architecture that only generates output (no separate encoder). Used by GPT, LLaMA, Claude, and our model.

**Degeneration** — When a language model gets stuck producing repetitive or nonsensical text. Caused by overconfidence or bad data.

**Distinct-N** — A metric measuring lexical diversity. The fraction of unique n-grams in generated text. Higher = more diverse.

**DPO (Direct Preference Optimization)** — A post-training technique that teaches the model to prefer better responses over worse ones.

**Dropout** — Randomly setting some values to zero during training. Prevents overfitting by forcing the model to not rely on any single feature.

### E

**Embedding** — A dense vector representation of a token. Each of our 32,000 tokens has a 768-dimensional embedding.

**Entropy** — A measure of uncertainty in a probability distribution. High entropy = uncertain. Low entropy = confident.

**Epoch** — One complete pass through the entire training dataset.

### F

**Feed-Forward Layer** — The part of a Transformer block that processes information after attention. Our model uses SwiGLU.

**Fine-Tuning** — Taking a pre-trained model and training it further on a specific task or dataset.

**Flash Attention** — An optimized attention implementation that produces the same result using less memory.

### G

**Gradient** — The direction and magnitude of change needed for each parameter to reduce the loss. Calculated via backpropagation.

**Gradient Clipping** — Limiting the maximum gradient magnitude to prevent destructive updates.

**GQA (Grouped-Query Attention)** — An attention variant that shares Key-Value heads across multiple Query heads to save memory.

### H

**Hallucination** — When a model generates plausible-sounding but factually incorrect information. All current language models do this.

**Head (Attention Head)** — One independent attention mechanism. Our model has 12 query heads and 4 KV heads.

**Hyperparameter** — A setting that controls training but is not learned by the model (learning rate, batch size, number of layers, etc.).

### L

**Learning Rate** — How much the model adjusts its parameters at each step. Too high = instability. Too low = slow learning. Ours starts at 3e-4 and decays to 3e-5.

**Loss** — A number measuring how wrong the model's predictions are. Lower = better.

**LM Head** — The final layer that converts the model's internal representation into probabilities over all 32,000 tokens.

### M

**Memory Mapping (memmap)** — Loading a file from disk on-demand instead of reading the entire file into RAM. We use this for our 9.5 GB training file.

**Mixed Precision** — Using lower-precision numbers (bfloat16) for speed while keeping critical calculations in full precision (float32).

### N

**N-gram** — A sequence of N consecutive tokens. Unigram (1), bigram (2), trigram (3).

**NaN (Not a Number)** — A numerical error that crashes training. Usually caused by numbers becoming too large.

### O

**OpenWebText** — An open-source dataset of web pages, recreating the dataset used to train GPT-2. Our training data source.

**Optimizer** — The algorithm that updates model parameters based on gradients. We use AdamW.

**Overfitting** — When the model memorizes the training data instead of learning general patterns. Detected when training loss decreases but validation loss increases.

### P

**Parameter** — A single learnable number in the model. Our model has 118 million parameters.

**Perplexity (PPL)** — A metric measuring how well the model predicts text. Equal to e^loss. Lower = better.

**Pre-training** — The initial training phase where the model learns general language patterns from a large text corpus.

### R

**Residual Connection** — Adding the input of a layer to its output. Creates a direct path for information to flow through the network.

**Repetition Penalty** — A generation-time technique that reduces the probability of tokens that have already appeared, preventing loops.

**RMSNorm** — Root Mean Square Normalization. Keeps activations in a stable range without subtracting the mean.

**RoPE** — Rotary Positional Embedding. Encodes token position by rotating query and key vectors.

### S

**Sampling** — The process of choosing the next token from the model's probability distribution. Methods include greedy, top-k, top-p (nucleus).

**Self-BLEU** — A metric measuring how similar different generated samples are to each other. Lower = more diverse.

**SFT (Supervised Fine-Tuning)** — Training on instruction-response pairs to make the model follow instructions.

**SwiGLU** — The gated feed-forward layer used in our model. Combines SiLU activation with gating.

### T

**Temperature** — A parameter that controls how "random" the model's output is. Low = deterministic. High = random.

**Token** — A piece of text (word, subword, or character) that the model processes as a single unit.

**Tokenizer** — The component that converts text to tokens and back.

**Top-k Sampling** — Restricting the model to choose from the top k most likely tokens.

**Top-p (Nucleus) Sampling** — Restricting the model to the smallest set of tokens whose total probability exceeds p.

**Transformer** — The neural network architecture used in this project. Based on the "Attention Is All You Need" paper (2017).

### V

**Validation Loss** — Loss computed on data the model has never seen during training. Used to detect overfitting.

**VRAM** — Video RAM. The GPU's memory. Our RTX 4060 has 8 GB.

### W

**Warmup** — Gradually increasing the learning rate at the start of training to prevent instability.

**Weight Decay** — A regularization technique that gently pushes parameters toward zero, preventing them from growing too large.

**Weight Tying** — Sharing the same parameters between the input embedding and the output projection.
