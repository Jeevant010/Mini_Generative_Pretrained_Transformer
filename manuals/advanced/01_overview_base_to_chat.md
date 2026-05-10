# From Base Language Model to Conversational AI

## The Four Stages of Building a Modern LLM

This manual explains the complete pipeline that transforms a raw next-word predictor into a model that can hold conversations, follow instructions, and behave helpfully. Understanding this pipeline is essential because it explains exactly why the current model cannot "talk" — and what steps are needed to get there.

## What the Base Model Actually Learns

The model trained in this project learns a conditional probability distribution:

$$
P_\theta(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

This means: given a sequence of tokens, predict the next one. The model was trained on web text (OpenWebText), so it learned to continue web text. When you give it a prompt like "How can I help", it does not interpret this as a question directed at it. Instead, it treats the prompt as the beginning of a web document and generates what typically follows such text on the internet.

This is why the model's outputs often read like news articles, blog posts, or forum snippets — because that is what the training data contains.

## Stage 1: Pre-Training (Where We Are Now)

### Objective

Learn general language patterns from a large corpus of unlabeled text.

### What the Model Learns

- Vocabulary and spelling
- Grammar and syntax
- Basic facts and world knowledge
- Writing styles from the training corpus
- Next-token prediction at a statistical level

### What the Model Does NOT Learn

- That it should answer questions
- That it should be helpful, harmless, or honest
- That it has a role as an "assistant"
- How to follow specific instructions
- When to stop generating

### Mathematical Objective

Minimize cross-entropy loss over the training corpus $D$:

$$
\mathcal{L}_{\text{PT}} = -\mathbb{E}_{x \sim D} \left[ \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t}) \right]
$$

### Current Status

The model is at this stage. It has learned meaningful language patterns (PPL dropped from 37,780 to ~37), can generate grammatically correct sentences, and shows topic awareness. But it cannot converse.

---

## Stage 2: Supervised Fine-Tuning (SFT)

### Objective

Teach the model to follow instructions by training on labeled (instruction, response) pairs.

### What Changes

The model learns a new data format:

```
### Instruction: Explain photosynthesis in simple terms.
### Response: Photosynthesis is the process by which plants use sunlight...
```

After SFT, the model understands that when it sees an instruction prompt, it should generate a helpful response — not continue writing a web article.

### Mathematical Objective

Fine-tune on a dataset of instruction-response pairs $D_{\text{SFT}} = \{(\text{instruction}_i, \text{response}_i)\}$:

$$
\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim D_{\text{SFT}}} \left[ \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t}) \right]
$$

Note: the loss is computed only on the response tokens $y$, not the instruction tokens $x$. The model should learn to generate responses, not to predict the instruction itself.

### Key Datasets for SFT

| Dataset | Size | Source | License |
|---|---|---|---|
| Stanford Alpaca | 52K pairs | GPT-3.5 generated | Apache 2.0 |
| Databricks Dolly | 15K pairs | Human written | Open |
| OpenAssistant (oasst1) | 161K messages | Human conversation trees | Apache 2.0 |
| LIMA | 1K pairs | Curated high-quality | Research |

### SFT Hyperparameters (for an 85M model)

| Parameter | Recommended Value | Rationale |
|---|---|---|
| Learning rate | 1e-5 to 5e-5 | Much lower than pre-training to preserve learned knowledge |
| Epochs | 3–5 | Small instruction datasets overfit quickly |
| Batch size | 4–8 | Small batches for fine-grained gradient updates |
| Warmup | 100 steps | Brief warmup, training is short |
| Weight decay | 0.01 | Light regularization |

### What SFT Produces

After SFT, the model can:
- Answer direct questions
- Follow simple instructions
- Generate text in the instruction-response format
- Produce more focused, on-topic responses

After SFT, the model still cannot:
- Reliably refuse harmful requests
- Prefer helpful answers over plausible-sounding wrong ones
- Maintain consistent behavior across diverse prompts

This is where preference alignment comes in.

---

## Stage 3: Reward Modeling + RLHF (or DPO)

### The Problem SFT Doesn't Solve

SFT teaches the model to produce responses that look like the training data. But there are many possible responses to any instruction, and SFT doesn't teach the model which responses are better than others.

Example: "Write a poem about summer"

- Response A (SFT): Produces a generic poem (technically correct)
- Response B (with preference alignment): Produces a creative, engaging poem (preferred by humans)

### Option A: RLHF (Reinforcement Learning from Human Feedback)

#### Step 1: Collect Human Preferences

Present pairs of model responses to human annotators. For each pair, the annotator marks which response is better.

$$
D_{\text{pref}} = \{(x_i, y_w^{(i)}, y_l^{(i)})\}_{i=1}^{N}
$$

Where $y_w$ is the preferred (winning) response and $y_l$ is the rejected (losing) response.

#### Step 2: Train a Reward Model

Train a separate model $R_\phi$ to predict human preferences:

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l)) \right]
$$

This model learns to assign higher scores to preferred responses.

#### Step 3: Optimize with PPO

Use Proximal Policy Optimization to maximize the reward while staying close to the SFT model:

$$
\mathcal{L}_{\text{RLHF}} = -\mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[ R_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{SFT}}) \right]
$$

The KL divergence term $D_{\text{KL}}$ prevents the model from deviating too far from the SFT baseline, which would cause incoherent text.

#### Practical Complexity

RLHF requires:
- A separate reward model
- PPO optimization (complex, sensitive to hyperparameters)
- Significant compute (generating samples during training)
- Careful tuning to avoid reward hacking

### Option B: DPO (Direct Preference Optimization) — Recommended

DPO eliminates the reward model entirely by proving that the optimal policy can be derived directly from preference data.

#### DPO Loss Function

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

Where:
- $\pi_\theta$ is the policy model being trained
- $\pi_{\text{ref}}$ is a frozen copy of the SFT model (reference)
- $\beta$ controls how much the model can deviate from the reference
- $y_w$ is the preferred response, $y_l$ is the rejected response

#### Why DPO is Better for This Project

| Aspect | RLHF | DPO |
|---|---|---|
| Needs reward model | Yes | No |
| Training stability | Fragile (PPO) | Stable (supervised) |
| Compute cost | High (sampling during training) | Moderate (same as SFT) |
| Implementation complexity | Very high | Moderate |
| Quality of results | Excellent | Comparable to RLHF |

---

## Stage 4: System Prompts and Chat Templates

### Purpose

After preference alignment, the model can follow instructions and produce preferred responses. The final step is structuring the interaction format so the model knows:

1. What its role is (system prompt)
2. Where the user's message starts and ends
3. Where its response should begin

### Chat Template Format

```text
<|system|>You are a helpful AI assistant. Answer questions clearly and concisely.<|end|>
<|user|>What is machine learning?<|end|>
<|assistant|>Machine learning is a branch of artificial intelligence...<|end|>
```

### Special Tokens Required

The tokenizer must be extended with new special tokens:

| Token | Purpose |
|---|---|
| `<\|system\|>` | Marks the start of system instructions |
| `<\|user\|>` | Marks user messages |
| `<\|assistant\|>` | Marks assistant responses |
| `<\|end\|>` | Marks the end of any section |

### Multi-Turn Conversations

For multi-turn dialogue, messages are concatenated:

```text
<|system|>You are helpful.<|end|>
<|user|>What is 2+2?<|end|>
<|assistant|>4<|end|>
<|user|>And 3+3?<|end|>
<|assistant|>6<|end|>
```

The model learns to attend to the full conversation history when generating its response.

---

## The Complete Pipeline Visualized

```
Pre-Training (600B+ tokens of web text)
    → Base LM: can continue any text
    
    ↓ + Instruction-response pairs (50K–200K examples)

SFT (Supervised Fine-Tuning)
    → Instruction-following model: can answer questions
    
    ↓ + Human preference judgments (50K–100K comparisons)

DPO / RLHF (Preference Alignment)
    → Aligned model: prefers helpful, harmless responses
    
    ↓ + Chat template and system prompt

Deployment
    → Conversational AI: understands roles, maintains context
```

## What This Means for the Current Project

The model is at Stage 1. The immediate next steps are:

1. Complete base training with good generation quality (PPL < 25, no degeneration)
2. Prepare an SFT dataset (start with Dolly 15K — it is open license and human-written)
3. Modify the training loop to handle instruction-response format
4. Fine-tune for 3–5 epochs
5. Evaluate using the quality metrics framework

SFT alone will make the model significantly more useful. DPO can be added after SFT if further quality improvement is needed.

## References

- Ouyang et al. (2022). "Training language models to follow instructions with human feedback." (InstructGPT / RLHF)
- Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." (DPO)
- Touvron et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models."
- Zhou et al. (2023). "LIMA: Less Is More for Alignment." (Shows SFT quality > quantity)
- Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla scaling laws)
