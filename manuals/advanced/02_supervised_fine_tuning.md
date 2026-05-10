# Supervised Fine-Tuning (SFT)

## Overview

Supervised Fine-Tuning is the process of taking a pre-trained base language model and training it further on labeled instruction-response data. After SFT, the model understands that when it receives an instruction, it should generate a helpful response — not simply continue generating web text.

This manual provides a complete guide for applying SFT to the Mini GPT model, including data preparation, code modifications, training procedures, and evaluation.

## Prerequisites

Before starting SFT:

1. The base model should be fully trained with stable validation loss
2. You should have the best checkpoint saved (`checkpoints/best_model.pt`)
3. The tokenizer (`bpe_tokenizer_32k.json`) must be available
4. Recommended: base model PPL < 30 on the validation set

## Data Format

### Alpaca Format

The standard format for instruction-tuning data uses three fields:

```json
{
    "instruction": "Explain the concept of gravity.",
    "input": "",
    "output": "Gravity is a fundamental force of nature..."
}
```

With optional context:

```json
{
    "instruction": "Summarize the following text.",
    "input": "The Industrial Revolution was a period...",
    "output": "The Industrial Revolution transformed manufacturing..."
}
```

### Converting to Training Sequences

Each example is converted to a single text sequence for training:

```
<bos>### Instruction:
Explain the concept of gravity.

### Response:
Gravity is a fundamental force of nature that attracts objects with mass toward each other. On Earth, it gives objects weight and causes them to fall toward the ground when dropped.<eos>
```

When the `input` field is present:

```
<bos>### Instruction:
Summarize the following text.

### Input:
The Industrial Revolution was a period of rapid industrialization...

### Response:
The Industrial Revolution transformed manufacturing from hand production to machine-based processes.<eos>
```

### Loss Masking

During SFT, the loss should only be computed on the **response tokens**, not the instruction tokens. This is critical because:

- The model should learn to generate responses, not to predict instructions
- Computing loss on instructions wastes gradient updates on text the model already handles
- It prevents the model from learning to generate instructions as continuations

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|} \sum_{t=\text{response\_start}}^{|y|} \log P_\theta(y_t \mid x, y_{<t})
$$

Where $x$ represents instruction tokens and $y$ represents response tokens.

## Recommended Datasets

### Tier 1: Start Here

| Dataset | Size | Quality | License | Best For |
|---|---|---|---|---|
| **Databricks Dolly** | 15K | High (human-written) | Open | First SFT experiment |
| **Stanford Alpaca** | 52K | Good (GPT-3.5 generated) | Apache 2.0 | Broader instruction coverage |

### Tier 2: For More Diversity

| Dataset | Size | Quality | License | Best For |
|---|---|---|---|---|
| **OpenAssistant (oasst1)** | 161K messages | Mixed | Apache 2.0 | Multi-turn conversations |
| **LIMA** | 1K | Very high (curated) | Research | Quality-focused fine-tuning |
| **Evol-Instruct** | 250K | Good | Apache 2.0 | Complex instructions |

### Recommendation for This Project

Start with **Dolly 15K** because:
- It is human-written (no synthetic data artifacts)
- It is openly licensed
- 15K examples is enough for an 85M model without severe overfitting
- It covers diverse categories: open QA, closed QA, brainstorming, classification, summarization, etc.

## Implementation Guide

### Step 1: Download the Dataset

```python
# download_sft_data.py
"""Download Dolly 15K dataset for SFT."""

import json
import os

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets")
    exit(1)

def download_dolly():
    """Download and save Dolly 15K in Alpaca format."""
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    examples = []
    for item in ds:
        examples.append({
            "instruction": item["instruction"],
            "input": item.get("context", ""),
            "output": item["response"],
            "category": item.get("category", ""),
        })

    os.makedirs("data", exist_ok=True)
    with open("data/dolly_15k.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(examples)} examples to data/dolly_15k.json")


if __name__ == "__main__":
    download_dolly()
```

### Step 2: Create the SFT Dataset Loader

```python
# sft_dataset.py
"""Dataset loader for instruction-response SFT training."""

import json
import random
import torch
import numpy as np
from tokenizer import BytePairTokenizer
import config


class SFTDataset:
    """
    Converts instruction-response pairs into token sequences with loss masks.
    """

    TEMPLATE_WITH_INPUT = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )

    TEMPLATE_NO_INPUT = (
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}"
    )

    def __init__(self, data_path: str, tokenizer: BytePairTokenizer,
                 max_length: int = None, val_fraction: float = 0.05):
        self.tokenizer = tokenizer
        self.max_length = max_length or config.block_size

        with open(data_path, "r", encoding="utf-8") as f:
            all_examples = json.load(f)

        # Shuffle and split
        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * (1 - val_fraction))
        self.train_examples = all_examples[:split_idx]
        self.val_examples = all_examples[split_idx:]

        print(f"SFT dataset: {len(self.train_examples)} train, "
              f"{len(self.val_examples)} val examples")

    def _format_example(self, example: dict) -> tuple:
        """Format and tokenize a single example. Returns (input_ids, label_ids)."""
        if example.get("input", "").strip():
            text = self.TEMPLATE_WITH_INPUT.format(**example)
        else:
            text = self.TEMPLATE_NO_INPUT.format(**example)

        # Find where the response starts
        response_marker = "### Response:\n"
        response_start = text.find(response_marker)
        instruction_text = text[:response_start + len(response_marker)]
        response_text = text[response_start + len(response_marker):]

        instruction_ids = self.tokenizer.encode(instruction_text, add_bos=True)
        response_ids = self.tokenizer.encode(response_text, add_eos=True)

        input_ids = instruction_ids + response_ids

        # Create labels: -100 for instruction tokens (ignored in loss), 
        # actual token IDs for response tokens
        labels = [-100] * len(instruction_ids) + response_ids

        # Truncate to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return input_ids, labels

    def get_batch(self, split: str = "train", batch_size: int = None):
        """Sample a batch of (input_ids, labels) pairs."""
        if batch_size is None:
            batch_size = config.batch_size

        examples = self.train_examples if split == "train" else self.val_examples
        batch_indices = random.sample(range(len(examples)), min(batch_size, len(examples)))

        batch_inputs = []
        batch_labels = []
        max_len = 0

        for idx in batch_indices:
            input_ids, labels = self._format_example(examples[idx])
            batch_inputs.append(input_ids)
            batch_labels.append(labels)
            max_len = max(max_len, len(input_ids))

        # Pad to max length in batch
        pad_id = self.tokenizer.special_to_id.get("<pad>", 0)
        for i in range(len(batch_inputs)):
            pad_len = max_len - len(batch_inputs[i])
            batch_inputs[i] = batch_inputs[i] + [pad_id] * pad_len
            batch_labels[i] = batch_labels[i] + [-100] * pad_len

        x = torch.tensor(batch_inputs, dtype=torch.long).to(config.device)
        y = torch.tensor(batch_labels, dtype=torch.long).to(config.device)

        return x, y
```

### Step 3: Create the SFT Training Script

```python
# sft_train.py
"""Supervised Fine-Tuning training loop."""

import os
import time
import math
import torch
import torch.nn.functional as F

import config
from model import GPTLanguageModel
from sft_dataset import SFTDataset
from tokenizer import BytePairTokenizer


def sft_loss(logits, labels):
    """
    Cross-entropy loss with label masking.
    Labels of -100 are ignored (instruction tokens).
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=0.1,
    )


def train_sft():
    device = config.device
    print(f"Starting SFT on {device}...")

    # Load tokenizer
    tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)

    # Load base model from best pre-training checkpoint
    model = GPTLanguageModel(config).to(device)
    base_ckpt = torch.load(
        "checkpoints/best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(base_ckpt["model_state_dict"])
    print(f"Loaded base model from step {base_ckpt.get('step', '?')}")

    # SFT hyperparameters
    sft_lr = 2e-5
    sft_epochs = 3
    sft_batch_size = 4
    eval_interval = 100
    save_interval = 500

    # Load SFT dataset
    dataset = SFTDataset(
        "data/dolly_15k.json", tokenizer,
        max_length=config.block_size
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=sft_lr, weight_decay=0.01
    )

    # Training
    os.makedirs("checkpoints/sft", exist_ok=True)
    model.train()
    step = 0
    best_val_loss = float("inf")

    steps_per_epoch = len(dataset.train_examples) // sft_batch_size

    for epoch in range(sft_epochs):
        print(f"\n--- Epoch {epoch + 1}/{sft_epochs} ---")
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx in range(steps_per_epoch):
            t0 = time.perf_counter()

            xb, yb = dataset.get_batch("train", sft_batch_size)

            with torch.autocast(
                device_type="cuda" if "cuda" in str(device) else "cpu",
                dtype=torch.bfloat16,
            ):
                logits, _ = model(xb)
                loss = sft_loss(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dt = time.perf_counter() - t0
            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1

            if step % 10 == 0:
                print(
                    f"  Step {step:4d} | Loss: {loss.item():.4f} | "
                    f"{dt*1000:.0f}ms/step"
                )

            # Evaluation
            if step % eval_interval == 0:
                model.eval()
                val_losses = []
                for _ in range(20):
                    vx, vy = dataset.get_batch("val", sft_batch_size)
                    with torch.no_grad():
                        vlogits, _ = model(vx)
                        vloss = sft_loss(vlogits, vy)
                    val_losses.append(vloss.item())
                avg_val = sum(val_losses) / len(val_losses)
                print(f"  >>> Eval: val_loss = {avg_val:.4f}")

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save({
                        "step": step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val,
                    }, "checkpoints/sft/best_sft_model.pt")
                    print(f"  New best SFT model saved!")

                model.train()

            # Periodic save
            if step % save_interval == 0:
                torch.save({
                    "step": step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, f"checkpoints/sft/sft_step_{step}.pt")

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1} complete | Avg loss: {avg_epoch_loss:.4f}")

    print("\nSFT Training Complete!")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_sft()
```

### Step 4: Generate Text After SFT

```python
# After SFT, use the same generate.py but with the SFT checkpoint:
# python generate.py --checkpoint checkpoints/sft/best_sft_model.pt \
#     --prompt "### Instruction:\nExplain what gravity is.\n\n### Response:\n" \
#     --max-tokens 100
```

## Parameter-Efficient Fine-Tuning (LoRA)

### Why LoRA?

Full fine-tuning updates all 85M parameters. LoRA (Low-Rank Adaptation) freezes the base model and adds small trainable matrices, reducing the number of trainable parameters by 90%+.

### How LoRA Works

For each weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA adds a low-rank update:

$$
W' = W + \Delta W = W + BA
$$

Where:
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$ is the rank (typically 8–64)
- Only $B$ and $A$ are trained; $W$ is frozen

### LoRA Benefits

| Aspect | Full Fine-Tuning | LoRA (rank 16) |
|---|---|---|
| Trainable parameters | 85M (100%) | ~1.5M (~1.8%) |
| GPU memory | Full model in optimizer | Minimal optimizer state |
| Training speed | Baseline | ~1.5× faster |
| Risk of catastrophic forgetting | Higher | Lower |
| Multiple task variants | Requires full model copies | Only small adapter files |

### Implementation Note

For the current 85M model on an RTX 4060, full SFT is feasible and recommended for simplicity. LoRA becomes important when:
- The model grows beyond GPU memory for full fine-tuning
- You want to maintain multiple fine-tuned versions (one base model, multiple adapters)
- You want to reduce the risk of catastrophic forgetting

## Evaluating After SFT

### Qualitative Checks

Generate responses to test instructions:

```
Instruction: "What is the capital of France?"
Expected: "The capital of France is Paris."

Instruction: "Write a haiku about winter."
Expected: A 5-7-5 syllable poem about winter.

Instruction: "Explain what a neural network is in simple terms."
Expected: A clear, simple explanation.
```

### Quantitative Metrics

1. **SFT validation loss** — should decrease and plateau
2. **Instruction-following accuracy** — what fraction of responses actually address the instruction
3. **Response quality** — use Distinct-N and Self-BLEU from the generation quality metrics manual
4. **Base model regression** — re-check PPL on the original validation set to ensure the model hasn't forgotten its language understanding

## Common Pitfalls

### 1. Catastrophic Forgetting

If the SFT learning rate is too high or training runs too long, the model "forgets" what it learned during pre-training. Signs:
- PPL on the original validation set increases significantly
- Generated text becomes incoherent outside the instruction format
- The model can only respond in the SFT template format

Prevention:
- Use a low learning rate (1e-5 to 5e-5)
- Train for only 3–5 epochs
- Monitor base PPL alongside SFT loss

### 2. Overfitting to SFT Data

With only 15K examples, the model can memorize responses. Signs:
- Training loss drops to near zero but validation loss increases
- Responses to unseen instructions are poor
- Model outputs verbatim training examples

Prevention:
- Use dropout (already in the model at 0.1)
- Stop training when validation loss plateaus
- Use a diverse instruction dataset

### 3. Format Rigidity

The model learns the exact template format and may not respond to slightly different instruction formats. Prevention:
- Include format variations in training data
- Test with multiple prompt formats during evaluation

## References

- Wei et al. (2022). "Finetuned Language Models Are Zero-Shot Learners." (FLAN — instruction tuning at scale)
- Hu et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models."
- Conover et al. (2023). "Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM."
- Taori et al. (2023). "Stanford Alpaca: An Instruction-following LLaMA Model."
- Zhou et al. (2023). "LIMA: Less Is More for Alignment."
