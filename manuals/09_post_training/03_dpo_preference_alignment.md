# DPO: Direct Preference Optimization

## Overview

Direct Preference Optimization (DPO) is a technique for aligning language models with human preferences without training a separate reward model. It is the recommended preference alignment method for this project because it is simpler, more stable, and more compute-efficient than RLHF.

DPO was introduced in Rafailov et al. (2023) with the key insight: the optimal RLHF policy can be expressed in closed form, which allows you to skip the reward model entirely and directly optimize on preference data.

## When to Use DPO

DPO should be applied **after SFT**, not after pre-training. The pipeline is:

```
Pre-trained base model → SFT → DPO → Deployment
```

If you skip SFT and apply DPO directly to the base model, results will be poor because the base model does not understand the instruction-response format that preference data assumes.

## The DPO Loss Function

### Intuition

Given a prompt $x$ and two responses:
- $y_w$ (the preferred/winning response)
- $y_l$ (the rejected/losing response)

DPO increases the probability of $y_w$ and decreases the probability of $y_l$, relative to a frozen reference model $\pi_{\text{ref}}$.

### Mathematical Formulation

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right) \right]
$$

Where:
- $\pi_\theta$: the policy model being trained
- $\pi_{\text{ref}}$: a frozen copy of the SFT model (reference baseline)
- $\sigma$: the sigmoid function
- $\beta$: temperature parameter controlling deviation from reference

### Breaking Down the Loss

Define the implicit reward as:

$$
r(x, y) = \beta \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)}
$$

Then the loss becomes:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right]
$$

This is the Bradley-Terry ranking loss: the model is trained so that the implicit reward of the preferred response exceeds that of the rejected response.

### The Role of $\beta$

| $\beta$ Value | Effect |
|---|---|
| **Low** (0.01–0.05) | Model can deviate far from reference → aggressive optimization |
| **Medium** (0.1) | Standard starting point → balanced |
| **High** (0.5–1.0) | Model stays very close to reference → conservative |

Recommended starting value: **$\beta = 0.1$**

## Preference Dataset Format

### Structure

Each example contains a prompt and two responses:

```json
{
    "prompt": "### Instruction:\nExplain what machine learning is.\n\n### Response:\n",
    "chosen": "Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed. It works by using statistical algorithms to find patterns in training data and then applying those patterns to make predictions on new data.",
    "rejected": "Machine learning is when computers think. It's basically AI. Computers are smart now."
}
```

### Key Principles for Preference Data

1. **Same prompt, different quality responses**: Both responses should address the same instruction, but `chosen` should be clearly better.

2. **In-distribution**: The prompts should be similar to what the model saw during SFT. Out-of-distribution prompts make the optimization unstable.

3. **Meaningful differences**: The `chosen` and `rejected` responses should differ in specific quality dimensions:
   - Accuracy (correct vs incorrect facts)
   - Completeness (thorough vs superficial)
   - Helpfulness (addresses the question vs generic response)
   - Safety (refuses harmful requests vs complies)

### Available Preference Datasets

| Dataset | Size | Source | License |
|---|---|---|---|
| **Anthropic HH-RLHF** | 170K pairs | Human + model generated | MIT |
| **OpenAssistant Preference** | 10K pairs | Human rankings | Apache 2.0 |
| **UltraFeedback** | 64K pairs | GPT-4 judged | MIT |

### Creating Preference Data from SFT Outputs

If pre-made preference datasets do not match your domain, you can generate preference pairs from your own SFT model:

1. Generate multiple responses per prompt from your SFT model
2. Rank them manually (or using a stronger model as judge)
3. Save the best and worst as `chosen` and `rejected`

```python
# Generating preference pairs from SFT model
def generate_preference_pairs(model, tokenizer, prompts, n_responses=4):
    """Generate multiple responses per prompt for human ranking."""
    pairs = []
    for prompt in prompts:
        responses = []
        for _ in range(n_responses):
            # Generate with different temperatures for diversity
            input_ids = tokenizer.encode(prompt, add_bos=True)
            idx = torch.tensor([input_ids], dtype=torch.long, device="cuda")
            output = model.generate(idx, max_new_tokens=200,
                                     temperature=0.9, top_k=50)
            text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            responses.append(text)
        
        pairs.append({
            "prompt": prompt,
            "responses": responses,
            # Human annotator will mark chosen/rejected
        })
    return pairs
```

## Implementation Guide

### Step 1: Compute Log Probabilities

The core computation in DPO is calculating the log-probability of a response given a prompt:

$$
\log \pi_\theta(y | x) = \sum_{t=1}^{|y|} \log P_\theta(y_t | x, y_{<t})
$$

```python
@torch.no_grad()
def compute_log_probs(model, input_ids, labels):
    """
    Compute per-token log probabilities for the response portion.
    
    Args:
        model: GPTLanguageModel instance.
        input_ids: Full sequence [batch, seq_len].
        labels: Token IDs for loss computation, -100 for instruction tokens.
    
    Returns:
        Total log probability for each example in the batch [batch].
    """
    logits, _ = model(input_ids)
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Per-token log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask out instruction tokens (where labels == -100)
    mask = (shift_labels != -100).float()
    token_log_probs = token_log_probs * mask
    
    # Sum log probs per example
    return token_log_probs.sum(dim=-1)
```

### Step 2: DPO Training Loop

```python
# dpo_train.py
"""Direct Preference Optimization training loop."""

import os
import json
import copy
import time
import torch
import torch.nn.functional as F

import config
from model import GPTLanguageModel
from tokenizer import BytePairTokenizer


class DPODataset:
    """Load and batch preference pairs for DPO training."""

    def __init__(self, data_path, tokenizer, max_length):
        with open(data_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _tokenize_pair(self, example):
        """Tokenize a preference pair into (prompt_ids, chosen_ids, rejected_ids)."""
        prompt = example["prompt"]
        chosen = prompt + example["chosen"]
        rejected = prompt + example["rejected"]

        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        chosen_ids = self.tokenizer.encode(chosen, add_bos=True)
        rejected_ids = self.tokenizer.encode(rejected, add_bos=True)

        # Create labels (mask prompt portion)
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids[len(prompt_ids):]
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids[len(prompt_ids):]

        # Truncate
        chosen_ids = chosen_ids[:self.max_length]
        rejected_ids = rejected_ids[:self.max_length]
        chosen_labels = chosen_labels[:self.max_length]
        rejected_labels = rejected_labels[:self.max_length]

        return chosen_ids, chosen_labels, rejected_ids, rejected_labels

    def get_batch(self, batch_size):
        """Get a batch of preference pairs."""
        import random
        indices = random.sample(range(len(self.examples)), 
                                min(batch_size, len(self.examples)))

        batch_chosen_ids = []
        batch_chosen_labels = []
        batch_rejected_ids = []
        batch_rejected_labels = []

        for idx in indices:
            c_ids, c_labels, r_ids, r_labels = self._tokenize_pair(
                self.examples[idx]
            )
            batch_chosen_ids.append(c_ids)
            batch_chosen_labels.append(c_labels)
            batch_rejected_ids.append(r_ids)
            batch_rejected_labels.append(r_labels)

        # Pad each batch independently
        def pad_batch(ids_list, labels_list, pad_id=0):
            max_len = max(len(ids) for ids in ids_list)
            padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in ids_list]
            padded_labels = [lbl + [-100] * (max_len - len(lbl)) for lbl in labels_list]
            return (
                torch.tensor(padded_ids, dtype=torch.long),
                torch.tensor(padded_labels, dtype=torch.long),
            )

        chosen_x, chosen_y = pad_batch(batch_chosen_ids, batch_chosen_labels)
        rejected_x, rejected_y = pad_batch(batch_rejected_ids, batch_rejected_labels)

        device = config.device
        return (
            chosen_x.to(device), chosen_y.to(device),
            rejected_x.to(device), rejected_y.to(device),
        )


def compute_log_probs(model, input_ids, labels):
    """Compute sum of log probabilities for response tokens."""
    logits, _ = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    mask = (shift_labels != -100).float()
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(policy_chosen_lp, policy_rejected_lp,
             ref_chosen_lp, ref_rejected_lp, beta=0.1):
    """
    Compute DPO loss.
    
    L = -log(sigma(beta * ((log pi(yw|x) - log ref(yw|x)) 
                          - (log pi(yl|x) - log ref(yl|x)))))
    """
    chosen_reward = beta * (policy_chosen_lp - ref_chosen_lp)
    rejected_reward = beta * (policy_rejected_lp - ref_rejected_lp)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss


def train_dpo():
    device = config.device
    beta = 0.1
    dpo_lr = 1e-5
    dpo_epochs = 1
    batch_size = 2  # DPO batches are memory-heavy (2 forward passes per step)

    # Load tokenizer
    tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)

    # Load SFT model as policy (trainable)
    policy_model = GPTLanguageModel(config).to(device)
    sft_ckpt = torch.load(
        "checkpoints/sft/best_sft_model.pt",
        map_location=device, weights_only=False,
    )
    policy_model.load_state_dict(sft_ckpt["model_state_dict"])

    # Create reference model (frozen copy of SFT model)
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load preference dataset
    dataset = DPODataset(
        "data/preference_pairs.json", tokenizer, config.block_size
    )

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=dpo_lr)

    os.makedirs("checkpoints/dpo", exist_ok=True)
    policy_model.train()
    step = 0
    steps_per_epoch = len(dataset.examples) // batch_size

    for epoch in range(dpo_epochs):
        print(f"\n--- DPO Epoch {epoch + 1}/{dpo_epochs} ---")

        for batch_idx in range(steps_per_epoch):
            chosen_x, chosen_y, rejected_x, rejected_y = dataset.get_batch(
                batch_size
            )

            # Policy model log probs
            policy_chosen_lp = compute_log_probs(policy_model, chosen_x, chosen_y)
            policy_rejected_lp = compute_log_probs(
                policy_model, rejected_x, rejected_y
            )

            # Reference model log probs (no grad)
            with torch.no_grad():
                ref_chosen_lp = compute_log_probs(ref_model, chosen_x, chosen_y)
                ref_rejected_lp = compute_log_probs(
                    ref_model, rejected_x, rejected_y
                )

            loss = dpo_loss(
                policy_chosen_lp, policy_rejected_lp,
                ref_chosen_lp, ref_rejected_lp, beta,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % 10 == 0:
                # Compute accuracy: how often does policy prefer chosen?
                with torch.no_grad():
                    chosen_r = policy_chosen_lp - ref_chosen_lp
                    rejected_r = policy_rejected_lp - ref_rejected_lp
                    accuracy = ((chosen_r - rejected_r) > 0).float().mean().item()

                print(
                    f"  Step {step:4d} | Loss: {loss.item():.4f} | "
                    f"Accuracy: {accuracy:.2%}"
                )

        # Save after epoch
        torch.save({
            "step": step,
            "epoch": epoch,
            "model_state_dict": policy_model.state_dict(),
        }, f"checkpoints/dpo/dpo_epoch_{epoch+1}.pt")

    print("\nDPO Training Complete!")


if __name__ == "__main__":
    train_dpo()
```

## Evaluating After DPO

### Win Rate

The primary metric for DPO is **win rate**: how often does the DPO model's response beat the SFT model's response?

1. Generate responses from both models on a held-out set of prompts
2. Have a human (or a strong LLM judge) compare responses pairwise
3. Win rate > 50% means DPO improved the model

### Reward Accuracy

During training, the DPO accuracy metric tells you how often the model assigns higher implicit reward to the chosen response vs the rejected response. This should be:

| Accuracy | Meaning |
|---|---|
| 50% | Random — model hasn't learned preferences |
| 60–70% | Learning — model is starting to distinguish quality |
| 75–85% | Good — model reliably prefers better responses |
| > 90% | May be overfitting to training preferences |

### Generation Quality

Also measure Distinct-N, Self-BLEU, and perplexity (see Generation Quality Metrics manual) to ensure DPO did not degrade the model's fluency.

## Common Issues

### 1. KL Divergence Explosion

If the model deviates too far from the reference, outputs become incoherent. Monitor the KL divergence:

$$
D_{\text{KL}} = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

If KL exceeds ~10, increase $\beta$ or reduce the learning rate.

### 2. Reward Hacking

The model finds shortcuts to maximize the implicit reward without actually improving quality. Signs:
- Accuracy reaches 100% but generations are poor
- The model generates very short or very long responses

Prevention: use a moderate $\beta$ (0.1) and diverse preference data.

### 3. Distribution Mismatch

If preference data is very different from SFT training data, the model struggles. Ensure prompts in preference data overlap with the instruction types seen during SFT.

## Summary

| Stage | Input | Output | Effort |
|---|---|---|---|
| **Pre-Training** | Raw web text | Base LM (next-word predictor) | Days–weeks |
| **SFT** | Instruction-response pairs | Instruction-following model | Hours |
| **DPO** | Preference pairs (chosen/rejected) | Preference-aligned model | Hours |
| **Chat Template** | Special tokens + format | Conversational model | Minutes |

## References

- Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model."
- Christiano et al. (2017). "Deep Reinforcement Learning from Human Preferences." (RLHF)
- Ouyang et al. (2022). "Training language models to follow instructions with human feedback." (InstructGPT)
- Tunstall et al. (2023). "Zephyr: Direct Distillation of LM Alignment." (Practical DPO on small models)
