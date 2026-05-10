# Chat Templates and Deployment

## Overview

After SFT and DPO, the model can follow instructions and produce preferred responses. The final step before deployment is establishing a **chat template** — a structured format that tells the model who is speaking, what its role is, and how to handle multi-turn conversations.

## Why Chat Templates Matter

Without a chat template, the model treats every input as a document continuation. Chat templates solve this by introducing:

1. **Role markers** — the model knows when the user is speaking vs when it should respond
2. **System instructions** — persistent behavior guidelines (tone, safety rules, identity)
3. **Turn boundaries** — the model knows when a response should end

## Chat Template Format

### Recommended Format for This Project

```text
<|system|>You are a helpful AI assistant. Answer questions clearly and concisely.<|end|>
<|user|>What is machine learning?<|end|>
<|assistant|>Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed for every task.<|end|>
```

### Multi-Turn Example

```text
<|system|>You are a helpful AI assistant.<|end|>
<|user|>What is 2+2?<|end|>
<|assistant|>2 + 2 = 4.<|end|>
<|user|>What about 3+3?<|end|>
<|assistant|>3 + 3 = 6.<|end|>
```

## Adding Special Tokens to the Tokenizer

The tokenizer must recognize the chat template markers as single tokens, not as character sequences.

### Step 1: Extend the Tokenizer

```python
# extend_tokenizer.py
"""Add chat template special tokens to the BPE tokenizer."""

from tokenizer import BytePairTokenizer

# New special tokens for chat format
CHAT_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


def extend_tokenizer(tokenizer_path, output_path=None):
    """Add chat template tokens to existing tokenizer."""
    tokenizer = BytePairTokenizer.load(tokenizer_path)

    # Add new special tokens
    for token in CHAT_TOKENS:
        tokenizer.tokenizer.add_special_tokens([token])

    # Update special_to_id mapping
    tokenizer._sync_special_ids()
    vocab = tokenizer.tokenizer.get_vocab()
    for token in CHAT_TOKENS:
        if token in vocab:
            tokenizer.special_to_id[token] = vocab[token]
            print(f"  Added: {token} -> ID {vocab[token]}")

    # Save
    save_path = output_path or tokenizer_path.replace(".json", "_chat.json")
    tokenizer.save(save_path)
    print(f"Extended tokenizer saved to {save_path}")
    print(f"New vocab size: {tokenizer.vocab_size}")

    return tokenizer


if __name__ == "__main__":
    extend_tokenizer("bpe_tokenizer_32k.json")
```

### Step 2: Resize the Model Embeddings

After adding tokens, the model's embedding layer needs to grow:

```python
import torch
import torch.nn as nn


def resize_token_embeddings(model, new_vocab_size):
    """
    Resize the token embedding and LM head to accommodate new special tokens.
    
    Preserves existing weights and initializes new token embeddings
    with the mean of existing embeddings.
    """
    old_vocab_size = model.token_embed.num_embeddings
    n_embd = model.token_embed.embedding_dim

    if new_vocab_size <= old_vocab_size:
        return model

    # Create new embedding
    new_embed = nn.Embedding(new_vocab_size, n_embd)
    new_embed.weight.data[:old_vocab_size] = model.token_embed.weight.data

    # Initialize new tokens as mean of existing embeddings
    mean_embed = model.token_embed.weight.data.mean(dim=0)
    new_embed.weight.data[old_vocab_size:] = mean_embed.unsqueeze(0).expand(
        new_vocab_size - old_vocab_size, -1
    )

    # Create new LM head
    new_lm_head = nn.Linear(n_embd, new_vocab_size, bias=False)
    new_lm_head.weight.data[:old_vocab_size] = model.lm_head.weight.data
    new_lm_head.weight.data[old_vocab_size:] = mean_embed.unsqueeze(0).expand(
        new_vocab_size - old_vocab_size, -1
    )

    # Replace in model
    model.token_embed = new_embed
    model.lm_head = new_lm_head

    # Re-tie weights
    model.token_embed.weight = model.lm_head.weight

    print(f"Resized embeddings: {old_vocab_size} -> {new_vocab_size}")
    return model
```

## System Prompt Engineering

### Effective System Prompts

The system prompt establishes the model's identity and behavior. For a small 85M model, keep it simple:

```text
<|system|>You are a helpful AI assistant. Answer questions accurately and concisely. If you don't know something, say so.<|end|>
```

### System Prompt Guidelines

| Principle | Good Example | Bad Example |
|---|---|---|
| **Be specific** | "Answer in 2-3 sentences" | "Be helpful" |
| **Set boundaries** | "If unsure, say 'I don't know'" | (no guidance) |
| **Define tone** | "Be friendly and professional" | "Act like a human" |
| **Keep it short** | 1-2 sentences | A full paragraph of rules |

For an 85M model, the system prompt should be under 50 tokens. Longer system prompts consume valuable context window space and may confuse a small model.

## Building a Chat Interface

### Simple Command-Line Chat

```python
# chat.py
"""Interactive chat interface for the fine-tuned model."""

import torch
import config
from model import GPTLanguageModel
from tokenizer import BytePairTokenizer


def chat():
    device = config.device
    tokenizer = BytePairTokenizer.load("bpe_tokenizer_32k_chat.json")

    model = GPTLanguageModel(config).to(device)
    # Load DPO checkpoint (or SFT if DPO not done)
    ckpt = torch.load("checkpoints/dpo/dpo_epoch_1.pt",
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    system_prompt = "You are a helpful AI assistant."
    conversation = f"<|system|>{system_prompt}<|end|>\n"

    print("Chat with Mini GPT (type 'quit' to exit)")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Add user message
        conversation += f"<|user|>{user_input}<|end|>\n<|assistant|>"

        # Encode
        input_ids = tokenizer.encode(conversation)

        # Truncate to fit context window
        if len(input_ids) > config.block_size - 100:
            # Keep system prompt + recent turns
            input_ids = input_ids[-(config.block_size - 100):]

        idx = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            output = model.generate(
                idx, max_new_tokens=150,
                temperature=0.7, top_k=40,
                repetition_penalty=1.2,
            )

        # Decode only the new tokens
        new_tokens = output[0][len(input_ids):].tolist()
        response = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Stop at end token
        end_marker = "<|end|>"
        if end_marker in response:
            response = response[:response.index(end_marker)]

        response = response.strip()
        print(f"\nAssistant: {response}")

        # Update conversation history
        conversation += response + "<|end|>\n"


if __name__ == "__main__":
    chat()
```

## Context Window Management

### The Problem

With a 384-token context window, multi-turn conversations fill up quickly. A system prompt (~30 tokens) + 3 exchanges (~100 tokens each) = ~330 tokens, leaving almost no room for responses.

### Solutions

#### 1. Sliding Window

Keep only the most recent N turns:

```python
def truncate_conversation(conversation, tokenizer, max_tokens):
    """Keep system prompt + recent turns within max_tokens."""
    # Always keep system prompt
    system_end = conversation.find("<|end|>") + len("<|end|>") + 1
    system = conversation[:system_end]
    rest = conversation[system_end:]

    # Split into turns
    turns = rest.split("<|end|>")
    turns = [t + "<|end|>" for t in turns if t.strip()]

    # Add turns from most recent until we hit the limit
    kept_turns = []
    current_tokens = len(tokenizer.encode(system))
    for turn in reversed(turns):
        turn_tokens = len(tokenizer.encode(turn))
        if current_tokens + turn_tokens > max_tokens - 100:
            break
        kept_turns.insert(0, turn)
        current_tokens += turn_tokens

    return system + "\n".join(kept_turns)
```

#### 2. Summary Compression

Summarize old turns into a condensed context (requires the model to generate summaries, which may not work well for an 85M model).

### Recommendation

For the 85M model with 384-token context, practical conversations are limited to 2–3 exchanges. This is a known limitation of small context windows and small models.

## Deployment Options

### Local CLI (Current)

The `chat.py` script above provides a simple interactive interface.

### Gradio Web Interface

```python
# app_gradio.py
"""Simple web chat interface using Gradio."""

import gradio as gr
import torch
import config
from model import GPTLanguageModel
from tokenizer import BytePairTokenizer

# Load model and tokenizer
tokenizer = BytePairTokenizer.load("bpe_tokenizer_32k_chat.json")
model = GPTLanguageModel(config).to(config.device)
ckpt = torch.load("checkpoints/sft/best_sft_model.pt",
                   map_location=config.device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


def respond(message, history):
    system = "<|system|>You are a helpful AI assistant.<|end|>\n"
    conversation = system

    for user_msg, bot_msg in history:
        conversation += f"<|user|>{user_msg}<|end|>\n"
        conversation += f"<|assistant|>{bot_msg}<|end|>\n"

    conversation += f"<|user|>{message}<|end|>\n<|assistant|>"

    input_ids = tokenizer.encode(conversation)
    input_ids = input_ids[-(config.block_size - 100):]
    idx = torch.tensor([input_ids], dtype=torch.long, device=config.device)

    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=150,
                                 temperature=0.7, top_k=40,
                                 repetition_penalty=1.2)

    new_tokens = output[0][len(input_ids):].tolist()
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)
    if "<|end|>" in response:
        response = response[:response.index("<|end|>")]

    return response.strip()


demo = gr.ChatInterface(respond, title="Mini GPT Chat")
demo.launch()
```

## What To Expect

### After SFT Only

The model will:
- Respond to instructions in the template format
- Produce relevant answers to simple questions
- Sometimes generate off-topic or low-quality responses

### After SFT + DPO

The model will:
- Produce higher-quality responses
- Be more consistent in following instructions
- Better handle edge cases

### Limitations of an 85M Model

Even with full post-training:
- Complex reasoning will be limited
- Multi-step instructions may fail
- World knowledge is constrained by training data
- Long conversations exceed the context window
- The model will still hallucinate frequently

These are fundamental limitations of model size and training data, not of the post-training pipeline.

## References

- Bai et al. (2023). "Qwen Technical Report." (Chat template design)
- Touvron et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." (Chat template)
- Jiang et al. (2023). "Mistral 7B." (Efficient chat format)
