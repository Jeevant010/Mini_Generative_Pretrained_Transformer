# Chapter 2.4 — Special Tokens

## What Are Special Tokens?

Besides regular text tokens, our tokenizer has a few special tokens with specific jobs. These are not words — they are control signals that tell the model where a document starts, ends, or where it should pay attention.

## Our Special Tokens

| Token | ID | Purpose |
|---|---|---|
| `<pad>` | 0 | Padding — fills empty space when sequences need to be the same length |
| `<bos>` | 1 | Beginning of Sequence — marks the start of a new text |
| `<eos>` | 2 | End of Sequence — marks the end of a document |
| `<unk>` | 3 | Unknown — used if a character cannot be tokenized (rare with byte-level BPE) |

## How They Are Used

### End of Sequence (`<eos>`)

This is the most important special token in our pipeline. During data preparation, every document gets an `<eos>` token at the end:

```
Document text: "The cat sat on the mat."
Token IDs:     [464, 3797, 3332, 319, 262, 2603, 13, 2]
                                                       ↑
                                                    <eos> token
```

Why this matters: The training data is one long stream of tokens. Without `<eos>`, the model would think all documents are one continuous text. The `<eos>` token tells the model: "This document is finished. The next token belongs to a different document."

### Beginning of Sequence (`<bos>`)

When you give the model a prompt for generation, the prompt starts with `<bos>`:

```python
context_ids = tokenizer.encode(prompt, add_bos=True)
# Result: [1, 464, 2003, ...]  ← 1 is <bos>
```

This tells the model: "A new piece of text is starting."

### Padding (`<pad>`)

When we process multiple texts in a batch, they need to be the same length. If one text is 50 tokens and another is 80 tokens, we add `<pad>` tokens to make the shorter one 80 tokens long:

```
Text 1: [464, 3797, 3332, ...]  (80 tokens)
Text 2: [262, 2603, 13, ..., 0, 0, 0, 0]  (50 tokens + 30 padding)
```

The model learns to ignore `<pad>` tokens.

### Unknown (`<unk>`)

With byte-level BPE, the `<unk>` token is almost never used. Since the tokenizer starts from individual bytes, it can represent any text — even Chinese characters, emojis, or binary data. The `<unk>` token exists as a safety net.

## Why Special Tokens Matter

Special tokens might seem like a minor detail, but they are critical:

1. **Without `<eos>`**, the model would blend documents together, learning wrong patterns like "the end of one article naturally flows into the beginning of another"
2. **Without `<bos>`**, the model would not know where to start generating
3. **Without `<pad>`**, we could not process multiple texts at once (batch training)

In later chapters (Chapter 10: What Comes Next), you will see that adding new special tokens — like `<|user|>` and `<|assistant|>` — is how the model learns to have conversations. But that is for the future branch.
