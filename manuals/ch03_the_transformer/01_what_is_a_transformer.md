# Chapter 3.1 — What Is a Transformer?

## The Big Picture

A Transformer is a type of neural network architecture. Think of it as a specific blueprint for building an AI model — like how a specific engine design (V8, turbocharged) is a blueprint for building a car engine.

The Transformer was invented in 2017 by a team at Google. Before Transformers, the main architectures for processing text were:

- **RNNs** (Recurrent Neural Networks) — Processed text one word at a time, left to right. Very slow because each word had to wait for the previous word to be processed.
- **LSTMs** (Long Short-Term Memory) — An improved RNN that could remember longer patterns, but still processed sequentially.

The Transformer fixed the speed problem by processing **all words at once** using a mechanism called "attention." This is why the original paper was titled "Attention Is All You Need."

## How the Transformer Processes Text

Here is the simplified flow of our model:

```
Input text: "The future of artificial intelligence is"
         ↓
Step 1: TOKENIZE — Convert to numbers [464, 2003, 286, 11666, 4430, 318]
         ↓
Step 2: EMBED — Convert each number into a list of 768 numbers (called a "vector")
         ↓
Step 3: ADD POSITION — Tell the model where each word is in the sequence
         ↓
Step 4: PASS THROUGH 12 TRANSFORMER BLOCKS — Each block:
        a) Look at all previous words (Attention)
        b) Process the combined information (Feed-Forward)
         ↓
Step 5: PROJECT — Convert the final vectors into probabilities for all 32,000 possible next tokens
         ↓
Output: The most likely next token (e.g., "changing" with 3% probability)
```

## The Key Components

Our Transformer has these components, each explained in its own section:

| Component | Section | What It Does |
|---|---|---|
| Token Embedding | 3.2 | Converts token IDs into vectors |
| Attention | 3.3 | Lets each position look at previous positions |
| RoPE | 3.4 | Tells the model about word order |
| SwiGLU | 3.5 | Processes information after attention |
| RMSNorm | 3.6 | Keeps numbers in a stable range |
| Transformer Block | 3.7 | Combines attention + feed-forward + normalization |
| Model Summary | 3.8 | All the numbers in our specific model |

## Why "Decoder-Only"?

The original Transformer had two parts:

- **Encoder** — Reads the input text
- **Decoder** — Generates the output text

For language modeling (predicting the next word), we only need the decoder part. That is why our model is called a "decoder-only Transformer." This is the same design used by GPT-2, GPT-3, GPT-4, LLaMA, and Claude.

The key rule in a decoder-only model: **each word can only look at words that came before it, never after it.** This makes sense — when predicting the next word, you cannot peek at the future.

## An Analogy

Think of the Transformer as a team of analysts reading a document:

1. Each analyst reads the document independently (embedding)
2. They have a meeting to discuss what they have read (attention)
3. Each analyst updates their understanding based on the discussion (feed-forward)
4. They repeat steps 2-3 twelve times, each time deepening their understanding
5. After 12 rounds, they make their prediction about what comes next

This is essentially what happens in our 12-layer Transformer.
