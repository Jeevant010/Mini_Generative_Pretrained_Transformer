# Chapter 1.2 — What Is GPT?

## The Name

GPT stands for **Generative Pre-trained Transformer**. Let us break that down:

- **Generative** — It generates (creates) new text
- **Pre-trained** — It is trained on a huge amount of text before being used for anything specific
- **Transformer** — It uses a specific architecture (design) called the Transformer

## A Brief History

In 2017, a team at Google published a research paper called "Attention Is All You Need." This paper introduced the Transformer architecture. It was originally designed for translating text from one language to another.

In 2018, OpenAI realized that the Transformer could be used in a different way: instead of translating, you could train it to predict the next word. They trained a Transformer on millions of web pages, and called it GPT (Generative Pre-trained Transformer). It was surprisingly good at writing coherent text.

In 2019, OpenAI trained a bigger version — GPT-2 — on more data. It was so good at writing realistic text that OpenAI initially refused to release the full model because they were worried about misuse.

Since then, the same basic idea has been scaled up to create:

- **GPT-3** and **GPT-4** (OpenAI) — the models behind ChatGPT
- **LLaMA** (Meta) — an open-source model family
- **Claude** (Anthropic) — another AI assistant
- **Gemini** (Google) — the model you might be talking to right now

All of these are variations of the same core idea: a Transformer trained to predict the next word.

## Our Mini GPT

Our project is called "Mini Generative Pre-trained Transformer" because it follows the same recipe as GPT-2, but at a smaller scale:

| Feature | GPT-2 (Small) | Our Model |
|---|---|---|
| Parameters | 124 million | 118 million |
| Training data | 40 GB (WebText) | 10 GB (OpenWebText) |
| Vocabulary | 50,257 tokens | 32,000 tokens |
| Context window | 1024 tokens | 384 tokens |
| Hardware | Cluster of GPUs | Single laptop GPU |
| Training time | Days on many GPUs | Days on one GPU |

Our model is similar in size to GPT-2 Small! The main difference is we use a smaller context window (384 vs 1024 tokens) and train on less data.

## Why "Mini"?

We call it "Mini" because compared to modern models, it is tiny:

| Model | Parameters |
|---|---|
| Our Mini GPT | 118 million |
| GPT-2 Small | 124 million |
| GPT-2 Large | 774 million |
| LLaMA 7B | 7 billion |
| GPT-3 | 175 billion |
| GPT-4 | Estimated 1+ trillion |

Our model has 118 million parameters. GPT-4 has over a trillion. That is about 10,000 times bigger. But the fundamental architecture — the Transformer — is the same.

Think of it like building a model airplane. A model airplane cannot carry passengers, but it teaches you the principles of flight — wings, engines, lift, drag. Our Mini GPT cannot have deep conversations, but it teaches you exactly how models like GPT-4 work.

## The Key Insight

The surprising thing about GPT-style models is how simple the core idea is:

1. Take a huge amount of text
2. Build a Transformer model
3. Train it to predict the next word
4. That is it

There is no explicit grammar rules. No dictionary. No teaching the model about sentences, paragraphs, or topics. It learns all of these implicitly, just by predicting the next word millions of times.

This is called **self-supervised learning** — the training data provides its own labels. Every word in a sentence is both a training input (everything before it) and a training target (the word itself).
