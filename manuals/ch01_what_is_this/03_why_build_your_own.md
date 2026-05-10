# Chapter 1.3 — Why Build Your Own?

## The Real Reason

You can download pre-trained models. You can use APIs. You can fine-tune existing models. So why spend days building one from scratch?

Because **using** AI and **understanding** AI are completely different things.

Using ChatGPT is like driving a car. Building a language model from scratch is like building the engine. When you build the engine, you understand:

- Why it works
- Why it sometimes fails
- What you can improve
- What the real limitations are

## What You Cannot Learn From APIs

When you use an AI through an API (like calling the OpenAI or Google API), you see:

- Input goes in → Output comes out

What you do **not** see:

- How the model converts your text into numbers
- How 118 million parameters work together to predict the next word
- Why the model sometimes repeats itself in loops
- Why it makes up fake facts (hallucination)
- How the learning rate affects training quality
- What happens when the data contains non-English text
- Why training for too long makes the model worse, not better

All of these are things we encountered and solved in this project. They are documented in detail in the later chapters.

## What This Project Demonstrates

This project proves several things:

### 1. You Can Train a Real Language Model on Consumer Hardware

We trained a 118-million-parameter Transformer on a single NVIDIA RTX 4060 Laptop GPU with 8 GB of VRAM. No cloud compute. No cluster of servers. One laptop.

### 2. The Architecture Actually Works

The model went from producing random noise at step 0:

> "isrophic Doyle LIuuensual shattered rewards Doyle column..."

To producing coherent English at step 100,000:

> "The future of artificial intelligence is in the hands of a man. The next time you hear a voice calling for a computer to be integrated into the world's AI network..."

### 3. Training Is Not Magic — It Is Engineering

Training a language model involves many engineering decisions:

- How big should the model be? (We chose 12 layers, 768 dimensions)
- How much data is enough? (We prepared 10 GB of tokenized text)
- How fast should the model learn? (We used a cosine learning rate schedule)
- When should we stop? (We monitored validation loss and perplexity)
- What breaks? (Many things — see Chapter 9: Mistakes and Lessons)

### 4. Things Go Wrong, and That Is the Most Interesting Part

At around 106,000 iterations on an earlier training run, the model started producing gibberish:

> "ibn nimy ibn nimy ibn ibn ibn ibn nimy ibn nimy ibn nimy..."

This is not a bug in the code. It is a real phenomenon called **degeneration** that happens to language models when they become too confident. Understanding why this happens — and how to fix it — is one of the most valuable lessons in this project.

## Who Is This Project For?

- **Students** studying machine learning or NLP who want to understand Transformers at a code level
- **Engineers** who use AI APIs and want to understand what is behind them
- **Researchers** who need a small, inspectable model for experiments
- **Anyone curious** about how AI language models actually work

## What You Need to Follow Along

To understand this manual, you need:

- Basic Python knowledge (variables, functions, loops)
- No prior machine learning experience (we explain everything)
- No math beyond high school algebra (we show formulas but always explain them in words first)

To actually run the code, you need:

- A computer with an NVIDIA GPU (any modern one works, 6+ GB VRAM recommended)
- Python 3.8+
- PyTorch installed with CUDA support
- About 15 GB of free disk space
