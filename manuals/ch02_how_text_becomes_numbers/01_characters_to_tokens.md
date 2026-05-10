# Chapter 2.1 — Characters to Tokens

## The Problem

Computers work with numbers. They do not understand letters, words, or sentences. Before we can train a model on text, we need to convert every piece of text into a sequence of numbers.

This process is called **tokenization**.

## A Simple Example

Take the sentence: "The cat sat."

One approach would be to give each **character** a number:

| Character | Number |
|---|---|
| T | 84 |
| h | 104 |
| e | 101 |
| (space) | 32 |
| c | 99 |
| a | 97 |
| t | 116 |
| ... | ... |

So "The cat sat." becomes: `[84, 104, 101, 32, 99, 97, 116, 32, 115, 97, 116, 46]`

That is 12 numbers for a 12-character sentence. This works, but it has a problem: the model sees one character at a time. It has to figure out that `T-h-e` is the word "The" on its own. That makes learning very slow.

## A Better Approach: Words

What if we gave each **word** a number?

| Word | Number |
|---|---|
| The | 1 |
| cat | 2 |
| sat | 3 |
| . | 4 |

Now "The cat sat." becomes `[1, 2, 3, 4]`. Only 4 numbers! Much more efficient.

But this has a different problem: what happens when the model sees a word it has never seen before? If it has never seen the word "cryptocurrency" in its vocabulary, it simply cannot process it. This is called the **out-of-vocabulary** problem.

## The Goldilocks Approach: Subword Tokens

Modern language models use a middle ground — they split text into pieces that are bigger than characters but sometimes smaller than words. These pieces are called **tokens**.

For example, the word "unhappiness" might be split into:

| Token | Meaning |
|---|---|
| un | prefix meaning "not" |
| happi | root of "happy" |
| ness | suffix meaning "state of" |

This way:
- Common words like "the" and "is" become a single token (efficient)
- Rare words get split into familiar pieces (no out-of-vocabulary problem)
- Any text can be represented (nothing is impossible to tokenize)

## Our Tokenizer

Our project uses a tokenizer with **32,000 tokens** in its vocabulary. This means every piece of English text is converted into a sequence of numbers between 0 and 31,999.

Some examples of what tokens look like:

- "the" → one token
- "computer" → one token (common enough to be a single token)
- "cryptocurrency" → probably 2-3 tokens
- "supercalifragilisticexpialidocious" → many tokens (very rare word)
- "Hello! How are you?" → about 5-6 tokens
- A full web page → thousands of tokens

## Why 32,000?

The vocabulary size (32,000) is a design choice. Here is why:

- **Too few tokens** (e.g., 256 = just bytes): Every word needs many tokens. Sequences become very long. The model needs more context to understand anything.
- **Too many tokens** (e.g., 1 million): The model needs a huge table to store all token embeddings. Most tokens would be so rare that the model never learns them well.
- **32,000 tokens**: A good balance. Common words are single tokens, rare words are split into 2-3 pieces, and the vocabulary is small enough to store efficiently.

GPT-2 uses 50,257 tokens. LLaMA uses 32,000. Our choice of 32,000 follows the LLaMA approach.

## What Comes Next

In the next section, we will explain exactly how the tokenizer decides where to split text — the algorithm is called Byte Pair Encoding (BPE).
