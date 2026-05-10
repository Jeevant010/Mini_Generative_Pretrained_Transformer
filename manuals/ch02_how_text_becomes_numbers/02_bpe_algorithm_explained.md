# Chapter 2.2 — The BPE Algorithm Explained

## What BPE Stands For

BPE stands for **Byte Pair Encoding**. It is the algorithm that decides which pieces of text become tokens. Think of it as an automatic process that looks at millions of sentences and figures out: "These letters often appear together, so I should combine them into one token."

## How It Works (Step by Step)

Let us walk through a tiny example. Imagine our entire training text is just:

```
"low lower lowest"
```

### Step 1: Start with Individual Characters

First, split everything into individual characters:

```
l o w   l o w e r   l o w e s t
```

Our initial vocabulary is: `{l, o, w, e, r, s, t, (space)}`

That is 8 tokens.

### Step 2: Count Pairs

Count how often each pair of adjacent characters appears:

| Pair | Count |
|---|---|
| l + o | 3 times |
| o + w | 3 times |
| w + (space) | 1 time |
| w + e | 2 times |
| e + r | 1 time |
| e + s | 1 time |
| s + t | 1 time |

### Step 3: Merge the Most Frequent Pair

"l + o" and "o + w" both appear 3 times. Pick one — say "l + o". Create a new token "lo":

```
lo w   lo w e r   lo w e s t
```

Vocabulary: `{l, o, w, e, r, s, t, (space), lo}`

### Step 4: Repeat

Count pairs again. Now "lo + w" appears 3 times. Merge it:

```
low   low e r   low e s t
```

Vocabulary: `{l, o, w, e, r, s, t, (space), lo, low}`

### Step 5: Keep Going

Count again. "low + e" appears 2 times. Merge it:

```
low   lowe r   lowe s t
```

Keep going until the vocabulary reaches the target size (32,000 in our case).

## The Key Idea

BPE learns the vocabulary from the data. It does not use a dictionary or any language rules. It just looks at what character sequences appear most often and combines them.

This means:
- **Common words** like "the", "and", "for" become single tokens quickly
- **Medium words** like "computer", "language" also become single tokens
- **Rare words** like "antidisestablishmentarianism" get split into smaller pieces

## Byte-Level BPE

Our project uses **byte-level** BPE. This means we start from individual bytes (0-255), not from characters. This is important because:

- Every possible text can be represented (including emojis, Chinese characters, code)
- There is no "unknown token" problem — even if the model has never seen a word, it can represent it as bytes
- The starting vocabulary is always 256 (one for each byte value)

The process is the same as above, but instead of starting with letters, we start with bytes.

## How Our Tokenizer Was Built

In our project, the tokenizer was trained automatically during data preparation:

1. We took a 200 MB sample of our training text
2. We ran BPE on it, starting from byte-level units
3. We kept merging pairs until the vocabulary reached 32,000 tokens
4. We saved the result as `bpe_tokenizer_32k.json`

This file contains the complete vocabulary — all 32,000 tokens and the rules for how to split any text into those tokens.

## A Real Example

Here is how our tokenizer handles a real sentence:

```
Input:  "The future of artificial intelligence is"
Output: [464, 2003, 286, 11666, 4430, 318]
```

That is 6 tokens for 6 words. Each common word becomes one token. If we had a rare word, it would be split into more tokens.

## Why This Matters

The tokenizer is the foundation of everything else. If the tokenizer is bad:

- Common words waste multiple tokens (inefficient)
- The model's context window fills up with tokens instead of meaning
- Rare words become unrecognizable strings

Our 32,000-token BPE vocabulary was trained on 200 MB of representative web text, which is large enough to learn all common English patterns.
