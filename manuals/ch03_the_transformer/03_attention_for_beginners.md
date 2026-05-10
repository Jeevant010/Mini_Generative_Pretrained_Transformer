# Chapter 3.3 — Attention for Beginners

## The Core Question

When the model is predicting the next word after "The cat sat on the", it needs to decide: which previous words are most important right now?

- "cat" is important — it tells us what sat on something
- "on" is important — it tells us the relationship
- "The" is less important — it is just a grammar word

**Attention** is the mechanism that lets the model figure out which previous words to focus on at each step.

## How Attention Works (Plain English)

Imagine you are reading a sentence and trying to predict the next word. At each word, your brain does three things:

1. **Ask a question** (Query): "What kind of information do I need right now?"
2. **Check each previous word** (Key): "Does this word have the information I need?"
3. **Get the answer** (Value): "Here is the actual information from that word."

The model does the same thing, but with numbers:

1. Each word generates a **Query** vector — "I am looking for X"
2. Each word generates a **Key** vector — "I contain information about X"
3. Each word generates a **Value** vector — "Here is my actual information"

The model compares every Query with every Key. When a Query and Key match well, the model pays more attention to that word's Value.

## A Concrete Example

For the sentence "The cat sat on the ___":

Position 6 (the blank) generates a Query: "I need to know what is sitting on something."

It compares this Query against all Keys:
- Position 1 ("The"): Key says "I am an article" → low match score
- Position 2 ("cat"): Key says "I am a living thing doing an action" → HIGH match score
- Position 3 ("sat"): Key says "I am a past-tense action verb" → medium match score
- Position 4 ("on"): Key says "I describe a spatial relationship" → HIGH match score
- Position 5 ("the"): Key says "I am an article before a noun" → medium match score

The model then takes a weighted average of all Values, paying most attention to "cat" and "on." This gives it the information it needs to predict "mat" or "floor."

## Multiple Heads

One type of attention might focus on grammar (subject-verb relationships). Another might focus on meaning (what topic we are discussing). A third might focus on syntax (punctuation and sentence structure).

Our model uses **12 attention heads** — 12 independent attention mechanisms that run in parallel. Each head has its own Query, Key, and Value projections, so each can learn to focus on different patterns.

After all 12 heads compute their results, the outputs are combined together.

## Grouped-Query Attention (GQA)

Normal multi-head attention has 12 Query heads, 12 Key heads, and 12 Value heads. That is a lot of parameters.

Our model uses a trick called **Grouped-Query Attention**: it has 12 Query heads but only 4 Key-Value heads. Every 3 Query heads share the same Key-Value head.

```
Query heads:  Q1  Q2  Q3  |  Q4  Q5  Q6  |  Q7  Q8  Q9  |  Q10 Q11 Q12
              ↓   ↓   ↓      ↓   ↓   ↓      ↓   ↓   ↓      ↓   ↓   ↓
KV heads:     KV1 KV1 KV1 |  KV2 KV2 KV2 |  KV3 KV3 KV3 |  KV4 KV4 KV4
```

This saves memory and parameters while keeping most of the quality. LLaMA 2 uses the same technique.

## The Causal Mask

In a language model, each word can only look at words that came **before** it. You cannot peek at the future when predicting the next word.

This is enforced by a **causal mask** — a rule that sets the attention score to negative infinity for any future position. After applying softmax (which converts scores to probabilities), negative infinity becomes zero probability. So the model literally cannot attend to future tokens.

```
Can attend to:
Word 1 → only itself
Word 2 → word 1, itself
Word 3 → word 1, word 2, itself
Word 4 → word 1, word 2, word 3, itself
```

## The Math (Optional)

If you are comfortable with math, attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V
$$

Where:
- $Q$ is the Query matrix
- $K$ is the Key matrix
- $V$ is the Value matrix
- $d_h = 64$ is the head dimension
- $\sqrt{d_h}$ prevents the scores from getting too large

The division by $\sqrt{64} = 8$ is called "scaled" attention. Without it, the dot products between Q and K can become very large, which makes the softmax output extreme (all probability on one token).

## Flash Attention

Computing attention requires storing a large matrix of scores (every token compared to every other token). For our 384-token context, that is a 384 × 384 matrix per head — manageable.

But for larger models with 4096+ token contexts, this matrix becomes enormous. **Flash Attention** is an optimization that computes the same result without materializing the full matrix in memory.

Our model uses PyTorch's built-in Flash Attention:
```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

This produces the same mathematical result as manual attention but uses less GPU memory and runs faster.
