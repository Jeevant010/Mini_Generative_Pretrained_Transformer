# Chapter 9.2 — Repetition Loops

## The Problem

Even without non-English contamination, language models can get stuck in repetition loops. Look at this output from step 60,000:

> "In the beginning, there was nothing but pure power that had gone unchecked in the past — a combination of power and power. In the end, there was nothing but pure power and power (and power) that needed to be utilized. As a result, the power of power — and power — was simply the means of production. The power of power was a force that was..."

The word "power" appears 11 times in one paragraph. This is a milder form of the same problem that causes the "ibn nimy" loops.

## Why Language Models Repeat

### The Mathematical Trap

When the model generates "power" and it fits the context, the probability of "power" at the next position increases slightly. Then, with "power" appearing even more in the context, its probability increases again. This creates a positive feedback loop:

```
"power" appears → probability of "power" increases → "power" appears again → ...
```

This is known as the **degeneration problem**, studied in the paper "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2020).

### Temperature and Repetition

Lower temperature makes repetition worse. Temperature controls how "spread out" the probability distribution is:

| Temperature | Effect | Repetition Risk |
|---|---|---|
| 0.1 | Very peaked — almost always picks the top token | Very high |
| 0.5 | Moderately peaked | Moderate |
| 0.8 | Balanced (our default) | Low |
| 1.0 | No change from model output | Low |
| 1.5 | Very spread out — picks random tokens often | Very low (but may be incoherent) |

## How We Detect Repetition

We use the **Distinct-N** metric (Chapter 7). Distinct-2 measures the fraction of unique bigrams (2-word pairs) in the output:

- Healthy text: Distinct-2 ≈ 0.6-0.8
- Mild repetition: Distinct-2 ≈ 0.3-0.5
- Severe loop: Distinct-2 < 0.1

The "power power power" output would have Distinct-2 around 0.3. The "ibn nimy" loop would have Distinct-2 around 0.1.

## Solutions We Implemented

### 1. Repetition Penalty

Divides the logit of any previously-generated token by a penalty factor (default 1.2). This makes it progressively harder for the model to repeat tokens.

### 2. Top-p Sampling

When the model is very confident about one token, top-p forces it to consider alternatives. This prevents the feedback loop from starting.

### 3. Top-k Filtering

Limits the model to choosing from the top 50 most likely tokens. This prevents it from generating extremely unlikely tokens (which can start loops in a different way).

## The Best Settings

Based on our experiments, the best generation settings for our model are:

```python
model.generate(
    idx,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
)
```

These settings produce coherent, diverse text without excessive repetition.

## Key Lesson

Repetition is not a bug in our model — it is a fundamental property of autoregressive language models. Even GPT-4 can repeat itself without proper sampling controls. The solution is always at inference time: repetition penalty + nucleus sampling.
