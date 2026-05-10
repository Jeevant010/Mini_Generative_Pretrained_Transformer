# Chapter 7.2 — Diversity and Repetition Metrics

## Beyond Perplexity

Perplexity tells us if the model predicts well on average. But we also need to know: **Is the model's generated text diverse and non-repetitive?**

We use several complementary metrics for this.

## Distinct-N (Lexical Diversity)

**What it measures:** The fraction of unique n-grams (word sequences) in the generated text.

**How to read it:**
- Distinct-1 = fraction of unique words
- Distinct-2 = fraction of unique two-word pairs (bigrams)
- Distinct-3 = fraction of unique three-word sequences (trigrams)

**Example:**

For "the cat sat on the mat on the floor":
- Distinct-1: 6 unique words / 9 total = 0.67
- Distinct-2: 7 unique bigrams / 8 total = 0.88

For "the the the the the the the the":
- Distinct-1: 1 unique word / 8 total = 0.125
- Distinct-2: 1 unique bigram / 7 total = 0.143

**Healthy ranges:**

| Metric | Healthy | Warning | Problem |
|---|---|---|---|
| Distinct-1 | > 0.5 | 0.3–0.5 | < 0.3 |
| Distinct-2 | > 0.6 | 0.3–0.6 | < 0.3 |
| Distinct-3 | > 0.7 | 0.4–0.7 | < 0.4 |

## Self-BLEU (Inter-Sample Similarity)

**What it measures:** How similar different generated samples are to each other.

Generate 5 samples from different prompts. If they all say similar things, Self-BLEU is high. If they say different things, Self-BLEU is low.

**How to read it:**
- Self-BLEU < 0.3 → Good diversity between samples
- Self-BLEU 0.3–0.5 → Moderate overlap
- Self-BLEU > 0.5 → Samples are too similar (the model is saying the same things regardless of prompt)

## Output Entropy

**What it measures:** How confident the model is when making predictions. Measured in bits.

**How to read it:**
- Entropy ~5–10 bits → Healthy. The model considers multiple tokens at each position.
- Entropy < 3 bits → Overconfident. The model is too sure, which leads to repetitive, deterministic output.
- Entropy > 12 bits → Underconfident. The model is too uncertain, which leads to random, incoherent output.

## Max Repeated N-gram

**What it measures:** The length of the longest repeated sequence in the generated text.

**How to read it:**
- Max repeat = 2–3 → Normal. Common phrases ("in the", "it is") repeat naturally.
- Max repeat = 5–8 → Warning. Multi-word phrases are repeating.
- Max repeat > 8 → Problem. The model is stuck in a repetition loop.

## Running the Evaluation

We created `evaluation/quality_metrics.py` to compute all of these metrics:

```bash
# Evaluate the latest checkpoint
python -m evaluation.quality_metrics

# Evaluate a specific checkpoint
python -m evaluation.quality_metrics --checkpoint checkpoints/ckpt_step_60000.pt

# Compare all checkpoints
python -m evaluation.quality_metrics --all-checkpoints
```

The `--all-checkpoints` mode produces a comparison table showing how each metric changes across training, making it easy to spot when quality starts to degrade.
