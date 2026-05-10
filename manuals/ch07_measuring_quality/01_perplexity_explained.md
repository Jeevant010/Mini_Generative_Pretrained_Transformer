# Chapter 7.1 — Perplexity Explained

## What Is Perplexity?

Perplexity (PPL) is the most common metric for language models. It answers the question: **"On average, how many tokens is the model choosing between at each position?"**

- **PPL = 32,000** → The model is equally uncertain about all 32,000 tokens (random guessing)
- **PPL = 100** → The model has narrowed it down to about 100 candidates
- **PPL = 33** → The model is choosing between about 33 tokens (our model at step 60K)
- **PPL = 1** → The model is perfectly certain (impossible in practice)

Lower perplexity = better model.

## How It Is Calculated

Perplexity is calculated from the average loss:

$$
\text{PPL} = e^{\text{average loss}}
$$

Our model's loss at step 60K is 3.517:

$$
\text{PPL} = e^{3.517} = 33.69
$$

## What Perplexity Does NOT Tell You

Perplexity is a useful overall measure, but it has blind spots:

1. **It does not detect repetition.** A model that repeats "the the the the" might have decent perplexity because "the" is a common word. But the text is useless.

2. **It does not measure coherence.** A model could predict common words accurately but fail to maintain topic over a paragraph.

3. **It does not measure factual accuracy.** A model might confidently predict the wrong fact.

This is why we use additional metrics in Chapter 7.2 and 7.3.

## Reference Values

| Model | Perplexity | Notes |
|---|---|---|
| Random (32K vocab) | ~32,000 | All tokens equally likely |
| Our model, step 0 | 37,780 | Worse than random due to initialization quirks |
| Our model, step 60K | 33.69 | Good for 118M parameters |
| GPT-2 Small (124M) | ~29 | Slightly better (more training, bigger context) |
| GPT-2 Large (774M) | ~22 | 6.5× more parameters |
