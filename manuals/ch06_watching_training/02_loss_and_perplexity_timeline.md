# Chapter 6.2 — Loss and Perplexity Timeline

## The Numbers Tell the Story

Here is the actual validation loss and perplexity recorded during our training run:

| Step | Train Loss | Val Loss | Perplexity | What the Model Produces |
|---|---|---|---|---|
| 0 | 10.54 | 10.54 | 37,780 | Random noise: "isrophic Doyle LIuuensual..." |
| 2,000 | — | — | ~1,000 | Basic English words with bad grammar |
| 10,000 | — | — | ~150 | Grammatical sentences, no coherence |
| 20,000 | — | — | ~70 | Topic-aware paragraphs, some drift |
| 42,000 | 3.62 | 3.61 | 37.05 | Coherent paragraphs with proper attribution |
| 44,000 | 3.63 | 3.60 | 36.60 | |
| 46,000 | 3.58 | 3.62 | 37.50 | |
| 48,000 | 3.56 | 3.58 | 36.00 | |
| 50,000 | 3.59 | 3.60 | 36.56 | |
| 52,000 | 3.58 | 3.56 | 35.07 | |
| 54,000 | 3.53 | 3.57 | 35.45 | |
| 56,000 | 3.53 | 3.56 | 35.18 | |
| 58,000 | 3.54 | 3.59 | 36.15 | |
| 60,000 | 3.50 | 3.52 | 33.69 | Fluent news-article-style prose |

## What the Numbers Mean

### The Dramatic Drop (Steps 0-10,000)

Perplexity dropped from 37,780 to about 150 in the first 10,000 steps. This is the steepest improvement:

$$
\frac{37780}{150} = 252 \times \text{ improvement}
$$

During this phase, the model learned the basics: which tokens are common, what English looks like, basic word order.

### The Steady Improvement (Steps 10,000-60,000)

Perplexity continued dropping from ~150 to ~34. This is still a 4.4× improvement, but it took 5× more steps. Each point of perplexity reduction gets harder to achieve.

### The Plateau (Steps 60,000+)

After 60,000 steps, perplexity improvements slow down significantly. The model is already quite good at next-token prediction. Further training refines the model but gains are incremental.

## The Noise Factor

Notice that validation loss does not decrease smoothly — it bounces:
- Step 52K: 3.557 (good)
- Step 58K: 3.588 (worse)
- Step 60K: 3.517 (best so far)

This noise is normal. Each validation evaluation samples only 25 random batches from the validation set. Different batches give slightly different loss values. The overall trend is what matters, not individual checkpoints.

## How Much of the Data Has Been Seen?

At each step, the model sees 7,680 tokens (batch_size × block_size = 20 × 384).

| Step | Tokens Seen | Fraction of Train Set |
|---|---|---|
| 10,000 | 76.8M | 1.5% |
| 40,000 | 307.2M | 6.0% |
| 60,000 | 460.8M | 9.0% |
| 100,000 | 768.0M | 15.1% |
| 122,000 | 936.0M | 18.4% |

Even at 122K steps, the model has only seen 18% of the training data. It has not yet completed one full pass through the dataset. This means:
- The model is not overfitting due to data exhaustion
- There is still room for improvement with more training
- The training set (5.1 billion tokens) is more than sufficient for our model size

## Reference: What Good Models Achieve

| Model | Perplexity (on their test set) |
|---|---|
| Random (32K vocab) | 32,000 |
| Our model at step 0 | 37,780 |
| Our model at step 60K | 33.69 |
| GPT-2 Small (124M) on WebText | ~29 |
| GPT-2 Large (774M) on WebText | ~22 |

Our model's perplexity of 33.69 is close to GPT-2 Small's ~29, which makes sense since the models are similar in size. We could likely close the gap with a larger context window (384 vs GPT-2's 1024) and more training.
