# Training Progression

## Current Long Run

The current important run is the `subset_10gb` training run.

| Item | Value |
| --- | ---: |
| Planned steps | 150,000 |
| Latest observed step | 60,000 |
| Tokens per step | 7,680 |
| Token positions seen by 60k | 460,800,000 |
| Train tokens | 5,100,766,548 |
| Fraction of train tokens by 60k | 0.0903 |

## Loss Progress

At initialization:

| Step | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 0 | 10.539526 | 37,779.67 |

At the latest observed checkpoint:

| Step | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 60,000 | 3.517095 | 33.69 |

Perplexity reduction factor:

$$
\frac{37779.67}{33.69} \approx 1121.4
$$

## Recent Validation Records

| Step | Train loss | Validation loss | PPL |
| ---: | ---: | ---: | ---: |
| 42,000 | 3.618999 | 3.612168 | 37.05 |
| 44,000 | 3.625918 | 3.600169 | 36.60 |
| 46,000 | 3.583047 | 3.624296 | 37.50 |
| 48,000 | 3.561008 | 3.583526 | 36.00 |
| 50,000 | 3.592763 | 3.598916 | 36.56 |
| 52,000 | 3.580003 | 3.557480 | 35.07 |
| 54,000 | 3.527637 | 3.568154 | 35.45 |
| 56,000 | 3.528134 | 3.560501 | 35.18 |
| 58,000 | 3.537106 | 3.587802 | 36.15 |
| 60,000 | 3.504482 | 3.517095 | 33.69 |

The trend is still improving, though individual evaluations are noisy because they average sampled batches.

## Interpreting Samples

The prompt:

```text
how can i help
```

produced fluent continuation-style text by 40k and 60k. This is a good sign for base pretraining, but it does not mean the model is instruction-tuned.

The base model solves:

$$
P(\text{next token} \mid \text{previous tokens})
$$

not:

$$
P(\text{assistant response} \mid \text{user request})
$$

To get chatbot behavior, add a supervised fine-tuning stage with instruction-response data.

## Milestone Interpretation

| Step range | Expected behavior |
| --- | --- |
| 0 | Random token predictions |
| 5k-10k | Basic token statistics and common fragments |
| 20k-30k | More fluent local syntax |
| 40k-60k | Coherent paragraphs, topic drift still common |
| 100k+ | Better consistency expected if validation loss continues improving |

## Recommended Plots

For the paper, plot from `logs/training_metrics.csv`:

- train loss vs step
- validation loss vs step
- perplexity vs step
- tokens/sec vs step
- VRAM vs step
- gradient norm vs step

