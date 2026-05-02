# Experiments, Results, And Discussion

## Experimental Setup

The current experiment trains the `subset_10gb` preset:

| Setting | Value |
| --- | --- |
| Model parameters | 117,787,392 |
| Layers | 12 |
| Embedding dimension | 768 |
| Query heads | 12 |
| KV heads | 4 |
| Context length | 384 |
| Vocabulary size | 32,000 |
| Batch size | 20 |
| Tokens per step | 7,680 |
| Max planned steps | 150,000 |
| Latest observed step | 60,000 |
| Optimizer | AdamW |
| Peak learning rate | 2.5e-4 |
| Minimum learning rate | 2.5e-5 |
| Warmup | 2,000 steps |
| LR decay | Cosine |
| Gradient clipping | 1.0 |
| Dropout | 0.1 |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU |

The dataset target is 10 GB of tokenized data:

| Split | Tokens |
| --- | ---: |
| Train | 5,100,766,548 |
| Validation | 267,942,572 |

## Loss And Perplexity Progress

The training log shows strong improvement from step 0 to step 60,000.

At step 0:

| Metric | Value |
| --- | ---: |
| Train loss | 10.537739 |
| Validation loss | 10.539526 |
| Perplexity | 37,779.67 |

At step 60,000:

| Metric | Value |
| --- | ---: |
| Train loss | 3.504482 |
| Validation loss | 3.517095 |
| Perplexity | 33.69 |

The absolute validation-loss reduction is:

$$
10.539526 - 3.517095 = 7.022431
$$

The perplexity reduction factor is:

$$
\frac{37779.67}{33.69} \approx 1121.4
$$

This means the model's next-token uncertainty has decreased by more than three orders of magnitude compared with the untrained initialization.

## Recent Validation Trend

The most recent validation metrics are:

| Step | Train loss | Validation loss | Perplexity |
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

The model is still improving by step 60,000. The validation loss has noise, which is expected because each evaluation uses sampled batches, but the overall trend is downward.

## Token Exposure Analysis

The training file contains:

$$
5{,}100{,}766{,}548
$$

tokens.

At 60,000 steps with 7,680 tokens per step:

$$
N_{seen} = 60{,}000 \times 7{,}680 = 460{,}800{,}000
$$

The token-equivalent dataset fraction is:

$$
\frac{460{,}800{,}000}{5{,}100{,}766{,}548} \approx 0.0903
$$

Thus, the current run has processed only about 9 percent of one full token-equivalent pass through the training set. This is important for interpreting samples: the model is still early relative to dataset size.

## Qualitative Generation Analysis

Prompt:

```text
how can i help
```

Observed 40k-step style output:

```text
how can i help with it?

I think that the biggest consideration is it's got a lot more depth and depth than an all-time great team...
```

This output is reasonable for a raw pretrained model. It shows:

- The model has learned English syntax.
- The model can continue the prompt as a plausible phrase.
- The model has learned topic-like continuation patterns from web text.
- The model is not aligned as an assistant.
- The model sometimes repeats phrases such as "depth and depth".

The important interpretation is that base language models perform continuation, not instruction following. The prompt `how can i help` is not understood as a user intent in the way a chat model would understand it. Instead, it is treated as a prefix in a web-text distribution.

## Why The Output Is Not Yet ChatGPT-Like

The model was trained with a next-token objective:

$$
\max_\theta \sum_t \log P_\theta(x_t \mid x_{<t})
$$

It was not trained with an instruction-response objective such as:

```text
User: ...
Assistant: ...
```

Therefore, it has no explicit reason to answer as an assistant. To make it conversational, a second-stage fine-tuning dataset should contain examples such as:

```text
<bos>User: how can i help?
Assistant: You can ask me to explain a topic, write code, summarize text, or debug an error.<eos>
```

The instruction-tuning objective remains next-token prediction, but the data distribution changes. The model learns the conditional format:

$$
P(\text{assistant response} \mid \text{user instruction})
$$

instead of only:

$$
P(\text{next web-text token} \mid \text{previous web-text tokens})
$$

## Current Strengths

The current model demonstrates:

- Successful training from scratch on a multi-gigabyte dataset.
- Stable optimization over 60,000 steps.
- Large perplexity reduction from initialization.
- Working tokenizer, data pipeline, checkpointing, and generation.
- Fluent local text generation.
- Modern Transformer architecture on consumer GPU hardware.

## Current Limitations

The model still has several limitations:

- It is a base model, not an instruction-tuned assistant.
- Context length is limited to 384 tokens.
- It has only processed about 9 percent of one token-equivalent dataset pass by step 60,000.
- Generated text can be locally fluent but globally inconsistent.
- The model may hallucinate entities or facts because no retrieval or grounding mechanism is used.
- Sampling settings strongly affect output quality.
- The dataset may contain web-text artifacts, topic imbalance, and boilerplate despite filtering.

## Ablation Study Plan

The code contains toggles for:

```python
USE_RMSNORM
USE_ROPE
USE_FLASH_ATTENTION
USE_GQA
```

A paper-ready ablation table can compare:

| Experiment | RMSNorm | RoPE | Flash Attention | GQA | Expected observation |
| --- | --- | --- | --- | --- | --- |
| Full model | Yes | Yes | Yes | Yes | Best balance of stability, speed, and memory |
| No RMSNorm | No | Yes | Yes | Yes | Unstable gradients or worse convergence |
| No RoPE | Yes | No | Yes | Yes | Weaker position modeling |
| No Flash Attention | Yes | Yes | No | Yes | Higher memory usage and lower throughput |
| No GQA | Yes | Yes | Yes | No | More parameters and memory in attention |

Recommended metrics:

- Final validation loss after fixed steps.
- Perplexity after fixed steps.
- Tokens per second.
- Peak VRAM usage.
- Sample quality from fixed prompts.
- Whether training remains stable.

## Discussion

The results support the claim that a modern small language model can be trained from scratch on consumer hardware when architecture and data handling are carefully engineered. The model is large enough to learn meaningful syntax and topical continuation, but small enough to fit into an RTX 4060 Laptop GPU workflow.

The gap between raw pretraining and assistant behavior is also clear. The model can continue text, but it does not know that a short phrase is a user request. This is not a failure of the architecture; it is a mismatch between the training objective/data and the desired interaction style. A base model must first learn language statistics. Then instruction tuning can shape the same model into a task-following assistant.

## Future Work

The most useful next steps are:

- Continue training toward the planned 150,000 steps.
- Add a validation curve plot from `logs/training_metrics.csv`.
- Run ablations at a smaller fixed budget.
- Add instruction fine-tuning after base pretraining.
- Increase context length if VRAM permits.
- Compare different sampling settings such as temperature, top-k, and top-p.
- Add repetition penalties or no-repeat n-gram controls during generation.
- Build a cleaner evaluation suite with fixed prompts and human scoring.

## Conclusion

This project demonstrates a complete small-language-model training pipeline with modern GPT-style architecture and practical systems engineering. At 60,000 steps on a 10 GB tokenized subset, the model reaches validation loss 3.5171 and perplexity 33.69, showing substantial learning from initialization. The generated outputs are consistent with an early base model: fluent enough to show learned language structure, but not yet aligned for assistant-like interaction. The work provides a strong foundation for a research paper on efficient language-model training under consumer hardware constraints.

