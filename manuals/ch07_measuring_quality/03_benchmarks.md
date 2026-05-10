# Chapter 7.3 — Benchmarks: Comparing Against Published Models

## Why Benchmarks Matter

Perplexity and diversity metrics measure our model in isolation. **Benchmarks** compare it against published models on standardized tasks. This answers: "Where does our 118M model actually stand?"

## Common Language Model Benchmarks

### HellaSwag (Commonsense Reasoning)

The model is given a sentence and must choose the most likely continuation from 4 options:

> "A woman is outside with a bucket. She pours water from the bucket onto..."
> - A) her head
> - B) the garden
> - C) the sky
> - D) the bucket itself

**What it measures:** Common sense and physical reasoning
**Expected score for our model:** ~28-32% (random = 25%)
**GPT-2 Small (124M):** ~33%
**GPT-3 (175B):** ~78%

### ARC (AI2 Reasoning Challenge)

Multiple-choice science questions:

> "Which property of a material determines whether it will float or sink in water?"
> - A) hardness B) mass C) density D) temperature

**What it measures:** Scientific reasoning
**Expected score for our model:** ~22-25% (random = 25% for 4-choice)
**Note:** Very small models often score near random on ARC

### LAMBADA (Language Understanding)

The model must predict the last word of a passage where the answer requires understanding the full context:

> "She said she would never speak to him again. But when he appeared at her door with flowers, she couldn't help but ___"

**What it measures:** Long-range dependency understanding
**Expected score for our model:** Limited by our 384-token context window

### WinoGrande (Coreference Resolution)

> "The trophy didn't fit in the suitcase because it was too ___."
> Options: big / small

"it" refers to the trophy (big) or the suitcase (small)?

**What it measures:** Pronoun resolution, common sense

## How to Run Benchmarks

The `lm-evaluation-harness` by EleutherAI is the standard tool:

```bash
pip install lm-eval

lm_eval --model hf \
    --model_args pretrained=./checkpoints/best_model.pt \
    --tasks hellaswag,arc_easy,lambada_openai \
    --batch_size 8
```

> **Note:** Running benchmarks requires wrapping our model to be compatible with the harness API. Detailed instructions are in `advanced/evaluation_harness_guide.md`.

## What to Expect From a 118M Model

Honest expectations for our model size:

| Benchmark | Random | Our Model (Expected) | GPT-2 Small | GPT-3 175B |
|---|---|---|---|---|
| HellaSwag | 25% | ~28-32% | ~33% | ~78% |
| ARC Easy | 25% | ~25-30% | ~43% | ~68% |
| LAMBADA | 0% | ~15-25% | ~45% | ~76% |
| WinoGrande | 50% | ~50-53% | ~55% | ~70% |

Our model will score **slightly above random** on most benchmarks. This is expected and normal for a 118M-parameter model. These benchmarks are designed to test reasoning capabilities that emerge at larger scales (billions of parameters).

The value of running benchmarks on our small model is not to get high scores — it is to establish a **baseline** and confirm the model is learning meaningful patterns (even slightly above random is evidence of learning).

## The Scaling Perspective

Research has shown that benchmark performance follows predictable scaling laws:

- **Small models (100M-1B):** Slightly above random on reasoning tasks, decent on language tasks
- **Medium models (1B-10B):** Clear above-random performance, emerging reasoning
- **Large models (10B-100B):** Strong performance, near-human on many tasks
- **Very large models (100B+):** Approaching or exceeding human performance

Our model sits at the very beginning of this curve, which is exactly what we expect.
