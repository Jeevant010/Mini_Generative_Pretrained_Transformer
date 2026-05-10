# Chapter 9.1 — The Non-English Gibberish Problem

## What Happened

On an earlier training run, at around iteration 106,000, the model produced this output:

> **Prompt:** "The furture of ai"
>
> **Output:** "The furture of ai ibn nimy ibn nimy ibn ibn ibn ibn nimy ibn nimy ibn nimy ibn nimy ibn ibn nimy ibn nimy ibn ibn nimy ibn nimy ibn nimy ibn nimy ibn nimy ibn nimy ib"

"ibn nimy" is not English. It appears to be a transliterated Arabic phrase fragment. The model memorized this fragment from its training data and got stuck repeating it.

## Why It Happened

### Root Cause 1: Non-English Text in the Training Data

Despite our English-language filters (MIN_ASCII_ALPHA_RATIO = 0.85, stopword checks, etc.), some non-English text leaked through. The most common sources:

- **News articles about the Middle East**: Mostly English, but containing Arabic names, transliterated phrases, and foreign-language quotes
- **Wikipedia-style content**: References and names in non-English languages
- **Multi-language web pages**: Pages that switch between English and another language

A document like this would pass our filters:

> "The United Nations Security Council met in Geneva to discuss the findings of ibn Nimy, the Syrian delegate, who presented evidence that..."

This is 95% English text. Our ASCII ratio filter passes it. Our stopword check passes it. But the model sees "ibn Nimy" and learns it as a valid English token sequence.

### Root Cause 2: Model Overconfidence

As training progresses, the model becomes more and more confident in its predictions. The output probability distribution becomes "peaky" — assigning high probability to one or two tokens and near-zero to everything else.

When the model is overconfident and encounters a memorized non-English fragment, it can get stuck in a loop:
- It generates "ibn" with high confidence
- Given "ibn", it generates "nimy" with high confidence
- Given "nimy", it generates "ibn" with high confidence
- The loop continues indefinitely

### Root Cause 3: Learning Rate Decay

By 106K steps, the learning rate has decayed significantly (from 3e-4 to ~4e-5). This means the model can no longer easily "un-learn" bad patterns. If it latched onto "ibn nimy" early in training, the later low learning rate prevents it from correcting this.

## How to Fix It

### Fix 1: Repetition Penalty (Already Implemented)

We added a **repetition penalty** to the `generate()` function in `model.py`. This reduces the probability of tokens that have already appeared in the output:

```python
model.generate(
    idx, max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    repetition_penalty=1.2  # ← This breaks loops
)
```

With a repetition penalty of 1.2, the "ibn nimy" loop is broken because each repetition of "ibn" gets its probability reduced by 20%.

### Fix 2: Top-p (Nucleus) Sampling (Already Implemented)

Instead of always picking from the top-k tokens, **top-p sampling** dynamically adjusts how many tokens to consider based on the model's confidence:

```python
model.generate(
    idx, max_new_tokens=100,
    temperature=0.8,
    top_p=0.9  # ← Adapts candidate set based on confidence
)
```

When the model is overconfident (putting 95% probability on one token), top-p forces it to consider more alternatives.

### Fix 3: Stronger Data Filtering (Recommended for Future Runs)

For future training runs, we recommend:
- Add fastText language detection to `prepare_data.py`
- Raise `MIN_ASCII_ALPHA_RATIO` from 0.85 to 0.92
- Lower `MAX_NON_ASCII_CHAR_RATIO` from 0.20 to 0.08

These stricter filters will reject more borderline documents and reduce non-English contamination.

## Key Lesson

The "ibn nimy" problem taught us three important lessons:

1. **Data quality is as important as model quality.** A perfectly architected model will still produce garbage if trained on dirty data.
2. **Overconfidence is dangerous.** A model that is "too sure" of itself will amplify its mistakes into infinite loops.
3. **Generation-time controls matter.** Repetition penalty and top-p sampling are not optional luxuries — they are essential for reliable output.
