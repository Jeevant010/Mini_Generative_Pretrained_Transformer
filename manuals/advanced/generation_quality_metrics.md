# Generation Quality Metrics

## Beyond Perplexity: Measuring What Matters

Perplexity measures how well the model predicts the next token on held-out data. It is necessary but insufficient. A model can have good perplexity and still produce degenerate text — repetitive loops, incoherent paragraphs, or parroted training data. This manual covers the metrics that catch these failure modes.

## Metric 1: Distinct-N (Lexical Diversity)

### What It Measures

Distinct-N measures how diverse the generated text is by counting unique n-grams as a fraction of total n-grams. It directly detects repetitive degeneration — the failure mode observed at steps 100K–106K in this project.

### Mathematical Definition

For a generated text with tokens $w_1, w_2, \ldots, w_T$, the set of n-grams is:

$$
\text{ngrams}(n) = \{(w_i, w_{i+1}, \ldots, w_{i+n-1}) \mid i = 1, \ldots, T-n+1\}
$$

Distinct-N is:

$$
\text{Distinct-}n = \frac{|\text{unique}(\text{ngrams}(n))|}{|\text{ngrams}(n)|}
$$

### Interpretation

| Distinct-N | Meaning |
|---|---|
| 1.0 | Every n-gram is unique — maximum diversity |
| 0.5–0.8 | Healthy diversity for natural text |
| 0.1–0.3 | Moderate repetition — model reuses many phrases |
| < 0.1 | Severe degeneration — model is stuck in loops |

### Example from This Project

Consider the 106K output: `"ibn nimy ibn nimy ibn nimy ibn nimy..."`:

- **Distinct-1**: Only 2 unique tokens out of ~20 → **0.10**
- **Distinct-2**: Only 2 unique bigrams out of ~19 → **0.11**

Compare with the 40K output (coherent paragraph):

- **Distinct-1**: ~45 unique tokens out of ~80 → **0.56**
- **Distinct-2**: ~65 unique bigrams out of ~79 → **0.82**

This quantitative difference — 0.11 vs 0.82 — is the mathematical proof that degeneration occurred.

### Implementation

```python
def distinct_n(text: str, n: int) -> float:
    """
    Calculate Distinct-N score for a text string.
    
    Args:
        text: Generated text string.
        n: N-gram size (1, 2, or 3).
    
    Returns:
        Float between 0 and 1. Higher = more diverse.
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    return len(set(ngrams)) / len(ngrams)
```

---

## Metric 2: Self-BLEU (Inter-Sample Repetitiveness)

### What It Measures

Self-BLEU measures how similar the model's different outputs are to each other. If you generate 10 samples from different prompts and they all sound the same, the model has collapsed to a narrow output distribution. Lower Self-BLEU means more diverse generation.

### Mathematical Definition

Given $N$ generated samples $S = \{s_1, s_2, \ldots, s_N\}$, Self-BLEU treats each sample as a hypothesis and all others as references:

$$
\text{Self-BLEU} = \frac{1}{N} \sum_{i=1}^{N} \text{BLEU}(s_i, S \setminus \{s_i\})
$$

Where $\text{BLEU}$ is the standard BLEU score (Papineni et al., 2002).

### Interpretation

| Self-BLEU | Meaning |
|---|---|
| < 0.2 | Excellent diversity — each output is distinct |
| 0.2–0.4 | Normal range for a language model |
| 0.4–0.6 | Moderate repetitiveness across samples |
| > 0.6 | Model has collapsed — all outputs look similar |

### Implementation

```python
from collections import Counter
import math

def bleu_score(hypothesis: list, references: list, max_n: int = 4) -> float:
    """Simplified BLEU score without brevity penalty."""
    scores = []
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter(tuple(hypothesis[i:i+n]) for i in range(len(hypothesis) - n + 1))
        ref_ngrams = Counter()
        for ref in references:
            ref_ngrams |= Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total = max(sum(hyp_ngrams.values()), 1)
        scores.append(clipped / total)
    
    if any(s == 0 for s in scores):
        return 0.0
    log_avg = sum(math.log(s) for s in scores) / len(scores)
    return math.exp(log_avg)


def self_bleu(samples: list) -> float:
    """
    Calculate Self-BLEU across a list of generated text strings.
    
    Args:
        samples: List of generated text strings.
    
    Returns:
        Float between 0 and 1. Lower = more diverse.
    """
    tokenized = [s.split() for s in samples]
    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [s for j, s in enumerate(tokenized) if j != i]
        scores.append(bleu_score(hyp, refs))
    return sum(scores) / len(scores) if scores else 0.0
```

---

## Metric 3: Output Entropy (Confidence Distribution)

### What It Measures

Output entropy measures how spread out the model's next-token predictions are. A healthy model should be moderately uncertain — considering multiple plausible continuations. A degenerate model becomes overconfident, placing nearly all probability on one token.

### Mathematical Definition

For the model's output probability distribution $P$ over vocabulary $V$ at each position:

$$
H(P) = -\sum_{v \in V} P(v) \log_2 P(v)
$$

For a vocabulary of size $|V| = 32{,}000$:

| Entropy (bits) | Meaning |
|---|---|
| $\log_2(32000) \approx 15$ | Maximum entropy — uniform random distribution |
| 8–12 | Healthy range — model is uncertain but informed |
| 5–8 | Confident predictions — good for well-trained models |
| 2–5 | Very confident — may indicate overtraining |
| < 2 | Degenerate — model always predicts same few tokens |

### Why Low Entropy Causes Repetition

When entropy drops below ~2 bits, the model effectively chooses from fewer than $2^2 = 4$ tokens at each step. This is why repetition loops occur — the model has become so confident in a few tokens that sampling always picks the same ones, creating `ibn nimy ibn nimy...` patterns.

### Implementation

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def output_entropy(model, input_ids: torch.Tensor) -> float:
    """
    Calculate average output entropy across all positions.
    
    Args:
        model: GPTLanguageModel instance.
        input_ids: Input token IDs tensor [1, seq_len].
    
    Returns:
        Average entropy in bits.
    """
    logits, _ = model(input_ids)
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log2(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch, seq_len]
    return entropy.mean().item()
```

---

## Metric 4: Repetition Ratio (Intra-Sample)

### What It Measures

Repetition ratio measures the fraction of n-grams in a single generated text that are repeated. Unlike Distinct-N (which looks at unique vs total), repetition ratio specifically counts how many tokens belong to repeated n-gram patterns.

### Mathematical Definition

$$
\text{RepRatio}(n) = 1 - \text{Distinct-}n
$$

This is the complement of Distinct-N, framed as "what fraction of n-grams are repetitions."

Additionally, a more targeted metric counts the longest repeated substring:

$$
\text{MaxRepeatLength} = \max_k \left\{ k : \exists \text{ substring of length } k \text{ that appears } \geq 2 \text{ times} \right\}
$$

### Implementation

```python
def repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are repeated (complement of Distinct-N)."""
    return 1.0 - distinct_n(text, n)


def max_repeated_ngram_length(text: str, max_n: int = 10) -> int:
    """Find the length of the longest n-gram that repeats in the text."""
    tokens = text.split()
    for n in range(max_n, 0, -1):
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if len(ngrams) != len(set(ngrams)):
            return n
    return 0
```

---

## Metric 5: Bits Per Character (BPC)

### What It Measures

BPC converts the cross-entropy loss into a character-level measure, making it comparable across different tokenizers and vocabularies.

### Mathematical Definition

$$
\text{BPC} = \frac{\mathcal{L}_{\text{val}}}{\ln 2}
$$

Where $\mathcal{L}_{\text{val}}$ is the average cross-entropy loss per token on the validation set.

For the current model at step 42,000:

$$
\text{BPC} = \frac{3.61}{0.693} \approx 5.21 \text{ bits}
$$

### Reference Values

| Model | BPC |
|---|---|
| Random (32K vocab) | ~15.2 |
| This project (step 42K) | ~5.21 |
| GPT-2 (124M, WebText) | ~1.76 |
| Human (estimated) | ~1.0–1.5 |

---

## Combining Metrics: The Quality Scorecard

No single metric tells the full story. The recommended approach is a combined scorecard evaluated at each checkpoint:

$$
\text{Quality Score} = w_1 \cdot \text{PPL}^{-1} + w_2 \cdot \text{Distinct-2} + w_3 \cdot (1 - \text{Self-BLEU}) + w_4 \cdot H_{\text{norm}}
$$

Where:
- $\text{PPL}^{-1}$ = inverse perplexity (higher is better)
- $\text{Distinct-2}$ = bigram diversity (higher is better)
- $(1 - \text{Self-BLEU})$ = inter-sample diversity (higher is better)
- $H_{\text{norm}}$ = entropy normalized to [0,1] range

This composite score captures both language modeling quality and generation quality.

## How These Metrics Detect Overfitting

| Symptom | PPL | Distinct-N | Self-BLEU | Entropy |
|---|---|---|---|---|
| Healthy training | Decreasing ↓ | Stable ~0.6 | Stable ~0.3 | Stable 5–10 |
| Overfitting begins | Val PPL increases ↑ | Starts dropping ↓ | Starts increasing ↑ | Starts dropping ↓ |
| Severe degeneration | Val PPL spikes | < 0.1 | > 0.6 | < 2 |
| Memorization | Train PPL ↓↓, Val PPL ↑ | May stay normal | May stay normal | Drops on training prompts |

### The Key Insight

Perplexity alone misses degeneration because the model can have low training perplexity while producing terrible text. The combination of PPL + Distinct-N + Entropy catches all failure modes:

1. **PPL catches underfitting** — the model hasn't learned enough
2. **Distinct-N catches repetition** — the model generates loops
3. **Self-BLEU catches mode collapse** — all outputs sound the same
4. **Entropy catches overconfidence** — the model is too sure of wrong answers

## Running the Full Evaluation

Use `evaluation/quality_metrics.py` to compute all metrics across checkpoints:

```bash
python -m evaluation.quality_metrics --checkpoint checkpoints/ckpt_step_40000.pt
python -m evaluation.quality_metrics --checkpoint checkpoints/ckpt_step_106000.pt
```

This produces a formatted report showing all metrics side-by-side.

## References

- Li et al. (2016). "A Diversity-Promoting Objective Function for Neural Conversation Models." (Distinct-N)
- Zhu et al. (2018). "Texygen: A Benchmarking Platform for Text Generation Models." (Self-BLEU)
- Holtzman et al. (2020). "The Curious Case of Neural Text Degeneration." (Repetition, Entropy, Top-p)
- Papineni et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation."
