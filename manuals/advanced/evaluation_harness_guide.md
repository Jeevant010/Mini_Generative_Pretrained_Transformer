# Evaluation Harness Guide

## What Is an Evaluation Harness?

An evaluation harness is a standardized framework that tests a language model against established academic benchmarks. Instead of relying solely on training loss or perplexity — which only measure how well the model predicts the next token on its own data — a harness evaluates the model's understanding of language, reasoning, and world knowledge against curated test sets with known correct answers.

The industry standard is **EleutherAI's lm-evaluation-harness** (`lm-eval`), the same framework used to evaluate GPT-2, LLaMA, Mistral, and virtually every published open-source model.

## Why Use It?

Perplexity tells you the model's average surprise on the validation set. It does not tell you:

- Whether the model can complete a sentence that requires commonsense reasoning
- Whether the model can distinguish correct facts from plausible-sounding false ones
- Whether the model has learned grammar at a structural level or is just memorizing n-grams

The evaluation harness tests these specific capabilities. Reporting harness scores alongside perplexity gives a much more complete picture of model quality.

## Key Benchmarks for a Small (85M) Model

Not all benchmarks are meaningful for a small model. Here are the ones that provide useful signal:

### Tier 1: Directly Useful

| Benchmark | What It Tests | Format | Why It Matters |
|---|---|---|---|
| **HellaSwag** | Commonsense sentence completion | Multiple choice (4 options) | Tests if the model understands plausible event sequences |
| **ARC-Easy** | Elementary science questions | Multiple choice | Tests basic factual knowledge and reasoning |
| **WinoGrande** | Pronoun resolution (coreference) | Binary choice | Tests syntactic and semantic understanding |
| **LAMBADA** | Long-range word prediction | Predict last word of passage | Tests context understanding over many sentences |
| **BoolQ** | Yes/no reading comprehension | Binary classification | Tests passage understanding |

### Tier 2: Informative but Expect Low Scores

| Benchmark | What It Tests | Why Scores Will Be Low |
|---|---|---|
| **MMLU** | Multi-domain knowledge (57 subjects) | Requires extensive world knowledge an 85M model cannot memorize |
| **ARC-Challenge** | Hard science questions | Requires deep reasoning beyond small model capacity |
| **TruthfulQA** | Resistance to common misconceptions | Small models strongly echo training data biases |

### Expected Score Ranges

| Benchmark | Random Baseline | 85M Model (Expected) | GPT-2 124M | GPT-2 774M |
|---|---|---|---|---|
| HellaSwag | 25.0% | 28–32% | 31.6% | 47.2% |
| ARC-Easy | 25.0% | 30–38% | 43.6% | 51.5% |
| WinoGrande | 50.0% | 50–52% | 52.2% | 55.4% |
| LAMBADA (acc) | 0% | 15–25% | 32.8% | 47.2% |
| BoolQ | 50.0% | 50–55% | 60.3% | 65.3% |

These numbers are important context. Your 85M model is smaller than GPT-2 Small (124M) and was trained on less data. Scoring above random on HellaSwag, ARC-Easy, and LAMBADA proves the model has learned meaningful language patterns.

## Setup

### Step 1: Install lm-evaluation-harness

```bash
pip install lm-eval
```

Or clone the repository for the latest version:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[hf]"
```

### Step 2: Convert Your Model to HuggingFace Format

The harness expects a model loadable via `transformers.AutoModelForCausalLM`. Your custom model uses a different architecture, so you need a conversion wrapper.

Create a file `export_to_hf.py` in the project root:

```python
"""
export_to_hf.py — Convert the custom GPT checkpoint to HuggingFace format.

This creates a directory with config.json and pytorch_model.bin that
lm-evaluation-harness can load.

Usage:
    python export_to_hf.py --checkpoint checkpoints/best_model.pt --output hf_model/
"""
import os
import json
import argparse
import torch

import config
from model import GPTLanguageModel


def export(checkpoint_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = GPTLanguageModel(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    # Save weights
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # Save config
    hf_config = {
        "architectures": ["GPTLanguageModel"],
        "model_type": "custom-gpt",
        "vocab_size": config.vocab_size,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_kv_heads": config.n_kv_heads,
        "block_size": config.block_size,
        "dropout": config.dropout,
        "ffn_mult": config.ffn_mult,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    print(f"Model exported to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="hf_model/")
    args = parser.parse_args()
    export(args.checkpoint, args.output)
```

### Step 3: Write a Custom Model Wrapper

Since your architecture is not a standard HuggingFace model, you need to write a custom wrapper that lm-eval can use. The simplest approach is to use the `local` model type with a custom script.

Alternatively, run evaluations directly using the perplexity-based evaluation code in `evaluation/quality_metrics.py` (see the Generation Quality Metrics manual for details).

### Step 4: Run the Harness

```bash
# Evaluate on key benchmarks
lm_eval --model hf \
    --model_args pretrained=./hf_model,trust_remote_code=True \
    --tasks hellaswag,arc_easy,winogrande,lambada_openai,boolq \
    --device cuda:0 \
    --batch_size 1 \
    --output_path eval_results/
```

### Step 5: Read the Results

The harness outputs a JSON file with scores for each task. The key fields are:

```json
{
  "results": {
    "hellaswag": {
      "acc": 0.2912,
      "acc_norm": 0.3145,
      "acc_stderr": 0.0045
    }
  }
}
```

- `acc`: Raw accuracy (fraction of correct answers)
- `acc_norm`: Length-normalized accuracy (preferred for multiple-choice tasks)
- `acc_stderr`: Standard error (tells you how confident the score is)

## Interpreting Results Mathematically

### Is the Model Better Than Random?

For a 4-way multiple choice task like HellaSwag, random baseline is 25%. If your model scores 30%, the question is: is that statistically significant?

Use a one-proportion z-test:

$$
z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}
$$

Where:
- $\hat{p}$ = your model's accuracy (0.30)
- $p_0$ = random baseline (0.25)
- $n$ = number of test examples

For HellaSwag ($n \approx 10{,}000$):

$$
z = \frac{0.30 - 0.25}{\sqrt{\frac{0.25 \times 0.75}{10000}}} = \frac{0.05}{0.00433} \approx 11.55
$$

A z-score of 11.55 means the improvement over random is extremely statistically significant ($p < 10^{-30}$). Even a 1-2% improvement over random on a 10,000-sample benchmark is meaningful.

### Tracking Progress Across Checkpoints

Run the harness on multiple checkpoints to show improvement:

| Checkpoint | HellaSwag | ARC-Easy | LAMBADA |
|---|---|---|---|
| Step 10,000 | 25.2% | 26.1% | 3.2% |
| Step 40,000 | 28.5% | 31.4% | 12.8% |
| Step 80,000 | 30.1% | 34.2% | 18.5% |
| Step 106,000 | 29.8% | 33.1% | 17.2% |

If scores plateau or decrease at later checkpoints (like the drop at 106K above), it confirms overfitting — the model is getting better at predicting its training data but not at generalizing to new tasks.

## Alternative: Direct Evaluation Without HuggingFace Conversion

If converting to HuggingFace format is complex, you can implement the benchmarks directly using your model. The project's `evaluation/quality_metrics.py` provides a framework for this. For multiple-choice tasks:

1. Encode each answer option
2. Compute the model's log-probability for each option given the context
3. Select the option with highest log-probability
4. Compare against the gold answer

This is exactly what the harness does internally, but without the HuggingFace abstraction layer.

## References

- Gao et al. (2023). "A Framework for Few-Shot Language Model Evaluation." EleutherAI.
- https://github.com/EleutherAI/lm-evaluation-harness
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2 benchmark scores)
