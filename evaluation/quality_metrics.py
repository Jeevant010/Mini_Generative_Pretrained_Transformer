"""
evaluation/quality_metrics.py — Comprehensive text generation quality evaluation.

Computes metrics beyond perplexity to detect overfitting, degeneration, and
repetition. Designed to run across multiple checkpoints to show training progression.

Metrics computed:
    - Distinct-1, Distinct-2, Distinct-3 (lexical diversity)
    - Self-BLEU (inter-sample repetitiveness)
    - Output Entropy (prediction confidence distribution)
    - Repetition Ratio (intra-sample n-gram repetition)
    - Max Repeated N-gram Length (longest repeated pattern)

Usage:
    python -m evaluation.quality_metrics
    python -m evaluation.quality_metrics --checkpoint checkpoints/ckpt_step_40000.pt
    python -m evaluation.quality_metrics --all-checkpoints
"""

import os
import math
import argparse
from collections import Counter

import torch
import torch.nn.functional as F

import config
from model import GPTLanguageModel
from tokenizer import BytePairTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Distinct-N (Lexical Diversity)
# ─────────────────────────────────────────────────────────────────────────────

def distinct_n(text: str, n: int) -> float:
    """
    Calculate Distinct-N score: fraction of unique n-grams.
    Higher = more diverse text. Lower = more repetitive.

    Reference: Li et al. (2016), "A Diversity-Promoting Objective Function
    for Neural Conversation Models."
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


# ─────────────────────────────────────────────────────────────────────────────
# Self-BLEU (Inter-Sample Repetitiveness)
# ─────────────────────────────────────────────────────────────────────────────

def _modified_precision(hypothesis: list, references: list, n: int) -> float:
    """Modified n-gram precision for BLEU."""
    hyp_ngrams = Counter(tuple(hypothesis[i:i+n]) for i in range(len(hypothesis) - n + 1))
    if not hyp_ngrams:
        return 0.0
    ref_ngrams = Counter()
    for ref in references:
        ref_count = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
        ref_ngrams |= ref_count
    clipped = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
    total = sum(hyp_ngrams.values())
    return clipped / total if total > 0 else 0.0


def bleu_score(hypothesis: list, references: list, max_n: int = 4) -> float:
    """Simplified BLEU score (without brevity penalty)."""
    precisions = []
    for n in range(1, max_n + 1):
        p = _modified_precision(hypothesis, references, n)
        if p == 0:
            return 0.0
        precisions.append(p)
    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    return math.exp(log_avg)


def self_bleu(samples: list) -> float:
    """
    Self-BLEU: average BLEU of each sample against all other samples.
    Lower = more diverse. Higher = more repetitive.

    Reference: Zhu et al. (2018), "Texygen: A Benchmarking Platform for
    Text Generation Models."
    """
    if len(samples) < 2:
        return 0.0
    tokenized = [s.split() for s in samples]
    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [s for j, s in enumerate(tokenized) if j != i]
        if not hyp or not refs:
            continue
        scores.append(bleu_score(hyp, refs))
    return sum(scores) / len(scores) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Output Entropy
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def output_entropy(model, input_ids: torch.Tensor) -> float:
    """
    Average output entropy (in bits) across all positions.
    Lower entropy = more confident/repetitive predictions.
    Healthy range: 5-10 bits for a 32K vocabulary.
    """
    logits, _ = model(input_ids)
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log2(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Repetition Metrics
# ─────────────────────────────────────────────────────────────────────────────

def repetition_ratio(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are repeated (1 - Distinct-N)."""
    return 1.0 - distinct_n(text, n)


def max_repeated_ngram_length(text: str, max_n: int = 15) -> int:
    """Find the length of the longest n-gram that repeats in the text."""
    tokens = text.split()
    for n in range(min(max_n, len(tokens) // 2), 0, -1):
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if len(ngrams) != len(set(ngrams)):
            return n
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Bits Per Character (BPC)
# ─────────────────────────────────────────────────────────────────────────────

def bits_per_character(avg_loss: float) -> float:
    """Convert average cross-entropy loss to bits per character."""
    return avg_loss / math.log(2)


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_samples(model, tokenizer, prompts, max_tokens=80,
                     temperature=0.8, top_k=50):
    """Generate text samples from fixed prompts."""
    device = next(model.parameters()).device
    model.eval()
    samples = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        idx = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(
                idx, max_new_tokens=max_tokens,
                temperature=temperature, top_k=top_k,
            )
        text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        samples.append(text)
    return samples


def evaluate_checkpoint(model, tokenizer, step=None):
    """Run full quality evaluation and return metrics dict."""
    prompts = getattr(config, "SAMPLE_PROMPTS", [
        "The future of artificial intelligence is",
        "Once upon a time in a land far away",
        "In the beginning, there was nothing but",
    ])

    # Generate samples
    samples = generate_samples(model, tokenizer, prompts)

    # Compute per-sample metrics
    d1_scores = [distinct_n(s, 1) for s in samples]
    d2_scores = [distinct_n(s, 2) for s in samples]
    d3_scores = [distinct_n(s, 3) for s in samples]
    rep_scores = [repetition_ratio(s, 3) for s in samples]
    max_rep = [max_repeated_ngram_length(s) for s in samples]

    # Compute aggregate metrics
    metrics = {
        "step": step,
        "distinct_1": sum(d1_scores) / len(d1_scores),
        "distinct_2": sum(d2_scores) / len(d2_scores),
        "distinct_3": sum(d3_scores) / len(d3_scores),
        "self_bleu": self_bleu(samples),
        "repetition_ratio_3": sum(rep_scores) / len(rep_scores),
        "max_repeated_ngram": max(max_rep),
        "samples": list(zip(prompts, samples)),
    }

    # Compute entropy on a sample input
    try:
        sample_text = "The future of artificial intelligence is"
        sample_ids = tokenizer.encode(sample_text)
        idx = torch.tensor([sample_ids], dtype=torch.long,
                           device=next(model.parameters()).device)
        metrics["output_entropy_bits"] = output_entropy(model, idx)
    except Exception:
        metrics["output_entropy_bits"] = None

    # Compute validation perplexity
    try:
        from evaluation.perplexity import calculate_perplexity
        ppl_result = calculate_perplexity(model, split="val", num_batches=25)
        metrics["val_loss"] = ppl_result["avg_loss"]
        metrics["perplexity"] = ppl_result["perplexity"]
        metrics["bpc"] = bits_per_character(ppl_result["avg_loss"])
    except Exception as e:
        metrics["val_loss"] = None
        metrics["perplexity"] = None
        metrics["bpc"] = None

    return metrics


def print_report(metrics):
    """Print a formatted evaluation report."""
    step_str = f"Step {metrics['step']}" if metrics.get("step") else "Current"

    print(f"\n{'='*65}")
    print(f"  QUALITY EVALUATION REPORT — {step_str}")
    print(f"{'='*65}")

    # Language modeling metrics
    if metrics.get("val_loss") is not None:
        print(f"\n{'-'*40}")
        print(f"  Language Modeling")
        print(f"{'-'*40}")
        print(f"  Validation Loss      : {metrics['val_loss']:.4f}")
        print(f"  Perplexity (PPL)     : {metrics['perplexity']:.2f}")
        print(f"  Bits Per Character   : {metrics['bpc']:.2f}")

    # Generation quality metrics
    print(f"\n{'-'*40}")
    print(f"  Generation Quality")
    print(f"{'-'*40}")
    print(f"  Distinct-1           : {metrics['distinct_1']:.4f}  (> 0.5 good)")
    print(f"  Distinct-2           : {metrics['distinct_2']:.4f}  (> 0.6 good)")
    print(f"  Distinct-3           : {metrics['distinct_3']:.4f}  (> 0.7 good)")
    print(f"  Self-BLEU            : {metrics['self_bleu']:.4f}  (< 0.4 good)")
    print(f"  Repetition Ratio (3) : {metrics['repetition_ratio_3']:.4f}  (< 0.4 good)")
    print(f"  Max Repeated N-gram  : {metrics['max_repeated_ngram']}")
    if metrics.get("output_entropy_bits") is not None:
        print(f"  Output Entropy       : {metrics['output_entropy_bits']:.2f} bits  (5-10 healthy)")

    # Overall assessment
    print(f"\n{'-'*40}")
    print(f"  Assessment")
    print(f"{'-'*40}")

    issues = []
    if metrics["distinct_2"] < 0.3:
        issues.append("[!] LOW DIVERSITY: Distinct-2 < 0.3 \u2014 text is repetitive")
    if metrics["self_bleu"] > 0.5:
        issues.append("[!] HIGH SELF-BLEU: > 0.5 \u2014 outputs are too similar to each other")
    if metrics["max_repeated_ngram"] > 5:
        issues.append(f"[!] LONG REPEAT: {metrics['max_repeated_ngram']}-gram repeats detected")
    if metrics.get("output_entropy_bits") and metrics["output_entropy_bits"] < 3:
        issues.append("[!] LOW ENTROPY: < 3 bits \u2014 model is overconfident")
    if metrics.get("perplexity") and metrics["perplexity"] > 100:
        issues.append("[!] HIGH PPL: > 100 \u2014 model still has much to learn")

    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  [OK] All metrics in healthy range")

    # Sample previews
    print(f"\n{'-'*40}")
    print(f"  Generated Samples")
    print(f"{'-'*40}")
    for prompt, sample in metrics.get("samples", []):
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Output:\n{sample}")

    print(f"\n{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generation quality metrics across checkpoints."
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint.")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Evaluate all checkpoints in checkpoints/.")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Custom prompts (overrides config).")
    args = parser.parse_args()

    device = config.device
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)

    if args.prompts:
        config.SAMPLE_PROMPTS = args.prompts

    if args.all_checkpoints:
        # Evaluate all checkpoints
        ckpt_dir = "checkpoints"
        if not os.path.exists(ckpt_dir):
            print("No checkpoints directory found.")
            return

        ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_step_")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        if not ckpts:
            print("No checkpoints found.")
            return

        print(f"\nEvaluating {len(ckpts)} checkpoints...")
        all_metrics = []

        for ckpt_name in ckpts:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            model = GPTLanguageModel(config).to(device)
            ckpt_data = torch.load(ckpt_path, map_location=device,
                                   weights_only=False)
            model.load_state_dict(ckpt_data["model_state_dict"])
            step = ckpt_data.get("step", 0)

            metrics = evaluate_checkpoint(model, tokenizer, step=step)
            all_metrics.append(metrics)
            print_report(metrics)

            del model
            if "cuda" in str(device):
                torch.cuda.empty_cache()

        # Print comparison table
        print(f"\n{'='*90}")
        print(f"  CHECKPOINT COMPARISON TABLE")
        print(f"{'='*90}")
        header = (
            f"{'Step':>8} | {'PPL':>8} | {'BPC':>6} | "
            f"{'D-1':>6} | {'D-2':>6} | {'D-3':>6} | "
            f"{'SelfBLEU':>8} | {'RepRatio':>8} | {'MaxRep':>6} | {'Entropy':>7}"
        )
        print(header)
        print("-" * 90)
        for m in all_metrics:
            ppl = f"{m['perplexity']:.1f}" if m.get("perplexity") else "N/A"
            bpc = f"{m['bpc']:.2f}" if m.get("bpc") else "N/A"
            entropy = f"{m['output_entropy_bits']:.1f}" if m.get("output_entropy_bits") else "N/A"
            print(
                f"{m.get('step', 0):>8} | {ppl:>8} | {bpc:>6} | "
                f"{m['distinct_1']:>6.3f} | {m['distinct_2']:>6.3f} | "
                f"{m['distinct_3']:>6.3f} | "
                f"{m['self_bleu']:>8.3f} | {m['repetition_ratio_3']:>8.3f} | "
                f"{m['max_repeated_ngram']:>6} | {entropy:>7}"
            )
        print(f"{'='*90}\n")

    else:
        # Single checkpoint evaluation
        model = GPTLanguageModel(config).to(device)

        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device,
                              weights_only=False)
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
            step = ckpt.get("step", None)
            print(f"Loaded: {args.checkpoint} (step {step})")
        else:
            ckpt_dir = "checkpoints"
            if os.path.exists(ckpt_dir):
                ckpts = [f for f in os.listdir(ckpt_dir)
                         if f.startswith("ckpt_step_")]
                if ckpts:
                    latest = sorted(
                        ckpts,
                        key=lambda x: int(x.split("_")[-1].split(".")[0])
                    )[-1]
                    path = os.path.join(ckpt_dir, latest)
                    ckpt = torch.load(path, map_location=device,
                                      weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"])
                    step = ckpt.get("step", None)
                    print(f"Loaded latest: {path} (step {step})")

        metrics = evaluate_checkpoint(model, tokenizer, step=step)
        print_report(metrics)


if __name__ == "__main__":
    main()
