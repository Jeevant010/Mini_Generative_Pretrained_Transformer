"""
evaluation/sample_generator.py — Periodic text generation during training.

Generates text from fixed prompts at each evaluation interval so you can
physically watch the model evolve from gibberish → coherent English.

Samples are saved to logs/samples/step_<N>.txt.

Usage (standalone):
    python -m evaluation.sample_generator --checkpoint checkpoints/best_model.pt

Usage (from training.py):
    from evaluation.sample_generator import generate_and_log_samples
    generate_and_log_samples(model, tokenizer, step, config)
"""

import os
import torch

import config


def generate_and_log_samples(model, tokenizer, step, cfg=None):
    """
    Generate text from fixed prompts and save to disk.

    Args:
        model: GPTLanguageModel instance.
        tokenizer: BytePairTokenizer instance with encode/decode.
        step: Current training step (for filename).
        cfg: Config module (defaults to global config).

    Returns:
        list of (prompt, generated_text) tuples.
    """
    if cfg is None:
        cfg = config

    prompts = getattr(cfg, "SAMPLE_PROMPTS", ["Once upon a time"])
    max_tokens = getattr(cfg, "SAMPLE_MAX_TOKENS", 80)
    temperature = getattr(cfg, "SAMPLE_TEMPERATURE", 0.8)
    top_k = getattr(cfg, "SAMPLE_TOP_K", 50)
    log_dir = getattr(cfg, "LOG_DIR", "logs")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    results = []
    lines = [f"Step {step} — Generated Samples", "=" * 60, ""]

    for i, prompt in enumerate(prompts):
        try:
            token_ids = tokenizer.encode(prompt)
            idx = torch.tensor([token_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                output = model.generate(
                    idx,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )

            generated_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            results.append((prompt, generated_text))

            lines.append(f"Prompt {i+1}: \"{prompt}\"")
            lines.append(f"Output:   {generated_text}")
            lines.append("-" * 60)
            lines.append("")
        except Exception as e:
            results.append((prompt, f"[ERROR: {e}]"))
            lines.append(f"Prompt {i+1}: \"{prompt}\"")
            lines.append(f"Output:   [ERROR: {e}]")
            lines.append("-" * 60)
            lines.append("")

    # Save to disk
    samples_dir = os.path.join(log_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sample_path = os.path.join(samples_dir, f"step_{step}.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if was_training:
        model.train()

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate text samples from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    from model import GPTLanguageModel
    from tokenizer import BytePairTokenizer

    device = config.device
    model = GPTLanguageModel(config).to(device)

    # Load checkpoint
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        step = ckpt.get("step", 0)
    else:
        step = 0

    tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)

    results = generate_and_log_samples(model, tokenizer, step)

    print(f"\n{'='*60}")
    print(f"📝 TEXT SAMPLES (step {step})")
    print(f"{'='*60}")
    for prompt, text in results:
        print(f"\nPrompt: \"{prompt}\"")
        print(f"Output: {text}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
