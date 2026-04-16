"""
generate.py — Text generation script for the production-ready Transformer.
"""
import os
import argparse
import torch
import config
from tokenizer import BytePairTokenizer
from model import GPTLanguageModel

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_step_")]
    if not ckpts:
        return None
    latest = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
    return os.path.join(checkpoint_dir, latest)

def main():
    parser = argparse.ArgumentParser(description="Generate text with the trained model.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Seed prompt.")
    parser.add_argument("--max-tokens", type=int, default=100, help="New tokens to generate.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (defaults to latest).")
    args = parser.parse_args()

    # 1. Setup Device
    device = config.device
    print(f"Using device: {device}")

    # 2. Load Tokenizer
    if not os.path.exists(config.TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {config.TOKENIZER_PATH}. Train it first!")
    tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)
    print(f"Loaded tokenizer from {config.TOKENIZER_PATH}")

    # 3. Load Model
    checkpoint_path = args.checkpoint or get_latest_checkpoint()
    if not checkpoint_path:
        raise FileNotFoundError("No checkpoint found. Please train the model first.")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = GPTLanguageModel(config).to(device)
    
    # Check if checkpoint contains full state or just weights
    ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt_data['model_state_dict'] if 'model_state_dict' in ckpt_data else ckpt_data
    
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model ready for inference.")

    # 4. Generate
    print(f"\nPrompt: {args.prompt}")
    print(f"{'─'*50}")
    
    # Encode prompt
    context_ids = tokenizer.encode(args.prompt, add_bos=True)
    x = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Predict
    output_ids = model.generate(x, max_new_tokens=args.max_tokens, temperature=0.8, top_k=50)
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    print(generated_text)
    print(f"{'─'*50}")

if __name__ == "__main__":
    main()
