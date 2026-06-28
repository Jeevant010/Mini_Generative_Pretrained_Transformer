import os
import json
import argparse
import sys
import torch

# Add root directory to path so we can import config and model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import config
from model import GPTLanguageModel

def export(checkpoint_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = GPTLanguageModel(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    # Save weights
    print(f"Saving PyTorch bin...")
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # Save config
    print(f"Generating HuggingFace config...")
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

    print(f"Model successfully exported to {output_dir}")
    print(f"You can now run lm-evaluation-harness on this directory!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # User requested to default to the best model or step 149k
    parser.add_argument("--checkpoint", default="../../checkpoints/ckpt_step_149000.pt", help="Path to checkpoint")
    parser.add_argument("--output", default="../../hf_model/", help="Output directory for HF format")
    args = parser.parse_args()
    
    export(args.checkpoint, args.output)
