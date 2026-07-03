import os
import argparse
import torch

import config
from model import GPTLanguageModel
from tokenizer import BytePairTokenizer

def get_latest_checkpoint():
    """Finds the most advanced checkpoint available."""
    dpo_dir = "checkpoints/dpo"
    sft_dir = "checkpoints/sft"
    
    if os.path.exists(dpo_dir):
        if os.path.exists(os.path.join(dpo_dir, "best_dpo_model.pt")):
            return os.path.join(dpo_dir, "best_dpo_model.pt")
            
    if os.path.exists(sft_dir):
        if os.path.exists(os.path.join(sft_dir, "best_sft_model.pt")):
            return os.path.join(sft_dir, "best_sft_model.pt")
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Interactive Chat with the Model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    device = config.device
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)

    # Load Model
    ckpt_path = args.checkpoint or get_latest_checkpoint()
    if not ckpt_path:
        print("No checkpoint found! Please train the model first.")
        return

    print(f"Loading model from: {ckpt_path}")
    model = GPTLanguageModel(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    print("\n" + "="*50)
    print("🤖 CHAT INTERFACE READY")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        # For now, we use the SFT prompt format. 
        # (We will update this to Chat Templates later)
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        
        input_ids = tokenizer.encode(prompt, add_bos=True)
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                idx, 
                max_new_tokens=200, 
                temperature=0.7, 
                top_p=0.9, 
                repetition_penalty=1.25,
                stop_token_id=tokenizer.special_to_id.get("<eos>", None)
            )
            
        # Decode and print ONLY the new generated tokens (skip the prompt)
        generated_ids = output[0].tolist()[len(input_ids):]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"\nModel:\n{response_text.strip()}")

if __name__ == "__main__":
    main()
