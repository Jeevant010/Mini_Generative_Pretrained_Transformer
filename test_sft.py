import os
import torch
import config
from model import GPTLanguageModel
from tokenizer import BytePairTokenizer

def test_sft():
    device = config.device
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)

    # Load SFT model
    checkpoint_path = "checkpoints/sft/best_sft_model.pt"
    print(f"Loading SFT checkpoint: {checkpoint_path}")
    
    model = GPTLanguageModel(config).to(device)
    ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt_data['model_state_dict'] if 'model_state_dict' in ckpt_data else ckpt_data
    model.load_state_dict(state_dict)
    model.eval()

    # The exact format the model was trained on!
    prompt = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
    print(f"\n[PROMPT]\n{prompt}")
    
    context_ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    output_ids = model.generate(x, max_new_tokens=100, temperature=0.7, top_k=50)
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    
    print(f"\n[OUTPUT]\n{generated_text}")

if __name__ == "__main__":
    test_sft()
