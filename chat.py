"""Small command-line chat loop for an SFT checkpoint."""

import argparse

import torch

import config
from model import GPTLanguageModel
from tokenizer import BytePairTokenizer


def build_prompt(user_text: str, history: list[tuple[str, str]]) -> str:
    parts = []
    for user, assistant in history:
        parts.append(f"User: {user}\n\nAssistant: {assistant}\n")
    parts.append(f"User: {user_text}\n\nAssistant:")
    return "\n".join(parts)


def clean_response(text: str) -> str:
    for marker in ("User:", "<eos>", "<bos>", "<pad>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with the SFT model.")
    parser.add_argument("--checkpoint", default="checkpoints/sft/best_sft_model.pt")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    args = parser.parse_args()

    tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)
    model = GPTLanguageModel(config).to(config.device)
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    eos_id = tokenizer.special_to_id.get("<eos>")
    history: list[tuple[str, str]] = []

    print("Mini GPT SFT chat. Type 'quit' to exit.")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"q", "quit", "exit"}:
            break

        prompt = build_prompt(user_text, history)
        input_ids = tokenizer.encode(prompt, add_bos=True)
        input_ids = input_ids[-config.block_size :]
        x = torch.tensor(input_ids, dtype=torch.long, device=config.device).unsqueeze(0)

        with torch.no_grad():
            output = model.generate(
                x,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop_token_id=eos_id,
            )

        new_ids = output[0][len(input_ids) :].tolist()
        response = clean_response(tokenizer.decode(new_ids, skip_special_tokens=False))
        print(f"Assistant: {response}")
        history.append((user_text, response))

        # Keep short context; block_size is only 384 in the current config.
        history = history[-3:]


if __name__ == "__main__":
    main()
