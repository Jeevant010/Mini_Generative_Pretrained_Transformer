# chat.py Explained

This file lets you talk to the fine-tuned model from the command line.

It does not train the model.

It loads the SFT checkpoint and uses it to generate assistant replies.

## Where This File Fits

```text
sft_train.py
  saves checkpoints/sft/best_sft_model.pt

chat.py
  loads checkpoints/sft/best_sft_model.pt
  waits for your input
  wraps your text as User/Assistant format
  generates a reply
```

The important trick is that the model was trained on a format like:

```text
User: explain gravity simply

Assistant: Gravity is ...
```

So at chat time, if you type:

```text
hello
```

the script gives the model:

```text
User: hello

Assistant:
```

That tells the model:

```text
Now produce the assistant response.
```

## Lines 1 to 9: Imports

```python
"""Small command-line chat loop for an SFT checkpoint."""
```

This describes the file.

It is a simple terminal chat program for an SFT checkpoint.

```python
import argparse
```

This allows command-line options.

Example:

```powershell
python chat.py --checkpoint checkpoints/sft/best_sft_model.pt
```

```python
import torch
```

PyTorch is used to load the model checkpoint and create tensors.

```python
import config
from model import GPTLanguageModel
from tokenizer import BytePairTokenizer
```

These import your project code.

`config` tells the script model size, device, and block size.

`GPTLanguageModel` is the actual Transformer model.

`BytePairTokenizer` converts between text and token IDs.

## Lines 12 to 17: Build the Prompt

```python
def build_prompt(user_text: str, history: list[tuple[str, str]]) -> str:
```

This function builds the text that will be sent to the model.

It takes:

`user_text`: what you just typed.

`history`: recent previous user/assistant turns.

It returns one string.

```python
parts = []
```

Start with an empty list of text pieces.

```python
for user, assistant in history:
    parts.append(f"User: {user}\n\nAssistant: {assistant}\n")
```

For each previous turn, add it to the prompt.

Example history:

```text
User: hello

Assistant: Hello!

User: what is AI?

Assistant: AI is artificial intelligence.
```

This gives the model some memory of the conversation.

```python
parts.append(f"User: {user_text}\n\nAssistant:")
```

Add the current user message and leave the assistant answer blank.

Example:

```text
User: explain gravity simply

Assistant:
```

This blank `Assistant:` is the place where the model should continue.

```python
return "\n".join(parts)
```

Join all text pieces into one prompt string.

## Lines 20 to 24: Clean the Response

```python
def clean_response(text: str) -> str:
```

This function cleans the model output.

Small models sometimes keep generating beyond the answer.

For example:

```text
Hello! How can I help you?

User:
```

We do not want to print the fake next `User:` turn.

```python
for marker in ("User:", "<eos>", "<bos>", "<pad>"):
```

Loop over stop markers.

`User:` means the model started inventing the next user turn.

`<eos>` means end of sequence.

`<bos>` means beginning of sequence.

`<pad>` means padding.

```python
if marker in text:
    text = text.split(marker, 1)[0]
```

If a marker appears, cut the response before it.

Example:

```text
Hello! User:
```

becomes:

```text
Hello!
```

```python
return text.strip()
```

Remove extra spaces and newlines.

## Lines 27 to 35: Command-Line Options

```python
def main():
```

Main program starts here.

```python
parser = argparse.ArgumentParser(description="Chat with the SFT model.")
```

Create a command-line parser.

```python
parser.add_argument("--checkpoint", default="checkpoints/sft/best_sft_model.pt")
```

Which model checkpoint to load.

By default, it loads:

```text
checkpoints/sft/best_sft_model.pt
```

```python
parser.add_argument("--max-tokens", type=int, default=120)
```

Maximum number of new tokens to generate for each answer.

Shorter value means shorter replies.

```python
parser.add_argument("--temperature", type=float, default=0.7)
```

Temperature controls randomness.

Lower temperature:

```text
more predictable, less creative
```

Higher temperature:

```text
more random, more creative, more risk of nonsense
```

For this model, `0.7` is a reasonable starting point.

```python
parser.add_argument("--top-k", type=int, default=50)
```

Top-k sampling means the model only chooses from the 50 most likely next tokens.

This avoids very unlikely weird tokens.

```python
parser.add_argument("--top-p", type=float, default=0.9)
```

Top-p sampling keeps the smallest group of likely tokens whose total probability reaches 0.9.

It adapts based on the model's confidence.

```python
parser.add_argument("--repetition-penalty", type=float, default=1.15)
```

This discourages repeating the same tokens too much.

Small models often repeat phrases, so this helps.

```python
args = parser.parse_args()
```

Read the command-line values.

## Lines 37 to 42: Load Tokenizer and Model

```python
tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)
```

Load the same tokenizer used during training.

```python
model = GPTLanguageModel(config).to(config.device)
```

Create the model architecture and move it to GPU or CPU.

```python
checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
```

Load the saved SFT checkpoint.

```python
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
```

Handle two possible checkpoint formats.

If the file contains a full checkpoint dictionary, use `model_state_dict`.

If it is only raw weights, use it directly.

```python
model.load_state_dict(state_dict)
```

Put the saved weights into the model.

```python
model.eval()
```

Set the model to evaluation mode.

This is important because we are generating text, not training.

## Lines 44 to 47: Chat Setup

```python
eos_id = tokenizer.special_to_id.get("<eos>")
```

Get the token ID for `<eos>`, the end-of-sequence token.

If the model generates this token, generation can stop.

```python
history: list[tuple[str, str]] = []
```

Start with empty conversation history.

History will store pairs like:

```python
("hello", "Hello!")
```

```python
print("Mini GPT SFT chat. Type 'quit' to exit.")
```

Show the user that chat has started.

## Lines 48 to 51: Read User Input

```python
while True:
```

Start an infinite loop.

The chat keeps running until you quit.

```python
user_text = input("\nYou: ").strip()
```

Ask for your message.

`.strip()` removes extra spaces and newlines.

```python
if user_text.lower() in {"q", "quit", "exit"}:
    break
```

If you type `q`, `quit`, or `exit`, stop the chat loop.

## Lines 53 to 56: Convert Prompt to Tensor

```python
prompt = build_prompt(user_text, history)
```

Create the full prompt with recent history and current user message.

```python
input_ids = tokenizer.encode(prompt, add_bos=True)
```

Convert the prompt text into token IDs.

Add a beginning-of-sequence token.

```python
input_ids = input_ids[-config.block_size :]
```

Keep only the last `block_size` tokens.

Your model has a short context window, currently around 384 tokens.

If the conversation gets too long, old tokens must be removed.

This line keeps the newest part of the conversation.

```python
x = torch.tensor(input_ids, dtype=torch.long, device=config.device).unsqueeze(0)
```

Convert token IDs into a PyTorch tensor.

`unsqueeze(0)` adds the batch dimension.

The model expects shape:

```text
batch_size, sequence_length
```

For one chat prompt, batch size is 1.

## Lines 58 to 67: Generate the Answer

```python
with torch.no_grad():
```

Do not track gradients.

We are not training during chat.

```python
output = model.generate(
    x,
    max_new_tokens=args.max_tokens,
    temperature=args.temperature,
    top_k=args.top_k,
    top_p=args.top_p,
    repetition_penalty=args.repetition_penalty,
    stop_token_id=eos_id,
)
```

Ask the model to generate new tokens.

`x` is the prompt.

`max_new_tokens` limits response length.

`temperature`, `top_k`, and `top_p` control randomness.

`repetition_penalty` reduces repeated phrases.

`stop_token_id=eos_id` lets the model stop if it emits `<eos>`.

## Lines 69 to 72: Decode and Store the Response

```python
new_ids = output[0][len(input_ids) :].tolist()
```

The model output includes both:

```text
the original prompt tokens
the newly generated tokens
```

This line keeps only the new tokens.

```python
response = clean_response(tokenizer.decode(new_ids, skip_special_tokens=False))
```

Convert new token IDs back to text.

Then clean it.

```python
print(f"Assistant: {response}")
```

Show the answer in the terminal.

```python
history.append((user_text, response))
```

Save this turn to conversation history.

This lets future prompts include recent previous messages.

## Lines 74 to 75: Limit Conversation Memory

```python
# Keep short context; block_size is only 384 in the current config.
history = history[-3:]
```

Keep only the last three turns.

Why?

Because your model's context window is short.

If you keep too many turns, the prompt becomes too long and important recent text can be pushed out.

## Lines 78 to 79: Script Entry Point

```python
if __name__ == "__main__":
    main()
```

This means:

```text
If someone runs chat.py directly, start the chat program.
```

So this works:

```powershell
python chat.py
```

## Beginner Summary

`chat.py` is the conversation wrapper.

It does not make the model smart by itself.

It does three important things:

1. Loads the fine-tuned model.
2. Converts your message into the same `User:` / `Assistant:` format used during SFT.
3. Cleans and prints the generated answer.

Without this wrapper, sending just:

```text
hello
```

may make the model continue text randomly.

With this wrapper, the model sees:

```text
User: hello

Assistant:
```

That is much closer to what it was trained to answer.

