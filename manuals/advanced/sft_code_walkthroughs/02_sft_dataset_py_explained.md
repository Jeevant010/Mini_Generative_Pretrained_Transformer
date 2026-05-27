# sft_dataset.py Explained

This file is the bridge between human-readable training examples and model-ready tensors.

Your Dolly file contains text:

```json
{
  "instruction": "Explain gravity simply.",
  "input": "",
  "output": "Gravity is the force that pulls objects with mass toward each other."
}
```

The model cannot directly train on JSON or English strings. It needs numbers.

So `sft_dataset.py` turns examples into:

```text
User: Explain gravity simply.

Assistant: Gravity is the force that pulls objects with mass toward each other.
```

Then it tokenizes that text into numbers and creates training labels.

The most important idea in this file is **loss masking**.

Loss masking means:

```text
Do not train the model to predict the user prompt.
Only train it to predict the assistant answer.
```

## Where This File Fits

```text
data/sft/dolly_15k.json
  raw examples

sft_dataset.py
  formats examples
  tokenizes them
  masks user tokens
  creates batches

sft_train.py
  asks this file for batches
  trains the model
```

## Lines 1 to 10: Imports and Setup

```python
"""Instruction-response dataset utilities for supervised fine-tuning."""
```

This says the file contains utilities for supervised fine-tuning.

Supervised fine-tuning means:

```text
We show the model examples of good user requests and good assistant answers.
```

```python
import json
```

This reads the `dolly_15k.json` file.

```python
import random
```

This shuffles examples and samples random examples for each batch.

```python
from typing import Dict, List, Tuple
```

These are type hints.

They do not change how the program runs, but they explain what data types are expected.

For example:

```python
Dict[str, str]
```

means:

```text
A dictionary where keys are strings and values are strings.
```

```python
import torch
```

This imports PyTorch, the deep learning library used to create tensors and train the model.

```python
import config
```

This imports your project settings, such as:

```text
block_size
device
```

`block_size` is the maximum number of tokens the model can see at once.

`device` is usually `cuda` if your GPU is available, otherwise `cpu`.

```python
from tokenizer import BytePairTokenizer
```

This imports your tokenizer class.

The tokenizer converts text into token IDs:

```text
"hello" -> [153, 421]
```

The model only understands token IDs, not raw text.

## Lines 13 to 14: The Dataset Class

```python
class SFTDataset:
    """Loads Dolly-style records and returns causal-LM batches with label masks."""
```

This creates a class named `SFTDataset`.

A class is like a reusable machine. Once created, this machine can:

1. Load Dolly examples.
2. Split them into train and validation.
3. Format each example as `User:` and `Assistant:`.
4. Convert text to token IDs.
5. Return batches for training.

The phrase `causal-LM` means causal language model.

A causal language model predicts the next token from previous tokens.

Example:

```text
The sky is
```

The model predicts:

```text
blue
```

## Lines 16 to 23: Constructor Arguments

```python
def __init__(
    self,
    data_path: str,
    tokenizer: BytePairTokenizer,
    max_length: int | None = None,
    val_fraction: float = 0.05,
    seed: int = 1337,
):
```

This method runs when you create the dataset:

```python
dataset = SFTDataset("data/sft/dolly_15k.json", tokenizer)
```

`data_path` is the path to the JSON file.

`tokenizer` is the tokenizer object.

`max_length` is the longest allowed token sequence.

If `max_length` is not given, the code uses `config.block_size`.

`val_fraction=0.05` means 5 percent of examples become validation examples.

Validation examples are not used to update the model. They are used to check whether the model is learning in a general way.

`seed=1337` makes the shuffle reproducible.

If you run the script again with the same seed, you get the same train/validation split.

## Lines 24 to 25: Store Important Settings

```python
self.tokenizer = tokenizer
self.max_length = max_length or config.block_size
```

The dataset object remembers the tokenizer.

It also sets the maximum sequence length.

This expression:

```python
max_length or config.block_size
```

means:

```text
Use max_length if it was provided.
Otherwise use config.block_size.
```

In your current config, `block_size` is 384, so examples are limited to around 384 tokens.

## Lines 27 to 28: Load the JSON File

```python
with open(data_path, "r", encoding="utf-8") as f:
    examples = json.load(f)
```

This opens `data/sft/dolly_15k.json` and loads all examples into memory.

After this, `examples` is a Python list:

```python
[
    {"instruction": "...", "input": "...", "output": "..."},
    {"instruction": "...", "input": "...", "output": "..."},
    ...
]
```

## Lines 30 to 31: Shuffle Examples

```python
rng = random.Random(seed)
rng.shuffle(examples)
```

This creates a random number generator using the seed.

Then it shuffles the examples.

Why shuffle?

Because the dataset may have examples grouped by type. If we did not shuffle, validation data might accidentally contain too many examples from one category.

Shuffling makes the split fairer.

## Lines 33 to 35: Train/Validation Split

```python
split_idx = int(len(examples) * (1.0 - val_fraction))
self.train_examples = examples[:split_idx]
self.val_examples = examples[split_idx:]
```

This calculates where to split the list.

If there are 15,000 examples and `val_fraction` is 0.05:

```text
95 percent train
5 percent validation
```

`self.train_examples` is what the model learns from.

`self.val_examples` is what we use to check progress.

## Lines 37 to 38: Dataset Length

```python
def __len__(self) -> int:
    return len(self.train_examples) + len(self.val_examples)
```

This lets Python understand:

```python
len(dataset)
```

It returns the total number of examples.

## Lines 40 to 52: Formatting One Example

```python
@staticmethod
def format_example(example: Dict[str, str]) -> Tuple[str, str]:
```

This function converts one JSON example into two pieces:

```text
prompt
response
```

We keep them separate because the prompt should be masked and the response should be trained.

```python
instruction = example.get("instruction", "").strip()
context = example.get("input", "").strip()
output = example.get("output", "").strip()
```

These lines take the useful text out of the example.

`instruction` is the user's request.

`context` is optional background text.

`output` is the answer we want the assistant to learn.

`.strip()` removes extra spaces and newlines from the beginning and end.

### If There Is Context

```python
if context:
    prompt = f"User: {instruction}\n\nContext:\n{context}\n\nAssistant:"
```

If the example has extra information, the prompt includes it.

Example:

```text
User: When did Virgin Australia start operating?

Context:
Virgin Australia ... commenced services on 31 August 2000 ...

Assistant:
```

This teaches the model to use the context before answering.

### If There Is No Context

```python
else:
    prompt = f"User: {instruction}\n\nAssistant:"
```

If there is no context, the prompt is simpler:

```text
User: Explain gravity simply.

Assistant:
```

### The Response

```python
response = f" {output}"
return prompt, response
```

The response is the desired assistant answer.

There is a leading space before `{output}` because the prompt ends with:

```text
Assistant:
```

The natural continuation is:

```text
Assistant: Gravity is...
```

## Lines 54 to 86: Encoding One Example

```python
def encode_example(self, example: Dict[str, str]) -> Tuple[List[int], List[int]]:
```

This converts one formatted example into:

```text
x = input token IDs
y = target labels
```

The model sees `x`.

The loss compares the model's predictions against `y`.

```python
prompt, response = self.format_example(example)
```

First, convert the JSON row into text.

```python
prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
response_ids = self.tokenizer.encode(response, add_eos=True)
```

These lines tokenize the prompt and response.

`add_bos=True` adds a beginning token to the prompt.

`add_eos=True` adds an end token to the response.

`bos` means beginning of sequence.

`eos` means end of sequence.

This teaches the model where text starts and where the assistant answer should stop.

## Lines 60 to 73: Smart Truncation

```python
# Preserve answer tokens when examples are too long. Dolly contains
# long context passages, and naive left-to-right truncation can remove
# the assistant response entirely, leaving no useful SFT loss.
```

This comment explains a bug we fixed.

Some Dolly examples have very long context passages. Since your model can only see about 384 tokens, long examples must be shortened.

A naive cut would do this:

```text
Keep the first 384 tokens and throw away the rest.
```

But if the answer comes after a long context, this can remove the answer completely.

That is bad because SFT needs answer tokens to learn from.

```python
if len(prompt_ids) + len(response_ids) > self.max_length:
```

This checks whether prompt plus response is too long.

```python
max_response_len = max(self.max_length - 1, 1)
```

This calculates the maximum space allowed for response tokens.

The `max(..., 1)` part makes sure the response gets at least one token.

```python
if len(response_ids) > max_response_len:
    response_ids = response_ids[:max_response_len]
```

If the answer itself is too long, cut the answer down.

We keep the beginning of the answer.

```python
max_prompt_len = max(self.max_length - len(response_ids), 1)
```

After reserving room for the answer, this calculates how much room is left for the prompt.

```python
if len(prompt_ids) > max_prompt_len:
```

If the prompt is still too long, shorten the prompt.

```python
if max_prompt_len == 1:
    prompt_ids = prompt_ids[-1:]
```

If there is only room for one prompt token, keep the last prompt token.

```python
else:
    prompt_ids = [prompt_ids[0]] + prompt_ids[-(max_prompt_len - 1):]
```

If there is room for more, keep:

1. The first token, usually the beginning token.
2. The end of the prompt.

Why keep the end of the prompt?

Because the end usually contains:

```text
Assistant:
```

The model needs that marker to know it should answer.

## Lines 75 to 84: Build Inputs and Masked Labels

```python
ids = prompt_ids + response_ids
```

Combine prompt and response into one sequence.

Example:

```text
<bos>User: Explain gravity.

Assistant: Gravity is ...<eos>
```

```python
x = ids[:-1]
```

The model input is every token except the last one.

Language models train by predicting the next token.

If:

```python
ids = [10, 20, 30, 40]
```

then:

```python
x = [10, 20, 30]
target = [20, 30, 40]
```

```python
prompt_target_mask_len = max(len(prompt_ids) - 1, 0)
```

This calculates how many prompt target positions should be ignored.

We do not want the model to learn to generate user prompts.

We want it to learn assistant answers.

```python
y = [-100] * prompt_target_mask_len + ids[len(prompt_ids) :]
```

This is the most important line.

`-100` means:

```text
Ignore this position in the loss.
```

So the labels look like:

```text
ignored, ignored, ignored, answer_token_1, answer_token_2, eos
```

The model is trained only on answer tokens.

```python
y = y[: len(x)]
```

This makes sure labels have the same length as inputs.

```python
return x, y
```

Return the input IDs and target labels.

## Lines 88 to 110: Creating a Batch

```python
def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
```

This function returns a training batch.

A batch is a group of examples processed together.

```python
if split not in {"train", "val"}:
    raise ValueError("split must be 'train' or 'val'")
```

This checks that the caller asks for either training data or validation data.

```python
examples = self.train_examples if split == "train" else self.val_examples
```

Choose the correct list.

```python
if not examples:
    raise ValueError(f"No examples available for split: {split}")
```

If the chosen list is empty, stop with a clear error.

```python
selected = random.choices(examples, k=batch_size)
```

Randomly pick examples for the batch.

```python
encoded = [self.encode_example(example) for example in selected]
```

Tokenize and label each selected example.

```python
max_len = max(len(x) for x, _ in encoded)
```

Find the longest example in this batch.

Neural networks want rectangular tensors. That means every example in a batch must have the same length.

```python
pad_id = self.tokenizer.special_to_id.get("<pad>", 0)
```

Find the token ID for padding.

Padding is filler added to short examples.

```python
xs, ys = [], []
for x, y in encoded:
    pad_len = max_len - len(x)
    xs.append(x + [pad_id] * pad_len)
    ys.append(y + [-100] * pad_len)
```

This pads every example to the same length.

Inputs get the pad token.

Labels get `-100` so padding does not affect the loss.

```python
xb = torch.tensor(xs, dtype=torch.long, device=config.device)
yb = torch.tensor(ys, dtype=torch.long, device=config.device)
return xb, yb
```

Convert lists into PyTorch tensors and move them to the right device.

`xb` is the batch input.

`yb` is the batch label.

These are returned to `sft_train.py`.

## The Core Idea

The model sees:

```text
User: Explain gravity simply.

Assistant:
```

The model is trained to produce:

```text
 Gravity is the force that pulls objects with mass toward each other.
```

It is not trained to reproduce:

```text
User: Explain gravity simply.
```

That is why loss masking matters.

## Beginner Summary

`sft_dataset.py` is like a translator.

It translates human teaching examples into the numerical format the model can learn from.

It also makes sure the model learns the assistant answer, not the user question.

