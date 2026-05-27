# sft_train.py Explained

This file performs supervised fine-tuning.

In simple words, it takes your already trained base model and teaches it how to answer instructions.

Your base model already learned language from web text. But it behaves like a text continuation machine.

SFT teaches a new behavior:

```text
User asks something.
Assistant answers.
```

## Where This File Fits

```text
ckpt_step_130000.pt
  your base model

data/sft/dolly_15k.json
  instruction-answer examples

sft_dataset.py
  turns examples into training batches

sft_train.py
  loads base model
  trains on Dolly examples
  saves best_sft_model.pt

chat.py
  uses best_sft_model.pt for conversation
```

## Big Picture

The training loop repeats this:

```text
1. Get a batch of examples.
2. Ask the model to predict answer tokens.
3. Measure how wrong it was.
4. Adjust the model weights a little.
5. Occasionally check validation loss.
6. Save the best checkpoint.
```

## Lines 1 to 14: Imports

```python
"""Supervised fine-tuning entry point for the current Mini GPT checkpoint."""
```

This says the file is the main entry point for SFT.

```python
import argparse
```

This lets you pass settings from the command line.

Example:

```powershell
python sft_train.py --batch-size 4 --lr 2e-5
```

```python
import math
```

Used for `math.ceil`, which rounds up the number of steps per epoch.

```python
import os
```

Used to create folders and build checkpoint paths.

```python
import time
```

Used to measure how long each training step takes.

```python
import torch
import torch.nn.functional as F
```

These are PyTorch imports.

`torch` handles tensors, models, checkpoints, and GPU operations.

`F` contains functions such as cross-entropy loss.

```python
import config
from model import GPTLanguageModel
from sft_dataset import SFTDataset
from tokenizer import BytePairTokenizer
```

These import your project pieces.

`config` has model size, device, block size, and other settings.

`GPTLanguageModel` is your Transformer model.

`SFTDataset` creates instruction-response batches.

`BytePairTokenizer` converts text to token IDs.

## Lines 17 to 22: SFT Loss

```python
def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
```

This calculates how wrong the model is.

`logits` are the model's raw predictions.

For every token position, the model predicts a probability-like score for every possible next token.

`labels` are the correct answers.

The reshape lines flatten the data.

Originally, logits are shaped like:

```text
batch_size, sequence_length, vocab_size
```

Cross-entropy wants:

```text
all_positions, vocab_size
```

So this:

```python
logits.reshape(-1, logits.size(-1))
```

turns many sequences into one long list of token predictions.

This:

```python
labels.reshape(-1)
```

turns labels into one long list too.

The key part:

```python
ignore_index=-100
```

This tells PyTorch:

```text
If a label is -100, ignore it.
```

That is how we avoid training on user prompt tokens.

## Lines 25 to 37: Validation Loss Function

```python
@torch.no_grad()
```

This tells PyTorch not to track gradients inside this function.

Gradients are only needed when training.

During evaluation, we only want to measure loss, so this saves memory and time.

```python
def estimate_sft_loss(model: GPTLanguageModel, dataset: SFTDataset, batch_size: int, batches: int):
```

This function estimates train and validation loss.

It does not update the model.

```python
model.eval()
```

This puts the model in evaluation mode.

Dropout and other training-only behavior are disabled.

```python
out = {}
```

Create an empty dictionary to store results.

```python
for split in ("train", "val"):
```

Measure both training loss and validation loss.

Training loss tells how well the model fits examples it learns from.

Validation loss tells how well it handles held-out examples.

```python
losses = []
```

Store loss values from several batches.

```python
for _ in range(batches):
```

Evaluate multiple batches and average them.

One batch can be noisy, so averaging is more stable.

```python
xb, yb = dataset.get_batch(split, batch_size)
```

Get a batch from the dataset.

`xb` is input token IDs.

`yb` is target labels.

```python
logits, _ = model(xb)
```

Run the model on the input.

We do not pass labels to the model here because we calculate SFT loss ourselves with `ignore_index=-100`.

```python
losses.append(sft_loss(logits, yb).item())
```

Calculate the SFT loss and store it as a normal Python number.

```python
out[split] = sum(losses) / max(len(losses), 1)
```

Average all losses for this split.

```python
model.train()
return out
```

Put the model back in training mode and return the results.

## Lines 40 to 44: Loading a Checkpoint

```python
def load_checkpoint(model: GPTLanguageModel, checkpoint_path: str):
```

This function loads saved model weights.

```python
checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
```

Load a `.pt` checkpoint file.

`map_location=config.device` means:

```text
Load it onto the current device, such as GPU or CPU.
```

```python
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
```

Some checkpoints are full dictionaries. Others are just raw weights.

This line handles both forms.

If the checkpoint has `model_state_dict`, use that.

Otherwise, assume the checkpoint itself is the state dictionary.

```python
model.load_state_dict(state_dict)
```

Put the saved weights into the model.

This is what starts SFT from your 130k trained base model instead of random weights.

```python
return checkpoint
```

Return the checkpoint so we can read information like the original training step.

## Lines 47 to 57: Saving a Checkpoint

```python
def save_checkpoint(path: str, model, optimizer, step: int, epoch: int, val_loss: float):
```

This function saves the model during or after SFT.

```python
torch.save(
    {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    },
    path,
)
```

This saves:

`step`: how many SFT steps were completed.

`epoch`: which pass through the dataset we were on.

`model_state_dict`: the model weights.

`optimizer_state_dict`: optimizer state, useful if you want to resume training.

`val_loss`: validation loss at save time.

The result is a `.pt` file.

## Lines 60 to 73: Command-Line Arguments

```python
def main():
```

This is the main function.

```python
parser = argparse.ArgumentParser(description="Fine-tune the base model on Dolly SFT data.")
```

Create a command-line parser.

This lets you configure training without editing the code.

```python
parser.add_argument("--data", default="data/sft/dolly_15k.json")
```

Path to the SFT dataset.

```python
parser.add_argument("--checkpoint", default="checkpoints/ckpt_step_130000.pt")
```

Path to the base checkpoint.

This is important. SFT should start from your trained base model.

```python
parser.add_argument("--out-dir", default="checkpoints/sft")
```

Folder where SFT checkpoints are saved.

```python
parser.add_argument("--epochs", type=int, default=3)
```

How many passes through the dataset to train.

One epoch means the model has seen roughly every training example once.

```python
parser.add_argument("--batch-size", type=int, default=4)
```

How many examples are processed together.

Smaller batch size uses less VRAM.

```python
parser.add_argument("--lr", type=float, default=2e-5)
```

Learning rate.

This controls how big each weight update is.

SFT uses a small learning rate because the base model already knows language.

```python
parser.add_argument("--weight-decay", type=float, default=0.01)
```

Weight decay is a regularization technique that helps reduce overfitting.

```python
parser.add_argument("--eval-interval", type=int, default=100)
```

Evaluate every 100 steps by default.

```python
parser.add_argument("--eval-batches", type=int, default=20)
```

Use 20 batches to estimate evaluation loss.

```python
parser.add_argument("--save-interval", type=int, default=500)
```

Save an extra checkpoint every 500 steps.

```python
parser.add_argument("--max-steps", type=int, default=None)
```

Optional limit for smoke tests.

For example:

```powershell
--max-steps 20
```

trains only 20 steps.

```python
args = parser.parse_args()
```

Read the actual command-line values.

## Lines 75 to 87: Prepare Data and Model

```python
os.makedirs(args.out_dir, exist_ok=True)
print(f"Using device: {config.device}")
```

Create the output folder and print whether training uses GPU or CPU.

```python
tokenizer = BytePairTokenizer.load(config.TOKENIZER_PATH)
```

Load the tokenizer.

The same tokenizer used in pretraining must be used in SFT.

```python
dataset = SFTDataset(args.data, tokenizer, max_length=config.block_size)
```

Create the SFT dataset object.

It will read Dolly, split train/validation, tokenize examples, and return batches.

```python
print(
    f"SFT examples: {len(dataset.train_examples)} train, "
    f"{len(dataset.val_examples)} val"
)
```

Print how many training and validation examples were loaded.

```python
model = GPTLanguageModel(config).to(config.device)
```

Create the model architecture and move it to the device.

```python
base = load_checkpoint(model, args.checkpoint)
```

Load your pretrained weights.

```python
print(f"Loaded base checkpoint: {args.checkpoint} (step {base.get('step', '?')})")
```

Print which checkpoint was loaded.

## Lines 89 to 103: Optimizer and Step Counts

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)
```

Create the optimizer.

The optimizer is the part that updates model weights after each loss calculation.

AdamW is a common optimizer for Transformer training.

```python
steps_per_epoch = math.ceil(len(dataset.train_examples) / args.batch_size)
```

Calculate how many steps are needed to roughly cover the dataset once.

```python
total_steps = steps_per_epoch * args.epochs
```

Calculate total training steps.

```python
if args.max_steps is not None:
    total_steps = min(total_steps, args.max_steps)
```

If `--max-steps` is provided, cap the run.

This is useful for quick smoke tests.

```python
best_val_loss = float("inf")
step = 0
model.train()
print(f"Starting SFT for up to {total_steps} steps")
```

Initialize tracking variables and put the model in training mode.

`best_val_loss` starts as infinity so the first real validation result will be better.

## Lines 105 to 124: The Training Step

```python
for epoch in range(args.epochs):
    for _ in range(steps_per_epoch):
```

Loop through epochs and steps.

```python
if args.max_steps is not None and step >= args.max_steps:
    break
```

Stop early if this is a smoke test.

```python
t0 = time.perf_counter()
xb, yb = dataset.get_batch("train", args.batch_size)
```

Start a timer and get a training batch.

```python
with torch.autocast(
    device_type="cuda" if "cuda" in str(config.device) else "cpu",
    dtype=torch.bfloat16,
):
```

Use mixed precision.

This can make training faster and use less memory.

```python
logits, _ = model(xb)
loss = sft_loss(logits, yb)
```

Run the model and calculate SFT loss.

```python
optimizer.zero_grad(set_to_none=True)
```

Clear old gradients.

Gradients are the signals that tell the model how to change its weights.

```python
loss.backward()
```

Backpropagation.

This computes gradients.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
```

Gradient clipping.

This prevents updates from becoming too large and destabilizing training.

```python
optimizer.step()
```

Apply the update to the model weights.

This is the moment the model actually learns.

## Lines 125 to 131: Progress Logging

```python
step += 1
if step % 10 == 0:
```

Increase the step counter and print progress every 10 steps.

```python
dt = time.perf_counter() - t0
```

Measure how long the step took.

```python
print(
    f"step {step:5d}/{total_steps} | epoch {epoch + 1} | "
    f"loss {loss.item():.4f} | {dt * 1000:.0f} ms"
)
```

Print:

```text
current step
total steps
epoch
loss
milliseconds per step
```

## Lines 133 to 143: Evaluation and Best Checkpoint

```python
if step % args.eval_interval == 0 or step == total_steps:
```

Evaluate periodically.

```python
losses = estimate_sft_loss(model, dataset, args.batch_size, args.eval_batches)
```

Measure train and validation loss.

```python
print(
    f">>> eval step {step}: train {losses['train']:.4f} | "
    f"val {losses['val']:.4f}"
)
```

Print the losses.

```python
if losses["val"] < best_val_loss:
```

If validation loss improved, save the model as the best model.

```python
best_val_loss = losses["val"]
best_path = os.path.join(args.out_dir, "best_sft_model.pt")
save_checkpoint(best_path, model, optimizer, step, epoch, best_val_loss)
print(f"saved best SFT checkpoint: {best_path}")
```

This saves:

```text
checkpoints/sft/best_sft_model.pt
```

This is the checkpoint you should usually use for chat.

## Lines 145 to 148: Periodic Checkpoints

```python
if step % args.save_interval == 0:
    path = os.path.join(args.out_dir, f"sft_step_{step}.pt")
    save_checkpoint(path, model, optimizer, step, epoch, best_val_loss)
    print(f"saved checkpoint: {path}")
```

Every `save_interval` steps, save a checkpoint.

This is useful if training crashes or if you want to compare different points in training.

## Lines 150 to 159: Finish Training

```python
if step >= total_steps:
    break
```

Stop when enough steps are done.

```python
final_path = os.path.join(args.out_dir, "last_sft_model.pt")
save_checkpoint(final_path, model, optimizer, step, args.epochs, best_val_loss)
```

Save the final model.

This may not be the best model. The best model is based on validation loss.

```python
print(f"SFT complete. Last checkpoint: {final_path}")
print(f"Best val loss: {best_val_loss:.4f}")
```

Print final results.

## Lines 162 to 163: Script Entry Point

```python
if __name__ == "__main__":
    main()
```

This means:

```text
If someone runs this file directly, start training.
```

So this command works:

```powershell
python sft_train.py
```

## Beginner Summary

`sft_train.py` is the teacher.

It loads:

```text
your base model
your tokenizer
your Dolly examples
```

Then it repeatedly shows the model examples like:

```text
User: What is AI?

Assistant: AI is ...
```

and updates the model so it becomes better at producing the assistant answer.

