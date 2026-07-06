"""Dataset loader for instruction-response SFT training."""

import json
import random
import torch
import os
import sys

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizer import BytePairTokenizer
import config

class SFTDataset:
    """
    Converts instruction-response pairs into token sequences with loss masks.
    """

    TEMPLATE_WITH_INPUT = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )

    TEMPLATE_NO_INPUT = (
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}"
    )

    def __init__(self, data_path: str, tokenizer: BytePairTokenizer,
                 max_length: int = None, val_fraction: float = 0.05):
        self.tokenizer = tokenizer
        self.max_length = max_length or config.block_size

        with open(data_path, "r", encoding="utf-8") as f:
            all_examples = json.load(f)

        # Shuffle and split
        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * (1 - val_fraction))
        self.train_examples = all_examples[:split_idx]
        self.val_examples = all_examples[split_idx:]

        print(f"SFT dataset: {len(self.train_examples)} train, "
              f"{len(self.val_examples)} val examples")

    def _format_example(self, example: dict) -> tuple:
        """Format and tokenize a single example. Returns (input_ids, label_ids)."""
        if example.get("input", "").strip():
            text = self.TEMPLATE_WITH_INPUT.format(**example)
        else:
            text = self.TEMPLATE_NO_INPUT.format(**example)

        # Find where the response starts
        response_marker = "### Response:\n"
        response_start = text.find(response_marker)
        instruction_text = text[:response_start + len(response_marker)]
        response_text = text[response_start + len(response_marker):]

        instruction_ids = self.tokenizer.encode(instruction_text, add_bos=True)
        response_ids = self.tokenizer.encode(response_text, add_eos=True)

        input_ids = instruction_ids + response_ids

        # Create labels: -100 for instruction tokens (ignored in loss), 
        # actual token IDs for response tokens
        labels = [-100] * len(instruction_ids) + response_ids

        # Truncate to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return input_ids, labels

    def get_batch(self, split: str = "train", batch_size: int = None):
        """Sample a batch of (input_ids, labels) pairs."""
        if batch_size is None:
            batch_size = config.batch_size

        examples = self.train_examples if split == "train" else self.val_examples
        batch_indices = random.sample(range(len(examples)), min(batch_size, len(examples)))

        batch_inputs = []
        batch_labels = []
        max_len = 0

        for idx in batch_indices:
            input_ids, labels = self._format_example(examples[idx])
            batch_inputs.append(input_ids)
            batch_labels.append(labels)
            max_len = max(max_len, len(input_ids))

        # Pad to max length in batch
        pad_id = self.tokenizer.special_to_id.get("<pad>", 0)
        for i in range(len(batch_inputs)):
            pad_len = max_len - len(batch_inputs[i])
            batch_inputs[i] = batch_inputs[i] + [pad_id] * pad_len
            batch_labels[i] = batch_labels[i] + [-100] * pad_len

        x = torch.tensor(batch_inputs, dtype=torch.long).to(config.device)
        y = torch.tensor(batch_labels, dtype=torch.long).to(config.device)

        return x, y
