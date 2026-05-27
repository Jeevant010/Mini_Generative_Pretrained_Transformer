"""Instruction-response dataset utilities for supervised fine-tuning."""

import json
import random
from typing import Dict, List, Tuple

import torch

import config
from tokenizer import BytePairTokenizer


class SFTDataset:
    """Loads Dolly-style records and returns causal-LM batches with label masks."""

    def __init__(
        self,
        data_path: str,
        tokenizer: BytePairTokenizer,
        max_length: int | None = None,
        val_fraction: float = 0.05,
        seed: int = 1337,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length or config.block_size

        with open(data_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        rng = random.Random(seed)
        rng.shuffle(examples)

        split_idx = int(len(examples) * (1.0 - val_fraction))
        self.train_examples = examples[:split_idx]
        self.val_examples = examples[split_idx:]

    def __len__(self) -> int:
        return len(self.train_examples) + len(self.val_examples)

    @staticmethod
    def format_example(example: Dict[str, str]) -> Tuple[str, str]:
        instruction = example.get("instruction", "").strip()
        context = example.get("input", "").strip()
        output = example.get("output", "").strip()

        if context:
            prompt = f"User: {instruction}\n\nContext:\n{context}\n\nAssistant:"
        else:
            prompt = f"User: {instruction}\n\nAssistant:"

        response = f" {output}"
        return prompt, response

    def encode_example(self, example: Dict[str, str]) -> Tuple[List[int], List[int]]:
        prompt, response = self.format_example(example)

        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        response_ids = self.tokenizer.encode(response, add_eos=True)

        # Preserve answer tokens when examples are too long. Dolly contains
        # long context passages, and naive left-to-right truncation can remove
        # the assistant response entirely, leaving no useful SFT loss.
        if len(prompt_ids) + len(response_ids) > self.max_length:
            max_response_len = max(self.max_length - 1, 1)
            if len(response_ids) > max_response_len:
                response_ids = response_ids[:max_response_len]

            max_prompt_len = max(self.max_length - len(response_ids), 1)
            if len(prompt_ids) > max_prompt_len:
                if max_prompt_len == 1:
                    prompt_ids = prompt_ids[-1:]
                else:
                    prompt_ids = [prompt_ids[0]] + prompt_ids[-(max_prompt_len - 1):]

        ids = prompt_ids + response_ids

        x = ids[:-1]

        # For causal LM training, target position i is ids[i + 1].
        # Mask all prompt targets except the final prompt token, which should
        # learn to predict the first assistant response token.
        prompt_target_mask_len = max(len(prompt_ids) - 1, 0)
        y = [-100] * prompt_target_mask_len + ids[len(prompt_ids) :]
        y = y[: len(x)]

        return x, y

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        examples = self.train_examples if split == "train" else self.val_examples
        if not examples:
            raise ValueError(f"No examples available for split: {split}")

        selected = random.choices(examples, k=batch_size)
        encoded = [self.encode_example(example) for example in selected]

        max_len = max(len(x) for x, _ in encoded)
        pad_id = self.tokenizer.special_to_id.get("<pad>", 0)

        xs, ys = [], []
        for x, y in encoded:
            pad_len = max_len - len(x)
            xs.append(x + [pad_id] * pad_len)
            ys.append(y + [-100] * pad_len)

        xb = torch.tensor(xs, dtype=torch.long, device=config.device)
        yb = torch.tensor(ys, dtype=torch.long, device=config.device)
        return xb, yb
