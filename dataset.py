"""
dataset.py — Memory-mapped data loading and batching for the SLM.

Uses mmap to efficiently read random chunks from the training text,
then builds (x, y) tensor pairs for the training loop.
"""

import os
import mmap
import random

import torch

from config import batch_size, block_size, device, DATA_FILE
from tokenizer import encode


_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILE)


def get_random_chunk(split: str) -> torch.Tensor:
    """
    Read a random chunk of text from the data file using mmap.

    Parameters
    ----------
    split : str
        'train' or 'val' (currently both read from the same file).

    Returns
    -------
    torch.Tensor
        1-D tensor of encoded token IDs.
    """
    with open(_data_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - block_size * batch_size)

            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            decoded_block = block.decode("utf-8", errors="ignore").replace("\r", "")
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


def get_batch(split: str):
    """
    Build a batch of (input, target) pairs from a random chunk.

    Parameters
    ----------
    split : str
        'train' or 'val'.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        x of shape (batch_size, block_size) and
        y of shape (batch_size, block_size), both on `device`.
    """
    data = get_random_chunk(split)

    if len(data) <= block_size + 1:
        raise ValueError("Random chunk too small for the requested block_size.")

    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
