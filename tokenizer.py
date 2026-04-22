"""
tokenizer.py — Character-level tokenizer for the SLM.

Reads the training text to build a vocabulary, then provides
encode (string → list[int]) and decode (list[int] → string).
"""

import os
from config import DATA_FILE


def _build_vocab(data_file: str):
    """Read the text file and return sorted unique characters."""
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    return chars


# ─── Build vocab at import time ────────────────────────
_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILE)
chars = _build_vocab(_data_path)
vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    """Convert a string to a list of integer token IDs."""
    return [string_to_int[c] for c in s]


def decode(tokens: list[int]) -> str:
    """Convert a list of integer token IDs back to a string."""
    return "".join(int_to_string[i] for i in tokens)
