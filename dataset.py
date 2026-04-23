import os
import numpy as np
import torch
import config

_train_data = None
_val_data = None

def _get_data(split: str):
    global _train_data, _val_data

    if split == "train":
        if _train_data is None:
            if not os.path.exists(config.TRAIN_BIN):
                raise FileNotFoundError(f"Binary file {config.TRAIN_BIN} not found. Run prepare_data.py first.")
            _train_data = np.memmap(config.TRAIN_BIN, dtype=np.uint16, mode="r")
        return _train_data

    if split == "val":
        if _val_data is None:
            if not os.path.exists(config.VAL_BIN):
                raise FileNotFoundError(f"Binary file {config.VAL_BIN} not found. Run prepare_data.py first.")
            _val_data = np.memmap(config.VAL_BIN, dtype=np.uint16, mode="r")
        return _val_data

    raise ValueError(f"Unknown split: {split}")

def get_batch(split: str):
    data = _get_data(split)
    max_start = len(data) - config.block_size - 1
    if max_start <= 0:
        raise ValueError(f"{split} split is too small for block_size={config.block_size}.")

    starts = np.random.randint(0, max_start + 1, size=config.batch_size, dtype=np.int64)
    offsets = starts[:, None] + np.arange(config.block_size, dtype=np.int64)

    x = torch.from_numpy(np.asarray(data[offsets], dtype=np.int64))
    y = torch.from_numpy(np.asarray(data[offsets + 1], dtype=np.int64))

    if "cuda" in str(config.device):
        x = x.pin_memory()
        y = y.pin_memory()

    return x.to(config.device, non_blocking=True), y.to(config.device, non_blocking=True)
