import os
import numpy as np
import torch
from config import batch_size, block_size, device

# Production paths
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"

def get_batch(split: str):
    """
    Production-ready batching using memory-mapped binary files.
    """
    filename = TRAIN_BIN if split == "train" else VAL_BIN
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Binary file {filename} not found. Run prepare_data.py first.")
    
    # mmap in read-only mode to save RAM
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # Select random offsets
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Read slices into torch tensors (converting uint16 -> int64)
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
