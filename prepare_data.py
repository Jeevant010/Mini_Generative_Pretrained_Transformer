import os
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset
from tokenizer import BytePairTokenizer

# --- Configuration ---
DATASET_PATH = r"D:\Dataset"
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"
TOKENIZER_PATH = "bpe_tokenizer_32k.json"
VAL_PERCENT = 0.05
SAMPLE_SIZE_MB = 100

def sample_text_subset(dataset, target_size_mb):
    current_size = 0
    sampled_text = []
    print(f"Sampling {target_size_mb}MB for tokenizer training...")
    for doc in dataset:
        text = doc["text"]
        sampled_text.append(text)
        current_size += len(text.encode("utf-8"))
        if current_size >= target_size_mb * 1024 * 1024:
            break
    return sampled_text # Return list, not joined string

def prepare():
    # 1. Load Dataset
    print(f"Loading Parquet files from {DATASET_PATH}...")
    ds = load_dataset("parquet", data_files=[os.path.join(DATASET_PATH, "*.parquet")], split="train")

    # 2. Train Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        training_docs = sample_text_subset(ds, SAMPLE_SIZE_MB)
        tokenizer = BytePairTokenizer()
        tokenizer.train(training_docs, vocab_size=32000)
        tokenizer.save(TOKENIZER_PATH)
        print(f"Tokenizer saved to {TOKENIZER_PATH}")
    else:
        tokenizer = BytePairTokenizer.load(TOKENIZER_PATH)
        print("Loaded existing tokenizer.")

    # 3. Process All Data
    train_ids, val_ids = [], []
    EOS_ID = tokenizer.special_to_id["<eos>"]
    
    print(f"Processing {len(ds)} documents...")
    for doc in tqdm(ds):
        tokens = tokenizer.encode(doc["text"])
        tokens.append(EOS_ID)
        if random.random() < VAL_PERCENT:
            val_ids.extend(tokens)
        else:
            train_ids.extend(tokens)

    # 4. Save Binary Files
    print(f"Saving binary files (uint16)...")
    np.array(train_ids, dtype=np.uint16).tofile(TRAIN_BIN)
    np.array(val_ids, dtype=np.uint16).tofile(VAL_BIN)
    print(f"Complete! Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")

if __name__ == "__main__":
    prepare()
