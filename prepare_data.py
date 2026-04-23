import glob
import os
import random

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

import config
from tokenizer import BytePairTokenizer

# --- Configuration ---
DATASET_PATH = r"D:\Openweb"
TRAIN_BIN = config.TRAIN_BIN
VAL_BIN = config.VAL_BIN
TOKENIZER_PATH = config.TOKENIZER_PATH
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


def get_parquet_files(dataset_path):
    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_path}")
    return parquet_files


def detect_text_column(parquet_path):
    schema_names = pq.ParquetFile(parquet_path).schema.names
    if "text" in schema_names:
        return "text"

    for candidate in ("content", "document", "body"):
        if candidate in schema_names:
            return candidate

    raise KeyError(
        f"Could not find a text column in {parquet_path}. "
        f"Available columns: {', '.join(schema_names)}"
    )


def iter_parquet_documents(parquet_files, text_column):
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=[text_column])
            for value in table.column(text_column).to_pylist():
                if value is None:
                    continue
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                else:
                    value = str(value)
                if value:
                    yield {"text": value}


def count_parquet_rows(parquet_files):
    total_rows = 0
    for parquet_path in parquet_files:
        total_rows += pq.ParquetFile(parquet_path).metadata.num_rows
    return total_rows

def prepare():
    # 1. Load Dataset
    print(f"Loading Parquet files from {DATASET_PATH}...")
    parquet_files = get_parquet_files(DATASET_PATH)
    text_column = detect_text_column(parquet_files[0])
    total_docs = count_parquet_rows(parquet_files)
    print(
        f"Found {len(parquet_files)} parquet shards, about {total_docs:,} rows, "
        f"using column '{text_column}'."
    )

    # 2. Train Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        training_docs = sample_text_subset(
            iter_parquet_documents(parquet_files, text_column),
            SAMPLE_SIZE_MB,
        )
        tokenizer = BytePairTokenizer()
        tokenizer.train(training_docs, vocab_size=32000)
        tokenizer.save(TOKENIZER_PATH)
        print(f"Tokenizer saved to {TOKENIZER_PATH}")
    else:
        tokenizer = BytePairTokenizer.load(TOKENIZER_PATH)
        print("Loaded existing tokenizer.")

    # 3. Process All Data (stream directly to disk)
    EOS_ID = tokenizer.special_to_id["<eos>"]
    train_token_count = 0
    val_token_count = 0

    print(f"Processing {total_docs:,} documents...")

    with open(TRAIN_BIN, "wb") as train_f, open(VAL_BIN, "wb") as val_f:
        for doc in tqdm(iter_parquet_documents(parquet_files, text_column), total=total_docs):
            tokens = tokenizer.encode(doc["text"])
            tokens.append(EOS_ID)

            arr = np.asarray(tokens, dtype=np.uint16)

            if random.random() < VAL_PERCENT:
                val_f.write(arr.tobytes())
                val_token_count += len(arr)
            else:
                train_f.write(arr.tobytes())
                train_token_count += len(arr)

    print(f"Complete! Train: {train_token_count:,} tokens | Val: {val_token_count:,} tokens")
if __name__ == "__main__":
    prepare()
