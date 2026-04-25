"""
tools/slice_data.py — Create a subset of your tokenized dataset.

Takes the existing train.bin / val.bin and creates smaller versions
for faster training iteration. Randomly samples from the full binary
to maintain data diversity.

Usage:
    python -m tools.slice_data --size-gb 10
    python -m tools.slice_data --size-gb 3 --output-dir subsets/3gb
    python -m tools.slice_data --size-gb 1 --source-train train.bin --source-val val.bin
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def slice_binary(source_path, target_path, target_tokens, block_size=384):
    """
    Creates a random subset of a token binary file.
    Samples contiguous chunks of `block_size * 8` tokens to maintain
    document coherence while ensuring diversity.
    """
    source_tokens = os.path.getsize(source_path) // 2  # uint16

    if target_tokens >= source_tokens:
        print(f"  ⚠ Requested {target_tokens:,} tokens but source only has {source_tokens:,}. Copying full file.")
        import shutil
        shutil.copy2(source_path, target_path)
        return source_tokens

    # Memory-map the source
    data = np.memmap(source_path, dtype=np.uint16, mode="r")

    # Sample contiguous chunks for better document coherence
    chunk_size = block_size * 8  # ~3072 tokens per chunk
    num_chunks = target_tokens // chunk_size

    # Generate random start positions
    max_start = len(data) - chunk_size
    starts = np.random.randint(0, max_start, size=num_chunks)
    starts.sort()  # Sort for sequential disk access (faster on HDD)

    # Collect chunks
    chunks = []
    for s in starts:
        chunks.append(np.array(data[s : s + chunk_size], dtype=np.uint16))

    result = np.concatenate(chunks)
    result.tofile(target_path)

    return len(result)


def main():
    parser = argparse.ArgumentParser(description="Create a data subset for training.")
    parser.add_argument("--size-gb", type=float, required=True, help="Target size in GB.")
    parser.add_argument("--source-train", type=str, default="train.bin", help="Source train binary.")
    parser.add_argument("--source-val", type=str, default="val.bin", help="Source val binary.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: subsets/<size>gb).")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio of subset.")
    args = parser.parse_args()

    # Validate source files
    for path in [args.source_train, args.source_val]:
        if not os.path.exists(path):
            print(f"❌ Source file not found: {path}")
            sys.exit(1)

    # Calculate token counts
    target_bytes = int(args.size_gb * 1024 * 1024 * 1024)
    target_tokens = target_bytes // 2  # uint16
    train_tokens = int(target_tokens * (1 - args.val_ratio))
    val_tokens = int(target_tokens * args.val_ratio)

    # Output directory
    if args.output_dir is None:
        size_label = f"{args.size_gb:.0f}gb" if args.size_gb >= 1 else f"{int(args.size_gb * 1024)}mb"
        args.output_dir = os.path.join("subsets", size_label)
    os.makedirs(args.output_dir, exist_ok=True)

    train_out = os.path.join(args.output_dir, "train.bin")
    val_out = os.path.join(args.output_dir, "val.bin")

    # Source info
    src_train_tokens = os.path.getsize(args.source_train) // 2
    src_val_tokens = os.path.getsize(args.source_val) // 2

    print("=" * 60)
    print("🔪 DATA SUBSET CREATOR")
    print("=" * 60)
    print(f"Source train : {args.source_train} ({src_train_tokens:,} tokens, {os.path.getsize(args.source_train) / 1e9:.2f} GB)")
    print(f"Source val   : {args.source_val} ({src_val_tokens:,} tokens, {os.path.getsize(args.source_val) / 1e9:.2f} GB)")
    print(f"Target size  : {args.size_gb:.1f} GB ({target_tokens:,} tokens)")
    print(f"Output dir   : {args.output_dir}")
    print("-" * 60)

    # Slice train
    print(f"\n  Slicing train.bin ({train_tokens:,} tokens)...")
    actual_train = slice_binary(args.source_train, train_out, train_tokens)

    # Slice val
    print(f"  Slicing val.bin ({val_tokens:,} tokens)...")
    actual_val = slice_binary(args.source_val, val_out, val_tokens)

    # Estimate training time
    est_tok_per_sec = 10000  # RTX 4060 estimate
    est_tokens_per_step = 20 * 384  # batch_size * block_size
    est_steps = actual_train // est_tokens_per_step
    est_hours = (est_steps * est_tokens_per_step) / est_tok_per_sec / 3600

    print(f"\n{'='*60}")
    print(f"✅ SUBSET CREATED")
    print(f"{'='*60}")
    print(f"Train : {train_out} ({actual_train:,} tokens, {actual_train * 2 / 1e9:.2f} GB)")
    print(f"Val   : {val_out} ({actual_val:,} tokens, {actual_val * 2 / 1e9:.2f} GB)")
    print(f"\n📈 Estimated Training:")
    print(f"   Steps to 1 epoch: ~{est_steps:,}")
    print(f"   Time (@ ~{est_tok_per_sec:,} tok/s): ~{est_hours:.1f} hours ({est_hours/24:.1f} days)")
    print(f"\n💡 Update config.py to use this subset:")
    print(f'   TRAIN_BIN = "{train_out}"')
    print(f'   VAL_BIN   = "{val_out}"')
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
