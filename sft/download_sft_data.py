"""Download Dolly 15K dataset for SFT."""

import json
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets library first: pip install datasets")
    sys.exit(1)

def download_dolly():
    """Download and save Dolly 15K in Alpaca format."""
    print("Downloading databricks-dolly-15k from HuggingFace...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    examples = []
    for item in ds:
        examples.append({
            "instruction": item["instruction"],
            "input": item.get("context", ""),
            "output": item["response"],
            "category": item.get("category", ""),
        })

    # Ensure data directory exists in the root of the project
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    
    out_path = os.path.join(data_dir, "dolly_15k.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(examples)} examples to {out_path}")

if __name__ == "__main__":
    download_dolly()
