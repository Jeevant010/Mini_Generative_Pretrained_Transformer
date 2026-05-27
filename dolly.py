from datasets import load_dataset
import json, os

ds = load_dataset("databricks/databricks-dolly-15k", split="train")

rows = []
for item in ds:
    rows.append({
        "instruction": item["instruction"],
        "input": item.get("context", ""),
        "output": item["response"],
        "category": item.get("category", "")
    })

os.makedirs("data/sft", exist_ok=True)

with open("data/sft/dolly_15k.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)