# dolly.py Explained

This file is the data download step. It takes the Dolly 15K instruction dataset from Hugging Face and saves it into your project as a JSON file.

In simple words:

1. Go to the internet dataset library.
2. Download the Dolly 15K training examples.
3. Convert each example into the format our project expects.
4. Save everything into `data/sft/dolly_15k.json`.

This file does not train the model. It only prepares the teaching material.

## Where This File Fits

The fine-tuning flow is:

```text
dolly.py
  downloads Dolly 15K
  saves data/sft/dolly_15k.json

sft_dataset.py
  reads data/sft/dolly_15k.json
  turns examples into User/Assistant token batches

sft_train.py
  trains the model on those batches

chat.py
  lets you talk to the fine-tuned model
```

So `dolly.py` is the first small door in the whole SFT pipeline.

## The Whole File

```python
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
```

## Line 1

```python
from datasets import load_dataset
```

This imports `load_dataset` from the Hugging Face `datasets` library.

Think of Hugging Face as a public library for AI datasets. `load_dataset` is the function that goes to that library and brings back a dataset.

Here, we use it to download:

```text
databricks/databricks-dolly-15k
```

That dataset contains instruction-response examples. Each example is like a tiny lesson:

```text
Instruction: Explain gravity simply.
Response: Gravity is the force that pulls objects with mass toward each other.
```

## Line 2

```python
import json, os
```

This imports two standard Python tools.

`json` is used for reading and writing JSON files.

JSON is a common format for structured data. Your `dolly_15k.json` file is JSON.

`os` is used for operating-system tasks, such as creating folders.

In this file, `os` is used to create:

```text
data/sft
```

if it does not already exist.

## Line 4

```python
ds = load_dataset("databricks/databricks-dolly-15k", split="train")
```

This downloads and loads the Dolly 15K training split.

`"databricks/databricks-dolly-15k"` is the dataset name on Hugging Face.

`split="train"` means:

```text
Give me the training part of the dataset.
```

The result is stored in `ds`.

After this line, `ds` behaves like a list of examples. You can loop through it one item at a time.

One raw Dolly item has fields similar to:

```json
{
  "instruction": "When did Virgin Australia start operating?",
  "context": "Virgin Australia ... commenced services on 31 August 2000 ...",
  "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue.",
  "category": "closed_qa"
}
```

The original field names are not exactly the names we want in our project, so the next lines convert them.

## Line 6

```python
rows = []
```

This creates an empty list.

This list will hold the cleaned examples that we want to save.

At the beginning:

```python
rows = []
```

After processing examples:

```python
rows = [
    {
        "instruction": "...",
        "input": "...",
        "output": "...",
        "category": "..."
    },
    ...
]
```

## Line 7

```python
for item in ds:
```

This starts a loop.

It means:

```text
For every example in the Dolly dataset, do the following work.
```

Each single example is temporarily called `item`.

If Dolly has about 15,000 examples, this loop runs about 15,000 times.

## Lines 8 to 13

```python
    rows.append({
        "instruction": item["instruction"],
        "input": item.get("context", ""),
        "output": item["response"],
        "category": item.get("category", "")
    })
```

This creates a new dictionary for each example and adds it to `rows`.

A dictionary is a set of named fields.

The code converts Dolly's original field names into our project's field names.

### The New `instruction` Field

```python
"instruction": item["instruction"],
```

This copies the instruction from Dolly.

Example:

```text
When did Virgin Australia start operating?
```

This is what the user asks.

### The New `input` Field

```python
"input": item.get("context", ""),
```

Dolly calls extra background text `context`.

Our project calls it `input`.

So this line means:

```text
Take Dolly's context field and store it as input.
```

The `.get("context", "")` part is a safe way to read the field.

It means:

```text
If item has a context field, use it.
If it does not, use an empty string.
```

This avoids crashes if an example has no context.

### The New `output` Field

```python
"output": item["response"],
```

Dolly calls the answer `response`.

Our project calls it `output`.

So this line copies the answer.

Example:

```text
Virgin Australia commenced services on 31 August 2000 as Virgin Blue.
```

This is what the assistant should learn to produce.

### The New `category` Field

```python
"category": item.get("category", "")
```

Dolly examples have categories such as:

```text
open_qa
closed_qa
classification
summarization
brainstorming
```

The category is not directly used during training right now, but it is useful for analysis.

For example, later you can ask:

```text
Is the model better at classification or summarization?
```

## Line 15

```python
os.makedirs("data/sft", exist_ok=True)
```

This creates the folder:

```text
data/sft
```

`exist_ok=True` means:

```text
If the folder already exists, do not crash.
```

This is useful because you may run `dolly.py` more than once.

Without `exist_ok=True`, Python would complain if the folder already existed.

## Line 17

```python
with open("data/sft/dolly_15k.json", "w", encoding="utf-8") as f:
```

This opens a file for writing.

The path is:

```text
data/sft/dolly_15k.json
```

The `"w"` means write mode.

If the file does not exist, Python creates it.

If the file already exists, Python replaces it.

`encoding="utf-8"` means the file can safely store many kinds of characters, including punctuation, symbols, and non-English text.

The `with open(...) as f:` style automatically closes the file when writing is done.

## Line 18

```python
    json.dump(rows, f, ensure_ascii=False, indent=2)
```

This writes the `rows` list into the JSON file.

`rows` is the cleaned list of Dolly examples.

`f` is the open file.

`ensure_ascii=False` means:

```text
Keep real characters instead of forcing everything into escaped ASCII codes.
```

`indent=2` means:

```text
Make the JSON file pretty and readable with two spaces of indentation.
```

Without `indent=2`, the JSON would be one huge line.

## What the Output File Looks Like

After running `dolly.py`, you get:

```text
data/sft/dolly_15k.json
```

Inside it, each example looks like:

```json
{
  "instruction": "Why can camels survive for long without water?",
  "input": "",
  "output": "Camels use the fat in their humps to keep them filled with energy and hydration for long periods of time.",
  "category": "open_qa"
}
```

This is now ready for `sft_dataset.py`.

## Why We Do This Conversion

The model training code should not need to know Hugging Face's exact field names.

By converting the dataset into our own simple format, the rest of the project can expect every example to have:

```text
instruction
input
output
category
```

This makes the next files simpler.

## Beginner Summary

`dolly.py` is like a teacher preparing a workbook.

It does not teach the model directly. It only downloads the lessons and writes them neatly into a local file.

The lessons are later used by `sft_dataset.py` and `sft_train.py`.

