# Mini Generative Pretrained Transformer

A compact GPT-style language model training project built with PyTorch. The repo uses a modern decoder-only architecture with:

- Grouped-Query Attention (GQA)
- RoPE positional encoding
- RMSNorm
- SwiGLU feed-forward blocks
- tied token embedding / LM head weights

It is designed to train from local parquet shards, write tokenized binaries to disk, and train from memory-mapped data without loading the full corpus into RAM.

## Repo Layout

```text
Mini_Generative_Pretrained_Transformer/
|-- config.py
|-- prepare_data.py
|-- dataset.py
|-- model.py
|-- training.py
|-- generate.py
|-- tokenizer.py
|-- checkpoints/
`-- Research/
```

## Current Defaults

The current default config in [config.py](config.py) is:

- `batch_size = 20`
- `block_size = 384`
- `max_iters = 300000`
- `learning_rate = 2.5e-4`
- `min_lr = 2.5e-5`
- `warmup_iters = 2000`
- `lr_decay_iters = max_iters`
- `grad_clip = 1.0`
- `eval_iters = 25`
- `eval_interval = 2000`
- `checkpoint_interval = 5000`
- `vocab_size = 32000`

Data artifacts are expected at:

- `train.bin`
- `val.bin`
- `bpe_tokenizer_32k.json`

## What The Pipeline Does

`prepare_data.py`
- scans local parquet shards from `DATASET_PATH`
- reads parquet row groups directly instead of building a temporary Arrow cache
- trains a tokenizer if one does not already exist
- tokenizes the corpus
- streams token IDs directly to `train.bin` and `val.bin`

`dataset.py`
- opens the training binaries with `np.memmap`
- samples random windows for each batch
- moves only the current batch to the GPU

`training.py`
- validates the setup before starting
- auto-resumes from the latest `ckpt_step_*.pt`
- logs loss, LR, throughput, and TFLOPS
- saves periodic checkpoints
- saves `checkpoints/best_model.pt` when validation loss improves

`generate.py`
- loads the tokenizer
- loads a checkpoint
- generates text from a prompt

## Setup

Create or activate your Python environment, then install dependencies:

```powershell
pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

## Configure The Dataset

Before preprocessing, open [prepare_data.py](prepare_data.py) and set:

```python
DATASET_PATH = r"D:\Openweb"
```

That folder should contain your parquet shards.

The preprocessing step now reads those parquet files directly, which avoids the large temporary `datasets` Arrow cache that can exhaust disk space on very large corpora.

If you want different output filenames, update these in [config.py](config.py):

```python
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"
TOKENIZER_PATH = "bpe_tokenizer_32k.json"
```

`prepare_data.py`, `dataset.py`, `training.py`, and `generate.py` now all use those config paths consistently.

## How To Run

### 1. Prepare The Data

From the project root:

```powershell
python prepare_data.py
```

This will:

- train the tokenizer if `bpe_tokenizer_32k.json` does not exist
- create `train.bin`
- create `val.bin`

### 2. Train The Model

```powershell
python training.py
```

Training behavior:

- validates that `train.bin` and `val.bin` exist and are large enough
- resumes automatically from the latest `checkpoints/ckpt_step_*.pt` if present
- prints a training log every 100 steps
- runs evaluation every `eval_interval`
- saves periodic checkpoints every `checkpoint_interval`
- saves `checkpoints/best_model.pt` whenever validation loss improves

### 3. Resume Training

No special command is needed. Just run:

```powershell
python training.py
```

If checkpoints exist, training resumes automatically.

### 4. Generate Text

Use the latest step checkpoint:

```powershell
python generate.py --prompt "The future of AI is" --max-tokens 100
```

Use a specific checkpoint, including the best validation model:

```powershell
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The future of AI is" --max-tokens 100
```

## Recommended Workflow

For a first sanity check:

1. Set smaller values in `config.py`, such as lower `max_iters`.
2. Run `python prepare_data.py`.
3. Run `python training.py`.
4. Confirm that `checkpoints/` contains both step checkpoints and `best_model.pt`.
5. Run `python generate.py` on one of those checkpoints.

After that, scale the training run gradually.

## Smoke Test Before A Long Run

Before launching the full training job, do one short validation run to confirm that:

- data preprocessing completed correctly
- `train.bin` and `val.bin` are readable
- training starts without setup errors
- checkpoints are written
- resume works
- generation works from a produced checkpoint

Temporarily change these values in [config.py](config.py):

```python
max_iters = 300
eval_iters = 10
eval_interval = 100
checkpoint_interval = 150
ENABLE_PROFILING = False
TIMER_TARGET_ITERATION = None
```

Then run:

```powershell
python training.py
```

During the smoke test, check that:

- step logs appear normally
- LR is shown in the training log
- evaluation runs at least once
- a file like `checkpoints/ckpt_step_150.pt` appears
- `checkpoints/best_model.pt` appears

Then run training one more time:

```powershell
python training.py
```

This second launch should resume from the existing checkpoint instead of starting from step 0.

Finally, test generation:

```powershell
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The future of AI is" --max-tokens 80
```

If all of that works, restore your long-run values in `config.py` and start the full run.

## Notes For Larger Datasets

The preprocessing path is now stream-based, which means token IDs are written to disk incrementally instead of being collected in Python lists first. That makes it much more suitable for large datasets.

The training path reads batches from memory-mapped binaries, so the full dataset stays on disk. Only the sampled batch is moved to the GPU each step.

## Optional Learning Rate Schedule

`training.py` supports these optional config fields:

```python
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 2.5e-5
```

If they are not present in `config.py`, training falls back to a constant learning rate.

## Profiling

Profiling is controlled in [config.py](config.py):

```python
ENABLE_PROFILING = True
PROFILING_WINDOW = (100, 110)
```

For normal long training runs, setting `ENABLE_PROFILING = False` is usually a better default.

Profiler traces are written under:

```text
log/profiler/
```

You can inspect them with TensorBoard, Chrome tracing, or Perfetto.

## Checkpoints

The project uses two checkpoint types:

- `checkpoints/ckpt_step_<N>.pt`
- `checkpoints/best_model.pt`

`ckpt_step_<N>.pt` is for periodic resume.

`best_model.pt` tracks the best validation loss seen so far.

## Useful Commands

Prepare data:

```powershell
python prepare_data.py
```

Train:

```powershell
python training.py
```

Generate:

```powershell
python generate.py --prompt "Once upon a time" --max-tokens 100
```

Syntax check:

```powershell
python -m py_compile training.py prepare_data.py dataset.py generate.py config.py tokenizer.py model.py
```

## License

See [LICENSE](LICENSE).
