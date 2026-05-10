# Chapter 10.2 — Branch Strategy

## The Git Plan

This project uses a branching strategy to keep the pre-training work clean and separated from post-training experiments.

### Current Branch (main)

Contains everything documented in this book:
- Complete pre-training codebase
- All manuals (this book)
- Training logs and sample outputs
- Evaluation scripts
- Model architecture with all modern components

**Action:** Push everything and create a stable snapshot.

### Next Branch: `post-training`

Will contain:
- SFT training script and dataset loader
- DPO training script
- Extended tokenizer with chat template tokens
- Chat interface (CLI + optional Gradio web UI)
- Post-training evaluation results

**Branch from:** main (after pushing pre-training docs)

### Why Branch?

1. **Clean separation.** Pre-training is complete and should not be mixed with experimental post-training code.
2. **Safe rollback.** If SFT experiments break something, we can always go back to the clean pre-training branch.
3. **Clear documentation.** Each branch has its own documentation scope. This book covers pre-training. The post-training branch will have its own README and guides.
4. **Reproducibility.** Anyone can clone the main branch and reproduce the pre-training results without needing post-training dependencies (like the Dolly dataset).

## Push Checklist

Before pushing the current branch:

- [ ] All chapter files are created in `manuals/`
- [ ] Old manual folders are removed
- [ ] `model.py` changes compile (generate() enhancements)
- [ ] `evaluation/quality_metrics.py` compiles
- [ ] No broken imports in existing code
- [ ] `.gitignore` excludes: `checkpoints/`, `train.bin`, `val.bin`, `logs/` (large files)
- [ ] Commit message: `docs: complete beginner-friendly manual book for pre-training phase`

## After Pushing

1. Create the `post-training` branch
2. Start implementing SFT using `advanced/02_supervised_fine_tuning.md` as the guide
3. Download the Dolly 15K dataset
4. Run SFT training
5. Evaluate with the quality metrics
6. If satisfactory, proceed to DPO

## Files That Stay vs Files For Next Branch

### Stay on main (push now)

| File/Folder | Status |
|---|---|
| `model.py` | Updated `generate()` — safe, does not affect training |
| `evaluation/quality_metrics.py` | New file — nothing imports it |
| `manuals/` (all chapters) | Documentation only |
| All existing source files | Unchanged |

### For next branch only

| File/Folder | Status |
|---|---|
| `sft_training.py` | To be created |
| `dpo_training.py` | To be created |
| `chat.py` | To be created |
| Extended tokenizer | To be created |
| SFT/DPO checkpoints | To be generated |

**The code changes on main are safe to push right now — they do not affect the running training loop.**
