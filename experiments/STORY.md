# Model Training Story

## The Goal
To build, train, and align a 118 Million parameter language model from scratch on a consumer laptop.

## Phase 1: Pre-training (Completed)
- **Architecture**: 118M parameters (12 layers, 12 heads, 768 embedding dimension).
- **Data**: Trained on 10GB of OpenWebText.
- **Duration**: Reached step 149,000 (roughly 5.5B tokens).
- **Status**: Absolute success. The model achieved a perplexity of 27.40 and demonstrates strong syntactical and structural understanding of the English language.

## Phase 2: Supervised Fine-Tuning (SFT) & Benchmarks (In Progress)
- Transitioning from an autocomplete text predictor to an instruction-following assistant.
- Using the Databricks Dolly 15K dataset.
- Establishing formal benchmark scores using EleutherAI's lm-evaluation-harness.

## Challenges & Learnings (Phase 2)
To ensure future robustness, we meticulously documented the engineering and mathematical hurdles faced during the transition to SFT:

1. **Windows Unicode Crashes**: The `quality_metrics.py` evaluation script initially crashed on Windows because the terminal could not encode standard UTF-8 emojis (`✅`, `⚠`) and box-drawing characters (`─`). We learned to strictly use ASCII-safe characters in terminal outputs for cross-platform compatibility.
2. **Terminal Buffering**: Python heavily buffers `print()` statements, causing the terminal to appear "frozen" during training. We learned to explicitly pass `flush=True` to all training loop print statements to force real-time terminal updates.
3. **Orphaned GPU Processes**: When forcefully stopping a `conda run` task on Windows, the OS killed the conda wrapper but left the actual `python.exe` script running invisibly on the GPU (resulting in mysterious checkpoints saving "out of nowhere"). We learned to use `tasklist` and `taskkill /F /PID` to hunt down and terminate rogue background GPU processes.
4. **The Off-By-One Shifting Bug (Catastrophic Failure)**: Our first SFT training run failed catastrophically (Perplexity exploded to 162,267 and the model only output blank lines). The root cause was an architectural oversight: causal language models must look at token $N$ to predict token $N+1$. Our loss function was accidentally feeding token $N$ and asking the model to predict token $N$. Because the model could "cheat" by looking at the answer, it learned to just copy the input and forgot how to predict the future. We fixed this by correctly shifting the logits (`logits[..., :-1, :]`) and labels (`yb[..., 1:]`) before calculating cross-entropy loss.
