# RTX 4060 Tuning Guide For Attention Notebook

This guide helps you tune the attention notebook from CPU to RTX 4060 efficiently.

## 1) Available profiles in notebook

- cpu_safe
- cpu_quality
- rtx_4060_balanced
- rtx_4060_quality
- rtx_4060_max

## 2) Quick profile selection

Start order:

1. rtx_4060_quality
2. if unstable, fallback to rtx_4060_balanced
3. if still OOM, reduce context/batch before reducing model depth

## 3) If you get CUDA out-of-memory

Reduce in this exact order:

1. batch_size
2. max_seq_len
3. grad_accum_steps increase (simulate larger effective batch)
4. n_layers
5. d_model

Keep n_heads consistent with d_model divisibility.

## 4) Why sequence length is expensive

Attention compute grows approximately with sequence length squared.

If sequence length doubles, attention compute is about 4x.

That is why context length should be increased gradually.

## 5) Practical stable progression

Recommended progression on 4060:

1. seq_len 128, short run sanity
2. seq_len 256, medium run
3. seq_len 384, longer run after confirming stability

Do not jump directly to maximum settings.

## 6) Throughput tips

1. Keep mixed precision enabled on CUDA.
2. Use GQA for strong quality/efficiency balance.
3. Keep dataloader logic simple for contiguous token batches.
4. Avoid very frequent evaluations (set eval_interval sensibly).

## 7) Quality tips

1. Prefer GQA over MQA when quality is priority.
2. Keep RoPE enabled for better positional behavior.
3. Train on broader dataset for semantic/generalization gains.
4. Evaluate text samples at fixed prompts to track improvement.

## 8) Environment checks when GPU is not used

If notebook shows CPU unexpectedly:

1. Check torch.cuda.is_available().
2. Ensure VS Code kernel points to CUDA-enabled environment.
3. Verify PyTorch CUDA build in that environment.
4. Restart kernel after environment changes.

## 9) Suggested experiment matrix

Run these three first:

1. Balanced baseline: rtx_4060_balanced, steps 400
2. Quality baseline: rtx_4060_quality, steps 800
3. Long-context trial: rtx_4060_quality with larger max_seq_len, same steps

Track:

- train loss
- val loss
- generation quality at fixed prompt
- runtime per 100 steps

## 10) Best default for your current roadmap

For your stage right now, the most practical choice is:

- GQA attention
- RoPE enabled
- rtx_4060_quality profile
- gradual context-length scaling

This usually gives better quality than tiny models while remaining feasible on 4060-class hardware.
