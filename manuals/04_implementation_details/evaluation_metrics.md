# Evaluation Metrics

## Cross-Entropy Loss

The primary training metric is next-token cross-entropy:

$$
\mathcal{L} =
-\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log P_\theta(y_{b,t} \mid x_{b,\le t})
$$

Lower loss means the model assigns higher probability to correct next tokens.

## Perplexity

Perplexity is:

$$
\operatorname{PPL} = e^{\mathcal{L}}
$$

It can be interpreted as the average effective number of likely next tokens.

At step 60,000:

$$
\mathcal{L}_{val} = 3.517095
$$

so:

$$
\operatorname{PPL} = e^{3.517095} \approx 33.69
$$

## Training And Validation Loss

Both train and validation loss are sampled estimates over `eval_iters = 25` batches.

If train loss decreases but validation loss increases, the model may be overfitting. In the current run, validation loss continues to improve overall through 60,000 steps.

## Throughput

Tokens per second:

$$
\operatorname{tok/s} = \frac{B \times T}{\Delta t}
$$

where `Delta t` is step time in seconds.

Current tokens per step:

$$
20 \times 384 = 7680
$$

## TFLOPS Estimate

The training script approximates FLOPs per token as:

$$
F_{token} \approx 6N_{params}
$$

For:

$$
N_{params} = 117{,}787{,}392
$$

we get:

$$
F_{token} \approx 706{,}724{,}352
$$

FLOPs per step:

$$
F_{step} \approx 706{,}724{,}352 \times 7680
\approx 5.43 \times 10^{12}
$$

TFLOPS:

$$
\operatorname{TFLOPS} =
\frac{F_{step}}{\Delta t \times 10^{12}}
$$

This is an estimate, not a profiler-exact value.

## Gradient Norm

The global gradient norm is:

$$
\|g\|_2 = \sqrt{\sum_i \|g_i\|_2^2}
$$

It is logged to identify instability. Very large spikes can indicate an unstable batch, bad learning rate, missing normalization, or numerical overflow.

## VRAM

When CUDA is available, the script logs:

```python
torch.cuda.max_memory_allocated(device)
```

This helps compare architecture variants such as GQA vs full MHA or Flash Attention vs manual attention.

## Qualitative Samples

Generated text samples are not a replacement for validation loss, but they are useful for spotting:

- repetition
- broken syntax
- topic drift
- memorization-like behavior
- base-model continuation behavior

Current samples show fluent local syntax but inconsistent long-range semantics, which is normal for the current training stage.

