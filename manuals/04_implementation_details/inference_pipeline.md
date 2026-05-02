# Inference Pipeline

## Entry Point

Use:

```powershell
python generate.py --prompt "The future of AI is" --max-tokens 100
```

To use a specific checkpoint:

```powershell
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The future of AI is" --max-tokens 100
```

## Loading Steps

`generate.py` performs:

1. Select device from `config.device`.
2. Load `bpe_tokenizer_32k.json`.
3. Find latest `ckpt_step_*.pt` if no checkpoint is specified.
4. Construct `GPTLanguageModel(config)`.
5. Load checkpoint state dict.
6. Encode prompt with BOS.
7. Generate tokens autoregressively.
8. Decode token IDs to text.

## Autoregressive Generation

At each generation step:

$$
x_{1:t} \rightarrow \operatorname{model}(x_{1:t})
$$

Only the last-position logits are used:

$$
z = \ell_t
$$

The model samples one new token:

$$
x_{t+1} \sim \operatorname{Categorical}(\operatorname{softmax}(z))
$$

and appends it to the context.

## Context Cropping

The model can only use the last `block_size` tokens:

```python
idx_cond = idx[:, -self.cfg.block_size:]
```

Current context length:

$$
T = 384
$$

If the generated sequence becomes longer than 384 tokens, older tokens are dropped from the active context.

## Temperature

Temperature rescales logits:

$$
z'_i = \frac{z_i}{\tau}
$$

where `tau` is temperature.

- Lower temperature makes output more deterministic.
- Higher temperature makes output more random.

Current default in `generate.py`:

```python
temperature = 0.8
```

## Top-k Sampling

Top-k keeps only the `k` most likely tokens:

$$
z_i =
\begin{cases}
z_i, & i \in \operatorname{TopK}(z,k) \\
-\infty, & \text{otherwise}
\end{cases}
$$

Current default:

```python
top_k = 50
```

## Interpreting Output

The model is a base language model. It continues text; it does not automatically follow instructions.

For example, the prompt:

```text
how can i help
```

is treated as a text prefix. A continuation such as article text, interview text, or dialogue text is expected from a base model. To make it respond like an assistant, the model needs instruction tuning on examples like:

```text
User: how can i help?
Assistant: You can ask me to explain, summarize, debug, or write code.
```

