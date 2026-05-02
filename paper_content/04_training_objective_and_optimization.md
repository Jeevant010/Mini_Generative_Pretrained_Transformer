# Training Objective And Optimization

## Autoregressive Language Modeling

The training objective is next-token prediction. Given a token sequence:

$$
x_1, x_2, ..., x_T
$$

the model estimates:

$$
P_\theta(x_t \mid x_1, x_2, ..., x_{t-1})
$$

for every position `t`.

The joint probability of the full sequence is factorized as:

$$
P_\theta(x_1, ..., x_T) = \prod_{t=1}^{T} P_\theta(x_t \mid x_{<t})
$$

Taking the log gives:

$$
\log P_\theta(x_1, ..., x_T) = \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

Training maximizes this log-likelihood, or equivalently minimizes negative log-likelihood.

## Logits And Softmax

For each token position, the model produces logits:

$$
z_t \in \mathbb{R}^{V}
$$

where `V = 32000`. The probability of token `i` is:

$$
P_\theta(x_{t+1}=i \mid x_{\leq t}) =
\frac{\exp(z_{t,i})}{\sum_{j=1}^{V} \exp(z_{t,j})}
$$

## Cross-Entropy Loss

For target token ID `y_t`, the per-token loss is:

$$
\ell_t = -\log P_\theta(y_t \mid x_{\leq t})
$$

For a batch with `B` sequences and context length `T`, the mean training loss is:

$$
\mathcal{L}(\theta) =
-\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log P_\theta(y_{b,t} \mid x_{b,\leq t})
$$

In the code, this is implemented as:

```python
loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
```

The reshape flattens all batch and time positions into one large classification problem over the vocabulary.

## Perplexity

Perplexity converts cross-entropy loss into a more interpretable value:

$$
\operatorname{PPL} = \exp(\mathcal{L})
$$

If the validation loss is:

$$
\mathcal{L}_{val} = 3.517095
$$

then:

$$
\operatorname{PPL} = e^{3.517095} \approx 33.69
$$

Perplexity can be interpreted as the model's average effective branching factor. A perplexity of 33.69 means the model is, on average, as uncertain as choosing among roughly 34 plausible next tokens at each step.

## AdamW Optimization

The optimizer is AdamW. Adam keeps moving averages of gradients and squared gradients:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

Bias-corrected estimates are:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

AdamW updates parameters using decoupled weight decay:

$$
\theta_t =
\theta_{t-1}
- \eta_t
\left(
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
+ \lambda \theta_{t-1}
\right)
$$

where:

- `eta_t` is the learning rate at step `t`.
- `lambda` is the weight decay coefficient.
- `g_t` is the gradient.

The code uses:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
```

## Learning-Rate Warmup

During warmup, the learning rate increases linearly:

$$
\eta_t = \eta_{max}\frac{t+1}{T_{warmup}}
$$

where:

- `eta_max = 2.5e-4`
- `T_warmup = 2000`

At step 0:

$$
\eta_0 = 2.5 \times 10^{-4} \times \frac{1}{2000}
= 1.25 \times 10^{-7}
$$

Warmup reduces early instability by avoiding a large optimizer step before the model has learned reasonable activation scales.

## Cosine Learning-Rate Decay

After warmup, the learning rate follows cosine decay:

$$
r_t = \frac{t - T_{warmup}}{T_{decay} - T_{warmup}}
$$

$$
c_t = \frac{1}{2}(1 + \cos(\pi r_t))
$$

$$
\eta_t = \eta_{min} + c_t(\eta_{max} - \eta_{min})
$$

For this run:

- `eta_max = 2.5e-4`
- `eta_min = 2.5e-5`
- `T_decay = 150000`

The schedule starts carefully, trains aggressively during the middle of the run, then anneals toward a smaller learning rate.

## Gradient Clipping

The training loop clips gradient norm:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
```

The global gradient norm is:

$$
\|g\|_2 =
\sqrt{
\sum_i \|g_i\|_2^2
}
$$

If:

$$
\|g\|_2 > c
$$

then gradients are rescaled:

$$
g_i \leftarrow g_i \frac{c}{\|g\|_2}
$$

where `c = 1.0`. This prevents rare unstable batches from causing very large updates.

## Mixed Precision

The training loop uses autocast with bfloat16:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits, loss = model(xb, yb)
```

bfloat16 reduces memory bandwidth and improves speed while retaining a wider exponent range than float16. This is useful for Transformer training because activation tensors are large.

## Throughput And FLOP Estimate

The training script estimates FLOPs per token as:

$$
F_{token} \approx 6N_{params}
$$

where `N_params` is the number of trainable parameters. The factor 6 is a common approximation for forward and backward training cost.

For this model:

$$
N_{params} = 117{,}787{,}392
$$

so:

$$
F_{token} \approx 706{,}724{,}352
$$

Each batch contains:

$$
B \times T = 20 \times 384 = 7680
$$

tokens, so estimated FLOPs per step are:

$$
F_{step} \approx 706{,}724{,}352 \times 7680
\approx 5.43 \times 10^{12}
$$

If a step takes `dt` seconds, throughput is:

$$
\operatorname{tokens/sec} = \frac{BT}{dt}
$$

and estimated TFLOPS are:

$$
\operatorname{TFLOPS} = \frac{F_{step}}{dt \times 10^{12}}
$$

## Generation Objective

During generation, the model samples one token at a time. At each step:

$$
z = \operatorname{model}(x)_{last}
$$

Temperature rescales logits:

$$
z'_i = \frac{z_i}{\tau}
$$

where `tau` is temperature. Lower temperature makes the distribution sharper; higher temperature makes it more random.

The probability distribution is:

$$
p_i = \operatorname{softmax}(z')_i
$$

With top-k sampling, only the `k` largest logits are retained:

$$
z_i =
\begin{cases}
z_i, & i \in \operatorname{TopK}(z, k) \\
-\infty, & \text{otherwise}
\end{cases}
$$

The current generation script uses:

- `temperature = 0.8`
- `top_k = 50`

