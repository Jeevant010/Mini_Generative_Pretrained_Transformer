# Chapter 3.5 — SwiGLU: The Feed-Forward Layer

## What Happens After Attention

After the attention step gathers information from previous tokens, the model needs to **process** that information. This is the job of the feed-forward layer — it takes the combined information and transforms it.

Think of attention as "gathering ingredients" and the feed-forward layer as "cooking" — it combines the raw ingredients into something useful.

## What SwiGLU Is

SwiGLU is a specific type of feed-forward layer. The name comes from combining two ideas:

- **SiLU** (Sigmoid Linear Unit) — A smooth activation function
- **GLU** (Gated Linear Unit) — A gating mechanism that controls information flow

## How It Works (Plain English)

The feed-forward layer does three things:

1. **Expand**: Take the 768-dimensional vector and project it to a larger space (2688 dimensions)
2. **Gate**: Use one projection to control which information passes through (the "gate")
3. **Compress**: Project back down from 2688 to 768 dimensions

The gating mechanism is the key innovation. Instead of just expanding and compressing, SwiGLU uses two separate projections of the input:

- One projection creates the "content" — what information might be useful
- The other projection creates the "gate" — a set of values between 0 and 1 that control how much of each content dimension passes through

The content is multiplied by the gate, so the model can selectively block or allow information.

## Why 2688 Dimensions?

The hidden dimension is calculated as:

$$
d_{ff} = \lfloor 3.5 \times 768 \rfloor = 2688
$$

The multiplier 3.5 is called `ffn_mult` in our config. This means the feed-forward layer temporarily expands the representation to 3.5× its size, processes it, and compresses it back.

Why expand? More dimensions = more room to represent complex patterns. The expansion-compression pattern gives the model a richer internal workspace.

## The Math (Optional)

$$
\text{SwiGLU}(x) = W_{out} \left( \text{SiLU}(xW_1) \odot xW_2 \right)
$$

Where:
- $W_1$ and $W_2$ project from 768 → 2688
- $\odot$ is element-wise multiplication (the gating operation)
- $W_{out}$ projects from 2688 → 768
- $\text{SiLU}(z) = z \cdot \sigma(z)$ where $\sigma$ is the sigmoid function

## SwiGLU vs Standard Feed-Forward

Older Transformers used a simpler feed-forward: just expand → ReLU → compress. SwiGLU adds the gating mechanism, which has been shown to improve model quality in practice. LLaMA, PaLM, and most modern models use SwiGLU.

## Parameters in the Feed-Forward Layer

Each SwiGLU layer has three weight matrices:

- $W_1$: 768 × 2688 = 2,064,384 parameters
- $W_2$: 768 × 2688 = 2,064,384 parameters
- $W_{out}$: 2688 × 768 = 2,064,384 parameters
- Total per layer: **6,193,152 parameters**

With 12 layers: 6,193,152 × 12 = **74,317,824 parameters** — about 63% of the total model.

The feed-forward layers are by far the largest component of the model.
