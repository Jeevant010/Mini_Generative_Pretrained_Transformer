# Chapter 3.4 — RoPE: How the Model Knows Word Order

## The Problem

Attention compares every word with every other word. But it does not inherently know the **order** of words. Without positional information, "The cat ate the fish" and "The fish ate the cat" would look the same to the model — both have the same words, just in different positions.

The model needs a way to know: this word is at position 1, this word is at position 5.

## The Solution: Rotary Positional Embeddings (RoPE)

RoPE encodes position by **rotating** the Query and Key vectors. Each position gets a different rotation angle. This means:

- Nearby positions have similar rotations (small angle difference)
- Far-apart positions have very different rotations (large angle difference)

When the model computes attention (Query × Key), the rotation naturally encodes the **relative distance** between two positions. Words that are close together in the text have a built-in similarity boost.

## How It Works (Simplified)

Imagine each dimension pair in the Query/Key vectors as a point on a circle. RoPE rotates this point by an angle that depends on the position:

- Position 0: rotate by 0°
- Position 1: rotate by θ°
- Position 2: rotate by 2θ°
- Position 100: rotate by 100θ°

When the model compares a Query at position 5 with a Key at position 3, the rotation difference is 2θ°. If it compares position 5 with position 1, the difference is 4θ° — a bigger angle, indicating they are farther apart.

## Why RoPE Instead of Learned Positions?

Older models (like the original GPT-2) used a simpler approach: a learned embedding for each position. Position 0 had one learned vector, position 1 had another, etc. This works but has a problem: the model can only handle positions it has seen during training. If trained on 384-token sequences, it cannot handle 385 tokens.

RoPE does not have this limitation — the rotation formula works for any position. This makes it possible to extend the context length after training (though quality may degrade).

## The Math (Optional)

RoPE rotates pairs of dimensions by an angle that depends on position $m$:

$$
\theta_i = 10000^{-2i/d_h}
$$

For each pair of dimensions $(a, b)$, the rotation is:

$$
\begin{bmatrix} a' \\ b' \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}
$$

RoPE is applied to Query and Key vectors only, not to Value vectors.

## Key Takeaway

Without RoPE, the model would not understand word order. "I love you" and "you love I" would generate the same attention scores. RoPE gives the model a sense of position, which is critical for grammar and meaning.
