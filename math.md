# Mathematical Foundations of the Generative Pre-Trained Transformer Pipeline

This document provides a comprehensive, rigorous mathematical breakdown of the exact equations driving the pipeline you just built. It is designed as a deep-study reference for the architecture, pre-training, supervised fine-tuning, and preference alignment.

---

## 1. The Core Architecture

### 1.1 Tokenization and Embedding
Given an input sequence of tokens $X = (x_1, x_2, \dots, x_N)$ where $x_i \in \mathbb{N}$ (token IDs), we project them into a continuous latent space of dimension $d_{\text{model}}$.

$$
E = \text{Embedding}(X) \in \mathbb{R}^{N \times d_{\text{model}}}
$$

### 1.2 Root Mean Square Normalization (RMSNorm)
Unlike standard LayerNorm which shifts by the mean, RMSNorm only scales by the root mean square, which is computationally faster and mathematically sufficient. Given an input vector $x \in \mathbb{R}^{d}$:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$
Where $\gamma$ is a learnable scale parameter and $\epsilon$ is a small constant for numerical stability.

### 1.3 Rotary Positional Embeddings (RoPE)
Instead of adding positional embeddings, RoPE rotates the query and key vectors in a 2D plane to encode relative positioning. For a vector $v$ at position $m$, we group its $d$ dimensions into $d/2$ pairs and apply a rotation matrix $R_{\Theta, m}$:

$$
\Theta = \{ \theta_i = 10000^{-2(i-1)/d} \}_{i=1}^{d/2}
$$
$$
v_{\text{rotated}} = 
\begin{pmatrix}
v_1 \cos(m\theta_1) - v_2 \sin(m\theta_1) \\
v_1 \sin(m\theta_1) + v_2 \cos(m\theta_1) \\
\vdots \\
v_{d-1} \cos(m\theta_{d/2}) - v_d \sin(m\theta_{d/2}) \\
v_{d-1} \sin(m\theta_{d/2}) + v_d \cos(m\theta_{d/2})
\end{pmatrix}
$$

### 1.4 Grouped-Query Attention (GQA)
To compute the self-attention, we project the normalized input $X$ into Queries ($Q$), Keys ($K$), and Values ($V$). In GQA, multiple query heads share a single key/value head to save memory.

$$
Q_i = X W_i^Q, \quad K_j = X W_j^K, \quad V_j = X W_j^V
$$

The attention score for a single query head $i$ and its corresponding key/value head $j$ is computed as:

$$
\text{Attention}(Q_i, K_j, V_j) = \text{softmax}\left( \frac{Q_i K_j^T}{\sqrt{d_k}} + M \right) V_j
$$
Where $d_k$ is the dimension of the head, and $M$ is the causal masking matrix (to prevent seeing the future):
$$
M_{a,b} = 
\begin{cases} 
0 & \text{if } a \ge b \\ 
-\infty & \text{if } a < b 
\end{cases}
$$

### 1.5 SwiGLU Feed-Forward Network
Instead of a standard ReLU network, we use the Swish-Gated Linear Unit (SwiGLU).

Given $x$, the Swish activation is:
$$ \text{Swish}(x) = x \cdot \sigma(\beta x) $$

The feed-forward equation becomes:
$$ \text{SwiGLU}(x) = (\text{Swish}(x W_1) \odot x W_2) W_3 $$
Where $W_1$ and $W_2$ project the input to a hidden dimension, and $W_3$ projects it back to $d_{\text{model}}$.

---

## 2. Phase 1: Pre-training (Causal Language Modeling)

During pre-training, the objective is to predict the next token $x_{t}$ given all previous tokens $X_{<t}$.

### 2.1 Logits and Softmax
The final output of the transformer blocks is projected back into the vocabulary space:
$$ z = H_{\text{final}} W_{\text{vocab}} \in \mathbb{R}^{N \times |V|} $$

The probability distribution over the vocabulary for token $t$ is obtained via softmax:
$$ P(x_t | X_{<t}) = \frac{\exp(z_{t, x_t})}{\sum_{v \in V} \exp(z_{t, v})} $$

### 2.2 Standard Cross-Entropy Loss
The model is trained to minimize the negative log-likelihood over the entire sequence:
$$ \mathcal{L}_{\text{CLM}} = - \frac{1}{N-1} \sum_{t=1}^{N-1} \log P(x_{t+1} | X_{\le t}) $$

---

## 3. Phase 2: Supervised Fine-Tuning (SFT)

During SFT, the prompt consists of an Instruction $I$ of length $L_I$, and a Response $R$ of length $L_R$. The total sequence length is $N = L_I + L_R$.

### 3.1 Masked Cross-Entropy Loss
We do *not* want to penalize the model for failing to predict the user's instruction. We only compute loss on the response tokens. The mask function is:

$$
w_t = 
\begin{cases} 
0 & \text{if } t \le L_I \\ 
1 & \text{if } t > L_I 
\end{cases}
$$

The SFT loss equation isolates the response generation:
$$ \mathcal{L}_{\text{SFT}} = - \frac{1}{L_R} \sum_{t=L_I}^{N-1} w_t \log P(x_{t+1} | X_{\le t}) $$

---

## 4. Phase 3: Direct Preference Optimization (DPO)

In DPO, we align the SFT model to human preferences without requiring a separate Reward Model (like RLHF).

### 4.1 The Bradley-Terry Preference Model
Assume a latent reward function $r^*(x, y)$ that assigns a scalar score to a response $y$ given a prompt $x$. The probability that human judges prefer response $y_w$ (winner) over $y_l$ (loser) is:

$$ P(y_w \succ y_l | x) = \sigma(r^*(x, y_w) - r^*(x, y_l)) $$
Where $\sigma$ is the logistic sigmoid function.

### 4.2 The Optimal Policy
RLHF tries to maximize the reward while staying close to the SFT model (to prevent the model from degenerating). The optimal policy $\pi^*$ under a KL-divergence constraint has a closed-form solution:

$$ \pi^*(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp \left( \frac{1}{\beta} r^*(x, y) \right) $$

Where:
* $\pi_{\text{ref}}$ is our frozen SFT model.
* $\beta$ is a temperature parameter controlling how much we allow the model to deviate from $\pi_{\text{ref}}$.
* $Z(x)$ is the partition function.

### 4.3 The DPO Loss Equation
By algebraically re-arranging the optimal policy equation, we can express the reward *purely in terms of the language model probabilities*:

$$ r^*(x, y) = \beta \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x) $$

Substituting this reward back into the Bradley-Terry model cancels out the annoying $Z(x)$ term entirely! This leads directly to the DPO loss, which we minimize:

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right] $$

**Mathematical Breakdown of the Loss:**
1. $\log \pi_\theta(y_w | x)$: The log-probability our current model assigns to the winning response.
2. $\log \pi_{\text{ref}}(y_w | x)$: The baseline log-probability from the SFT model.
3. The difference between these log-probs is the **implicit reward**.
4. The loss decreases when the implicit reward of the winner ($y_w$) is significantly higher than the implicit reward of the loser ($y_l$).
