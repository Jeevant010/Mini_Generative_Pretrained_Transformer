"""
model.py — GPT-style language model architecture for the SLM.

Contains all nn.Module sub-components:
  • Head              — single causal self-attention head
  • MultiHeadAttention — parallel multi-head attention
  • FeedForward       — position-wise feed-forward network
  • Block             — transformer block (pre-norm variant)
  • GPTLanguageModel  — full model with embeddings, blocks, and LM head
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import n_embd, n_head, n_layer, block_size, dropout


# ─── Single Attention Head ─────────────────────────────
class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Lower-triangular mask for causal (autoregressive) attention
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)       # (B, T, hs)
        q = self.query(x)     # (B, T, hs)

        # Scaled dot-product attention
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)     # (B, T, hs)
        out = wei @ v         # (B, T, hs)
        return out


# ─── Multi-Head Attention ──────────────────────────────
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# ─── Feed-Forward Network ─────────────────────────────
class FeedForward(nn.Module):
    """Position-wise FFN: Linear → ReLU → Linear → Dropout."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Transformer Block ────────────────────────────────
class Block(nn.Module):
    """Transformer block: pre-norm self-attention + feed-forward."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ─── GPT Language Model ───────────────────────────────
class GPTLanguageModel(nn.Module):
    """
    A small GPT-style language model.

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens in the vocabulary.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    # ── Weight Initialization ──────────────────────────
    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward Pass ───────────────────────────────────
    def forward(self, index: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Parameters
        ----------
        index : Tensor of shape (B, T)
        targets : Tensor of shape (B, T) or None

        Returns
        -------
        logits : Tensor of shape (B, T, vocab_size)
        loss   : scalar Tensor or None
        """
        B, T = index.shape

        tok_emb = self.token_embedding_table(index)                   # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=index.device)
        )                                                             # (T, C)
        x = tok_emb + pos_emb                                        # (B, T, C)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                     # (B, T, V)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    # ── Text Generation ────────────────────────────────
    @torch.no_grad()
    def generate(self, index: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Auto-regressively generate `max_new_tokens` new tokens.

        Parameters
        ----------
        index : Tensor of shape (B, T)
            Seed / prompt token IDs.
        max_new_tokens : int

        Returns
        -------
        Tensor of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = index[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits_last = logits[:, -1, :]            # (B, vocab_size)
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, idx_next), dim=1)
        return index
