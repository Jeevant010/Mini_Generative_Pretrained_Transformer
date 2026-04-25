"""
model.py — GPT Language Model with ablation toggles.

Architecture: Decoder-only Transformer with GQA, RoPE, RMSNorm, SwiGLU, weight tying.

Ablation toggles (controlled via config.py):
    USE_RMSNORM         → Pre-norm vs. no normalization
    USE_ROPE            → Rotary positional encoding vs. none
    USE_FLASH_ATTENTION → F.scaled_dot_product_attention vs. manual matmul
    USE_GQA             → Grouped-Query Attention vs. full Multi-Head Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * x * rms


class Identity(nn.Module):
    """No-op module used when RMSNorm is disabled for ablation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU Feed-Forward
# ─────────────────────────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_mult: float = 3.5, dropout: float = 0.0):
        super().__init__()
        hidden = int(ffn_mult * dim)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w_out(x)
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Positional Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────────────────────
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype).unsqueeze(0).unsqueeze(0), emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


# ─────────────────────────────────────────────────────────────────────────────
# Manual (non-Flash) Attention — used when USE_FLASH_ATTENTION = False
# ─────────────────────────────────────────────────────────────────────────────
def manual_causal_attention(q, k, v, dropout_p=0.0, training=False):
    """
    Standard matrix-multiplication attention with explicit causal mask.
    This is the SLOW path — materialises the full T×T attention matrix.
    Used as an ablation baseline to prove Flash Attention's memory savings.
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Build causal mask
    T = q.size(-2)
    causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(attn_weights, dim=-1)
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    return torch.matmul(attn_weights, v)


# ─────────────────────────────────────────────────────────────────────────────
# Grouped-Query Attention (with ablation toggles)
# ─────────────────────────────────────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        # ── GQA Ablation Toggle ──
        # When USE_GQA is False, use full MHA (n_kv_heads = n_head)
        self.use_gqa = getattr(cfg, "USE_GQA", True)
        self.n_kv_heads = cfg.n_kv_heads if self.use_gqa else cfg.n_head

        # ── RoPE Ablation Toggle ──
        self.use_rope = getattr(cfg, "USE_ROPE", True)

        # ── Flash Attention Ablation Toggle ──
        self.use_flash = getattr(cfg, "USE_FLASH_ATTENTION", True)

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)

        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor):
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # ── RoPE Toggle ──
        if self.use_rope:
            cos, sin = self.rope(q_len, x.device, q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # ── GQA: expand KV heads to match query heads ──
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        # ── Flash Attention Toggle ──
        if self.use_flash:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            attn_out = manual_causal_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                training=self.training
            )

        out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block (with RMSNorm ablation toggle)
# ─────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ── RMSNorm Ablation Toggle ──
        use_rmsnorm = getattr(cfg, "USE_RMSNORM", True)
        NormClass = RMSNorm if use_rmsnorm else Identity

        self.norm1 = NormClass(cfg.n_embd) if use_rmsnorm else Identity()
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = NormClass(cfg.n_embd) if use_rmsnorm else Identity()
        self.ffn = SwiGLU(cfg.n_embd, ffn_mult=cfg.ffn_mult, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GPT Language Model
# ─────────────────────────────────────────────────────────────────────────────
class GPTLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])

        use_rmsnorm = getattr(cfg, "USE_RMSNORM", True)
        self.norm_f = RMSNorm(cfg.n_embd) if use_rmsnorm else Identity()

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight  # Weight tying

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.token_embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
