import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse


device = {
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'  
}   

# HyperParameters
batch_size = 32
block_size = 128
max_iters = 20000
learning_rate = 3e-4
eval_iters = 200
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.1


# chars = ""
# with open('../wizard_of_oz.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
#     chars = sorted(list(set(text)))

# vocab_size = len(chars)    

# Suppose chars is a list or string of unique characters
# e.g., chars = sorted(list(set(text)))

# string_to_int = {ch: i for i, ch in enumerate(chars)}
# int_to_string = {i: ch for i, ch in enumerate(chars)}

# # Encode: string -> list of integers
# encode = lambda s: [string_to_int[c] for c in s]

# # Decode: list of integers -> string
# decode = lambda l: ''.join(int_to_string[i] for i in l)

# # Convert encoded data to a tensor
# data = torch.tensor(encode(text), dtype=torch.long)

# print(data[:100])



def get_random_chunk(split):
    filename = "../wizard_of_oz.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)
            
            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size-1)
            
            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long )
            
        return data
    
    
    
def get_batch(split) :
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + batch_size] for i in ix])
    y = torch.stack([data[i + 1 : i + batch_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y




class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # tril is a lower-triangular mask for causal self-attention
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                     # (B, T, hs)
        q = self.query(x)                   # (B, T, hs)

        # attention weights
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted sum of values
        v = self.value(x)                   # (B, T, hs)
        out = wei @ v                       # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate head outputs along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ───────────────────────────────────────────────
# GPT Model
# ───────────────────────────────────────────────

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        # token + position embeddings (device-safe)
        tok_emb = self.token_embedding_table(index)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=index.device)
        )                                           # (T, C)
        x = tok_emb + pos_emb                       # (B, T, C)

        # pass through transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                    # (B, T, vocab_size)

        # compute loss if targets given
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T)
            )

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = index[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits_last = logits[:, -1, :]          # (B, vocab_size)
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, idx_next), dim=1)
        return index


# ───────────────────────────────────────────────
# Instantiate model and move to device
# ───────────────────────────────────────────────
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model = GPTLanguageModel(vocab_size).to(device)
m = model



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # switch to eval mode (no dropout, etc.)

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)  # keep on same device

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss  # keep as tensor, not .item()

        # Move mean to CPU only when needed
        out[split] = losses.mean().detach().cpu()

    model.train()  # back to training mode
    return out


import torch
import torch._dynamo
import pickle

# ───────────────────────────────
# Device setup
# ───────────────────────────────
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True              # best kernel for fixed shapes
torch.set_float32_matmul_precision("high")         # enable fast TF32 on Ampere+
torch._dynamo.config.suppress_errors = True        # avoid torch.compile crashes

# ───────────────────────────────
# Model setup (safe compile)
# ───────────────────────────────

try:
    model = torch.compile(model)
    print("✅ Using torch.compile() optimized mode.")
except Exception as e:
    print(f"⚠️ torch.compile() failed ({e}); running in eager mode.")

m = model  # keep alias consistent

# ───────────────────────────────
# Data batching
# ───────────────────────────────
def get_batch(split):
    data = get_random_chunk(split)
    if len(data) <= block_size + 1:
        raise ValueError("Random chunk too small for requested block_size.")
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), device=device)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

# ───────────────────────────────
# Optimizer and AMP (mixed precision)
# ───────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ✅ New API for AMP GradScaler (no deprecation warning)
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

# ───────────────────────────────
# Training loop
# ───────────────────────────────
for iter in range(max_iters):

    # Evaluate every 100 steps instead of every step
    if iter % 100 == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"Step {iter:5d} | "
            f"train loss {losses['train'].item():.3f} | "
            f"val loss {losses['val'].item():.3f}"
        )

    xb, yb = get_batch("train")

    # Mixed precision forward/backward
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
        logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    # Clip gradients for stability
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Step optimizer safely
    scaler.step(optimizer)
    scaler.update()

print(f"✅ Training complete. Final loss: {loss.item():.4f}")

# ───────────────────────────────
# Save model
# ───────────────────────────────
torch.save(model.state_dict(), "model-01.pt")
print("💾 Model saved to model-01.pt")
