# Chapter 3.7 — Putting It All Together: The Transformer Block

## The Full Block

Each Transformer block combines the components we have discussed. Here is the complete flow through one block:

```
Input (768 dimensions per token)
    │
    ├──→ RMSNorm ──→ Attention ──→ Add back to input (residual connection)
    │                                    │
    │                                    ▼
    ├──→ RMSNorm ──→ SwiGLU ────→ Add back (another residual connection)
    │                                    │
    │                                    ▼
    └──────────────────────────────── Output (768 dimensions per token)
```

## Residual Connections

Notice the "Add back" steps. These are called **residual connections** — the output of attention/SwiGLU is **added** to the original input, rather than replacing it.

Why? Imagine you are writing an essay and someone gives you feedback. You do not throw away your essay and start over — you add their suggestions to your existing work. Residual connections work the same way. The attention and SwiGLU layers provide "suggestions" that get added to the existing representation.

This has a crucial engineering benefit: it creates a direct path for information to flow through all 12 blocks without being transformed at every step. This makes training much more stable.

## The Full Model Stack

Our model stacks 12 of these blocks on top of each other:

```
Token Embeddings (32,000 → 768 dimensions)
    ↓
Block 1:  RMSNorm → Attention → Add → RMSNorm → SwiGLU → Add
    ↓
Block 2:  RMSNorm → Attention → Add → RMSNorm → SwiGLU → Add
    ↓
Block 3:  RMSNorm → Attention → Add → RMSNorm → SwiGLU → Add
    ↓
    ... (blocks 4 through 11) ...
    ↓
Block 12: RMSNorm → Attention → Add → RMSNorm → SwiGLU → Add
    ↓
Final RMSNorm
    ↓
LM Head (768 → 32,000 dimensions) — produces probabilities for next token
```

Each block sees the same 768-dimensional vectors but can transform them in different ways. Early blocks tend to learn simple patterns (syntax, common phrases). Later blocks tend to learn more abstract patterns (topic, coherence).

## Why 12 Blocks?

The number of blocks (layers) controls how "deep" the model can think. More layers = more processing steps = more complex patterns.

| Layers | Model Scale | Example |
|---|---|---|
| 6 | Very small | Quick experiments |
| 12 | Small | Our model, GPT-2 Small |
| 24 | Medium | GPT-2 Medium |
| 36 | Large | GPT-2 Large |
| 96 | Very Large | GPT-3 175B |

We chose 12 layers because it fits comfortably on our 8 GB GPU while being deep enough to learn meaningful patterns.

## Information Flow

When the model processes "The cat sat on the ___":

1. **Block 1**: Learns that "cat" is a noun, "sat" is a verb. Basic word-level understanding.
2. **Blocks 2-4**: Learns that "cat sat" forms a subject-verb pair. Grammatical structure.
3. **Blocks 5-8**: Learns that "sat on" suggests a location is coming. Semantic understanding.
4. **Blocks 9-12**: Combines everything to predict that "mat", "floor", or "chair" are likely next words.

This is a simplification — in reality, the roles of layers are not so cleanly separated. But the general principle holds: deeper layers build on the understanding of earlier layers.
