# Beginner Guide: Attention From Scratch

This guide explains the attention notebook in a beginner-friendly way.

## 1) What is attention?

Attention helps each token decide which other tokens matter most.

Example idea:

- In "Dorothy saw Toto and she smiled", the token "she" should attend strongly to "Dorothy".

Attention computes weighted combinations of token representations so the model can focus on useful context.

## 2) Why attention is powerful

Without attention, models struggle to connect far-apart words.

With attention, each token can use global context in one layer.

That is why transformer models became the foundation of modern LLMs.

## 3) Core pieces in simple words

Every token creates three vectors:

- Query: what I am looking for
- Key: what I contain
- Value: what information I pass forward

Similarity between Query and Key produces attention weights.
Those weights mix Values into a new representation.

## 4) The 4 attention types in your notebook

## Type 1: Multi-Head Self-Attention

- Many heads attend in parallel.
- Each head learns different relation patterns.

## Type 2: Masked Causal Self-Attention

- Same as self-attention, but future tokens are hidden.
- Needed for text generation.

## Type 3: Multi-Query Attention

- Multiple query heads share one K/V head.
- Faster and lighter in memory.

## Type 4: Grouped-Query Attention

- Multiple query heads share a few K/V groups.
- Better quality than MQA, lower cost than full MHA.

Bonus:

- Cross-attention is also included for encoder-decoder style use cases.

## 5) Why masking matters

If mask is not used in generation training, token at position t can peek at token t+1.
That makes training unrealistic and breaks autoregressive behavior.

Masked causal attention prevents this leak.

## 6) Why your architecture is cleaner and stronger now

Your notebook uses:

- RMSNorm (stable and efficient)
- SwiGLU FFN (strong nonlinearity)
- RoPE positional encoding (good relative position handling)
- weight tying (efficient output head)
- profile-based scaling for CPU and RTX 4060

This is much closer to modern practical transformer design than a basic textbook block.

## 7) How training works in this notebook

1. Load BPE tokenizer and encode corpus.
2. Build random next-token batches.
3. Forward pass through transformer.
4. Compute next-token cross-entropy.
5. Backpropagate and update parameters.
6. Evaluate train/val losses at intervals.
7. Generate sample text from a prompt.

## 8) What to do next (simple path)

1. Re-run with your larger dataset.
2. Increase context length gradually.
3. Move profile to RTX 4060 quality.
4. Train longer and save periodic checkpoints.
5. Compare generated text every few hundred steps.

## 9) Common beginner mistakes to avoid

1. Using unmasked attention for autoregressive generation.
2. Increasing model size too quickly on limited hardware.
3. Not tracking validation loss.
4. Changing tokenizer after embedding/model training started.
5. Forgetting to keep config + checkpoint together.

## 10) One-line mental model

Tokenizer makes token IDs, embeddings turn IDs into vectors, attention lets tokens talk to each other, and the LM head predicts the next token.
