# Chapter 9.3 — Key Takeaways

## Everything We Learned the Hard Way

This chapter summarizes the most important lessons from building and training Mini GPT. Each lesson was learned through experience, not theory.

### Lesson 1: Data Quality > Data Quantity

Having 5 billion tokens of data is pointless if 10% of it is non-English garbage. The "ibn nimy" problem proved that **one bad data pattern** can corrupt the model's output. Always invest more time in filtering than in collecting more data.

### Lesson 2: The Model Cannot Do What It Was Not Trained For

Our model was trained on web text continuation. When users tried to have a conversation ("Hello, how are you?"), the model did not answer — it continued writing a web article. This is not a bug; it is the expected behavior of a base language model. Conversational ability requires a separate fine-tuning step (SFT), which is planned for the next branch.

### Lesson 3: Perplexity Is Not Everything

Our model has a good perplexity (33.69), but it can still produce repetitive or incoherent text. Perplexity measures average next-token prediction accuracy, which misses:
- Repetition patterns
- Topic coherence over long passages
- Factual accuracy (hallucination)

Always use multiple metrics: Perplexity + Distinct-N + Self-BLEU + Entropy.

### Lesson 4: Generation Settings Are Critical

The same model can produce excellent or terrible text depending on the generation settings:

| Settings | Output Quality |
|---|---|
| temperature=0.1, no penalty | Severe repetition loops |
| temperature=1.5, no penalty | Incoherent random text |
| temperature=0.8, top_k=50, top_p=0.9, rep_penalty=1.2 | Coherent, diverse text |

The model itself has not changed — only the sampling strategy has. This means generation-time controls are not optional extras; they are essential for usable output.

### Lesson 5: Checkpointing Saves Everything

Training takes days. Power outages happen. Programs crash. Without checkpoints, all progress is lost. Our training script saves every 2,000 steps and tracks the best model separately. This saved us multiple times when training was interrupted.

### Lesson 6: Small Models Can Produce Surprisingly Good Text

Our 118M parameter model — running on a laptop GPU — produces text that reads like real news articles. You do not need a 175-billion-parameter model to demonstrate the fundamental capabilities of language modeling. Small models are perfect for learning, experimenting, and understanding how the technology works.

### Lesson 7: The Training-Inference Gap

The model performs differently during training (where it sees the correct next token) vs inference (where it must use its own predictions). This is why a model with good training loss can still produce bad generated text — errors during generation compound because each wrong token becomes context for the next prediction.

### Lesson 8: Modern Architecture Matters

Using GQA, RoPE, RMSNorm, and SwiGLU instead of the original Transformer components gives us better quality and efficiency. These are not just academic improvements — they make a real difference in training stability and model quality on consumer hardware.

## What We Would Do Differently

If we started this project over:

1. **Stricter data filtering from day one** — would have prevented the non-English contamination
2. **Implement evaluation metrics earlier** — would have caught repetition problems sooner
3. **Use a larger context window** (512 or 1024 tokens) — would improve coherence at the cost of memory
4. **Add label smoothing to the loss function** — would reduce overconfidence during training
5. **Log more training statistics** — gradient norms, learning rate, and token-level entropy at each checkpoint
