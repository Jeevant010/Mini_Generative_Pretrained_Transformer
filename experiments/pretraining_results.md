# Pre-training Results (Step 149,000)

## Quality Metrics
- **Validation Loss**: 3.3104
- **Perplexity (PPL)**: 27.40 (Excellent for 118M parameters)
- **Bits Per Character**: 4.78

## Diversity & Repetition
- **Distinct-1**: 0.6998 (> 0.5 is healthy)
- **Distinct-2**: 0.9117 (> 0.6 is healthy)
- **Self-BLEU**: 0.0000 (Extremely diverse outputs across samples)
- **Output Entropy**: 6.41 bits (Healthy confidence)

## Qualitative Samples
**Prompt**: "The future of artificial intelligence is"
**Output**: "The future of artificial intelligence is not certain and there is some reason to think that there are other things as well. When you are working in th..."

**Prompt**: "Once upon a time in a land far away"
**Output**: "Once upon a time in a land far away, many people from the east (like myself) were either unaware or unknowingly dying of a tragic accident."
