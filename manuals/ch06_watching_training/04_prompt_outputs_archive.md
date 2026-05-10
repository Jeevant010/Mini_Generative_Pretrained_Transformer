# Chapter 6.4 — Prompt Outputs Archive

## What This Chapter Contains

This is a record of actual model outputs saved at different training milestones. These come from two sources:

1. **`logs/samples/`** — Automatically generated every 2,000 steps during training using three fixed prompts
2. **`Prompt_Outputs/`** — Manually generated and saved at key milestones with various prompts

Both old and new outputs are preserved here as a historical record of the model's evolution.

---

## Automatic Log Samples (logs/samples/)

These were generated automatically during training with fixed prompts:
- Prompt 1: "The future of artificial intelligence is"
- Prompt 2: "Once upon a time in a land far away"
- Prompt 3: "In the beginning, there was nothing but"

### Sample at Step 0
> **P1:** "...isrophic Doyle LIuuensual shattered rewards Doyle column LI 198 198 shrug cheapest..."
>
> **P2:** "...ount dead roster uturenils averagescentury interruptRELtt calendar time prow gaining..."
>
> **P3:** "...hang mainWS persecutediguration ost conscience receptionhang sad Classicground Essex..."

### Sample at Step 10,000
Readable English with basic grammar but incoherent topics.

### Sample at Step 40,000
Fluent prose with attribution: "says Dr. James Clark, a professor of computer science at the University of Manchester"

### Sample at Step 60,000
Peak fluency on prompt 1. Some repetition on prompt 3 ("power" repeated 6×).

### Sample at Step 80,000
Coherent: "The future of artificial intelligence is being explored and there is now a growing awareness that there are several challenges facing the world."

### Sample at Step 100,000
High quality: "The future of artificial intelligence is in the hands of a man. The next time you hear a voice calling for a computer to be integrated into the world's AI network..."

### Sample at Step 106,000
Good quality in logs: "The future of artificial intelligence is increasingly uncertain, but it will be at least in part determined by the human mind..."

### Sample at Step 122,000 (Latest)
"The future of artificial intelligence is not as bleak as the prospects for future technology, but it is a possibility for the future of information technology," says Peter Woll, professor of robotics at the University of Washington."

---

## Manual Prompt Outputs (Prompt_Outputs/)

These were manually tested with different prompts.

### Iteration 5,000 (2.3 hours of training)

> **Prompt:** "Hello, how are you?"
>
> **Output:** "Hello, how are you? I love getting something done with a doctor who is a good teacher and I am a member of a company. I love him and I am very proud for him."

Note: Model does not answer the question — it continues text as a web document would.

### Iteration 10,000 (2.11 hours)
Basic English, coherent sentences but unfocused topics.

### Iteration 16,000 (1.37 hours)
Improving grammar, starting to maintain topic for 2-3 sentences.

### Iteration 22,000 (2.08 hours)
Generating plausible news-style content.

### Iteration 30,000 (2.1 hours)
Good coherence, proper punctuation, quotes used correctly.

### Iteration 40,000 (4 hours)

> **Prompt:** "how can i help"
>
> **Output:** "how can i help with it? I think that the biggest consideration is it's got a lot more depth and depth than an all-time great team. I'm really looking forward for the season."

### Iteration 46,000 (3 hours)
Continued improvement in topical consistency.

### Iteration 60,000 (5 hours)
Near-peak quality for base model pre-training.

### Iteration 89,000 (7 hours)
Maintained quality with diverse outputs.

### Iteration 100,000 (5 hours)
Strong quality across all prompts.

### Iteration 106,000 (3 hours) — THE DEGENERATION EXAMPLE

> **Prompt:** "The furture of ai"
>
> **Output:** "The furture of ai ibn nimy ibn nimy ibn ibn ibn ibn nimy ibn nimy ibn nimy..."

This is the degenerate output from an earlier training configuration. The current training run's log samples at 106,000 are healthy — see logs/samples/step_106000.txt.

---

## How to Read These Outputs

| Stage | What to Look For |
|---|---|
| Step 0 | Random tokens, no English structure |
| Steps 1K-5K | Real English words appearing, basic grammar |
| Steps 10K-30K | Sentences forming, topic starting to emerge |
| Steps 40K-60K | Coherent paragraphs, proper attribution, quotes |
| Steps 80K+ | High quality prose, fabricated but plausible names and facts |
| Degeneration | Repetitive loops, non-English fragments |

## Note on Output Tracking

Both `logs/samples/` and `Prompt_Outputs/` should be preserved in the repository as a historical record. When newer training runs produce different outputs, keep the old outputs — they document the evolution of the model and help identify when quality changes.
