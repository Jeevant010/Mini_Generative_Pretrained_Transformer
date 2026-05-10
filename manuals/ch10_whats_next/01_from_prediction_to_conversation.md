# Chapter 10.1 — From Prediction to Conversation

## Where We Are

We have a trained base language model. It predicts the next word given previous words. When you give it a prompt, it continues the text as if it were writing a web article. It does **not** answer questions, follow instructions, or have conversations.

This is exactly what GPT-2 was. And this is where the real-world AI models we use every day — ChatGPT, Claude, Gemini — were at one point during their development.

## The Three Stages After Pre-Training

To go from "web text predictor" to "helpful assistant," the model needs three more stages:

### Stage 1: Supervised Fine-Tuning (SFT)

Show the model thousands of examples of conversations:

```
User: What is the capital of France?
Assistant: The capital of France is Paris.
```

```
User: Write a poem about rain.
Assistant: Gentle drops upon the glass,
           A rhythm soft and slow...
```

By training on these examples, the model learns: "When text looks like a question, I should generate an answer, not continue a web article."

**Result:** The model becomes an instruction-follower. It can answer questions and follow basic commands, but its answers may be mediocre or harmful.

### Stage 2: Preference Alignment (DPO/RLHF)

Show the model pairs of responses where one is better than the other:

```
User: How do I make coffee?
Good response: "Boil water, add ground coffee to a filter, pour water through..."
Bad response: "Coffee is made from beans that grow on trees..."
```

The model learns to prefer responses that humans rated as better.

**Result:** The model produces higher-quality, more helpful responses. It avoids harmful content and stays on topic.

### Stage 3: Chat Template

Add special tokens so the model knows when the user is speaking, when it should respond, and when a response ends:

```
<|user|>What is 2+2?<|end|>
<|assistant|>2 + 2 = 4.<|end|>
```

**Result:** The model can handle multi-turn conversations with clear role boundaries.

## Why We Are Not Doing This Yet

These stages require:
1. Different training data (instruction datasets, not web text)
2. Different training objectives (not just next-word prediction)
3. Changes to the tokenizer (new special tokens)
4. Careful evaluation (quality of responses, safety, helpfulness)

All of this is documented in detail in the `advanced/` folder, ready for the next branch. The pre-training phase — which this book covers — is the foundation that everything else builds on.

## The Roadmap

```
[DONE] Pre-training on OpenWebText
   ↓   (current branch — push and close)
[NEXT] Supervised Fine-Tuning with Dolly 15K
   ↓   (next branch: post-training)
[THEN] DPO Preference Alignment
   ↓
[THEN] Chat Template + Deployment
   ↓
[GOAL] A conversational AI running on a laptop
```

## What Changes in Code

| Stage | Files Modified | What Changes |
|---|---|---|
| SFT | `training.py`, `dataset.py`, tokenizer | New dataset loader, loss masking for assistant-only tokens |
| DPO | New `dpo_training.py` | New loss function comparing preferred/rejected responses |
| Chat | `tokenizer.py`, `model.py`, new `chat.py` | New special tokens, embedding resize, chat interface |

All implementation details are documented in:
- `advanced/02_supervised_fine_tuning.md`
- `advanced/03_dpo_preference_alignment.md`
- `advanced/04_chat_templates_and_deployment.md`
