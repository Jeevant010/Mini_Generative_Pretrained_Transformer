# Chapter 1.4 — What You Will Learn

## Chapter Roadmap

This book is organized so you can read it front-to-back like a story. Each chapter builds on the previous one. Here is what each chapter covers:

### Chapter 1: What Is This? (You Are Here)

What language models are, what GPT means, and why we built our own.

### Chapter 2: How Text Becomes Numbers

Computers do not understand words — they understand numbers. Before we can train a model, we need to convert text into numbers. This chapter explains tokenization: how "Hello, world!" becomes `[15496, 11, 995, 0]`.

### Chapter 3: The Transformer

The Transformer is the architecture (design) behind our model. This chapter explains each component in plain English:

- How the model looks at previous words to predict the next one (attention)
- How the model knows word order (positional encoding)
- How information flows through the model (layers, blocks)
- How many numbers the model has and what they do (parameters)

### Chapter 4: The Data Pipeline

Where does the training data come from? This chapter explains:

- What OpenWebText is (the dataset we use)
- How we filter out bad or non-English content
- How we tokenize billions of words efficiently
- How we store data on disk without running out of memory

### Chapter 5: Training

This chapter explains what happens when we press "start training":

- What "loss" means and how the model learns from mistakes
- How the learning rate controls how fast the model learns
- How we save progress (checkpoints)
- How mixed precision makes training faster
- What hardware we used and why

### Chapter 6: Watching Training Happen

The most exciting chapter. This shows you **exactly** what the model produced at each stage of training, with real outputs from our actual training run:

- Step 0: Complete gibberish
- Step 2,000: Words that almost make sense
- Step 20,000: Readable paragraphs with topic drift
- Step 60,000: Coherent, fluent English
- Step 100,000+: High quality outputs from the log samples

### Chapter 7: Measuring Quality

How do we prove, mathematically, that the model is getting better? This chapter explains:

- Perplexity (how "confused" the model is)
- Diversity metrics (is the model repetitive?)
- Repetition detection (does the model get stuck in loops?)
- Benchmarks (how does our model compare to published models?)

### Chapter 8: Ablation Studies

What happens when we remove one component from the model?

- Remove normalization → training becomes unstable
- Remove positional encoding → the model loses grammar
- Remove Flash Attention → training gets slower but quality stays the same

These experiments prove which components are essential and which are optimizations.

### Chapter 9: Mistakes and Lessons

Everything that went wrong during this project, and what we learned:

- The non-English gibberish problem ("ibn nimy ibn nimy...")
- The repetition loop problem
- Signs that training has gone too far (overfitting)
- The learning rate decay trap
- Data quality issues
- Key takeaways for anyone building their own model

### Chapter 10: What Comes Next

Where this project goes from here:

- How to turn a word-predictor into a chatbot (SFT — Supervised Fine-Tuning)
- How to make the chatbot prefer good answers over bad ones (DPO — Direct Preference Optimization)
- The branch strategy for the next phase of development

### Appendices

Reference material:

- Glossary of terms
- Complete codebase map
- Configuration reference
- All hyperparameter presets
- Academic references

### Advanced (Future Work)

Detailed technical guides for post-training work (SFT, DPO, chat templates, evaluation harness). These will be used on the next branch after pre-training is pushed.

## How to Read This Book

**If you are a complete beginner:** Read chapters 1 through 6 in order. Skip the math sections on the first read — every formula has a plain English explanation above it.

**If you know some Python but not ML:** Read everything in order. The math sections will make more sense to you.

**If you already know machine learning:** Jump to chapters 6, 7, and 9. That is where the real project results, metrics, and lessons are.

**If you are here for the code:** See the Appendices for the codebase map, then dive into the source code directly.
