# Chapter 6.1 — From Gibberish to English: The Training Story

## Overview

This chapter tells the story of our training run using **real outputs from the model at each stage**. Every text sample shown here is an actual output from our model, copied directly from the `logs/samples/` and `Prompt_Outputs/` directories.

Three fixed prompts were used throughout training to track progress:

1. "The future of artificial intelligence is"
2. "Once upon a time in a land far away"
3. "In the beginning, there was nothing but"

Let us watch the model evolve.

---

## Step 0 — Complete Random Noise

The model has just been initialized. All 118 million parameters are random numbers. It has never seen a single word of English.

> **Prompt:** "The future of artificial intelligence is"
>
> **Output:** "The future of artificial intelligence isrophic Doyle LIuuensual shattered rewards Doyle column LI 198 198 shrug cheapest Analysiszip Streuber number insistence tournament column billboard Planzip..."

> **Prompt:** "Once upon a time in a land far away"
>
> **Output:** "Once upon a time in a land far awayount dead roster uturenils averagescentury interruptRELtt calendar time prow gaining undergoing cleanson cuteBitcoinBitcoinactingosen gaining B manipulating bombed..."

> **Prompt:** "In the beginning, there was nothing but"
>
> **Output:** "In the beginning, there was nothing buthang mainWS persecutediguration ost conscience receptionhang sad Classicground Essex OS mentionedSandgage internalifting solutionsERS hier Step..."

**What we see:** Pure chaos. Words are mashed together without spaces. Token fragments like "zip", "REL", "Bitcoin" appear randomly. There is zero grammar, zero meaning, zero structure.

**Loss:** 10.54 | **Perplexity:** 37,780

---

## Step 2,000 — First Signs of English

After 2,000 steps, the model has seen about 15 million tokens. It is beginning to learn basic English patterns.

> **Prompt:** "The future of artificial intelligence is"
>
> **Output:** "The future of artificial intelligence is all the important. This is in the case of what's the world about, too. The only difference is that it is the best thing that these people are getting out of the world and some of the other. We are not very much surprised when it comes to the problem."

> **Prompt:** "Once upon a time in a land far away"
>
> **Output:** "Once upon a time in a land far away than the rise of an air (and its own). The same was always a big difference between the more major and more important things, but while the cost of our current population (because the majority is a whole, it was not necessarily the right of all kinds."

**What we see:** Massive improvement! Proper spaces between words. Real English words. Sentences that almost make grammatical sense. But the content is vague and incoherent — the model has learned the *shape* of English (words, spaces, punctuation, sentence structure) but not the *meaning*.

---

## Step 5,000 — Learning to Continue Text (from Prompt_Outputs)

> **Prompt:** "Hello, how are you?"
>
> **Output:** "Hello, how are you? I love getting something done with a doctor who is a good teacher and I am a member of a company. I love him and I am very proud for him. He is a great teacher and I will have a great child."

**What we see:** The model does not answer "I'm fine" — because it is a base model, not a chatbot. It just continues the text as if it were a web page. The grammar is mostly correct now, but the content wanders aimlessly.

---

## Step 20,000 — Topic Awareness Emerging

> **Prompt:** "The future of artificial intelligence is"
>
> **Output:** "The future of artificial intelligence is the only thing to truly appreciate and help us understand what is happening in the future. How to launch a business is to get an idea of the business, and how to improve it."

> **Prompt:** "Once upon a time in a land far away"
>
> **Output:** "Once upon a time in a land far away from the coast of China, China's economic policies were being rolled into the region. 'There have been issues there. We have not been successful. We have to be successful,' says Eid Jong, the director of the Chinese military agency."

**What we see:** The model now stays on topic better. When given "artificial intelligence," it talks about technology and the future. When given "land far away," it generates a geopolitical narrative. It is inventing plausible-sounding names ("Eid Jong") and using quotation marks correctly.

---

## Step 40,000 — Coherent Paragraphs

> **Prompt:** "The future of artificial intelligence is"
>
> **Output:** "The future of artificial intelligence is something like that, and it's what makes it a reality," says Dr. James Clark, a former U.S. government official and a professor of computer science, who is also a professor of computer science at the University of Manchester who studies artificial intelligence. "I think I can give you a glimpse at how people are using digital and artificial intelligence to solve human problems in a real world."

> **Prompt:** "how can i help" (from Prompt_Outputs)
>
> **Output:** "how can i help with it? I think that the biggest consideration is it's got a lot more depth and depth than an all-time great team. I'm really looking forward for the season. I think that the first few months will have a lot more depth and depth from the game."

**What we see:** The model generates fluent English with proper quotation marks, attribution ("says Dr. James Clark"), and institutional references ("University of Manchester"). The text reads like a real news article. However, the content is fabricated — there is no "Dr. James Clark" saying these exact words. This is **hallucination**, a known property of all language models.

---

## Step 60,000 — Peak Fluency

> **Prompt:** "The future of artificial intelligence is"
>
> **Output:** "The future of artificial intelligence is at stake for the future of the technology. Indeed, in this respect, it's unlikely to be a revolution in computing. There are many more questions to be raised about artificial intelligence. In the coming years, we will be looking at how AI can be applied and the implications of it."

> **Prompt:** "In the beginning, there was nothing but"
>
> **Output:** "In the beginning, there was nothing but pure power that had gone unchecked in the past — a combination of power and power. In the end, there was nothing but pure power and power (and power) that needed to be utilized."

**What we see:** Excellent fluency, good vocabulary. But notice the repetition in Prompt 3 — "power" appears 6 times in one paragraph. This is an early sign of a repetition problem that will worsen later.

**Loss:** 3.52 | **Perplexity:** 33.69

---

## Steps 80,000–122,000 — Continued Quality (Log Samples)

The `logs/samples/` directory shows the model continuing to produce good outputs at these later stages:

> **Step 80,000:** "The future of artificial intelligence is being explored and there is now a growing awareness that there are several challenges facing the world. In the new book, the University of Cambridge's Watson Institute, the technology industry has been in a battle with increasing technological power."

> **Step 100,000:** "The future of artificial intelligence is in the hands of a man. The next time you hear a voice calling for a computer to be integrated into the world's AI network, ask yourself: 'What about artificial intelligence?' In the future, your only hope is that, as the future of AI becomes more advanced, you will be able to develop your own AI software."

> **Step 122,000:** "The future of artificial intelligence is not as bleak as the prospects for future technology, but it is a possibility for the future of information technology," says Peter Woll, professor of robotics at the University of Washington."

**What we see:** The model maintains high quality even at 122K steps. These later log samples are actually among the best outputs — coherent, topical, and well-structured.

---

## The Degeneration Exception (Earlier Training Run)

On an earlier training configuration, the model produced degenerate output at 106K steps:

> **Prompt:** "The furture of ai"
>
> **Output:** "The furture of ai ibn nimy ibn nimy ibn ibn ibn ibn nimy ibn nimy ibn nimy ibn nimy ibn ibn nimy ibn nimy..."

This is the **degeneration problem** — the model got stuck in a repetition loop. This happened on a specific training run and is discussed in detail in Chapter 9. The key point: this is not inevitable. The current training run's log samples at 106K are perfectly fine:

> **Step 106,000 (log):** "The future of artificial intelligence is increasingly uncertain, but it will be at least in part determined by the human mind and not by the way humans work."

The difference between a healthy run and a degenerate one comes down to learning rate, data quality, and sampling strategy — all covered in Chapter 9.
