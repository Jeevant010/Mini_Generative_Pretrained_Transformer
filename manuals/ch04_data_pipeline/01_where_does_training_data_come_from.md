# Chapter 4.1 — Where Does Training Data Come From?

## The Principle

Language models learn by reading text. The more text they read, and the higher the quality of that text, the better they learn. Our model reads web text — specifically, web pages that real humans shared and upvoted on Reddit.

## Why Web Text?

The internet contains the largest collection of human-written text ever assembled. It covers every topic imaginable — science, history, fiction, news, opinions, conversations, code, recipes, legal documents, academic papers, and much more.

By training on web text, the model learns the full breadth of human language, not just one narrow domain.

## Why Not Just Use Books?

Books are high quality, but they have limitations:

- Limited diversity of writing styles (formal prose)
- Smaller total volume compared to the internet
- Often restricted by copyright

Web text provides:

- Diverse writing styles (formal, informal, technical, conversational)
- Massive volume (terabytes of text)
- Current events and modern language

In practice, the best language models use a mixture: web text + books + Wikipedia + code. Our project uses web text only, which is the core of most training mixtures.

## The OpenWebText Dataset

Our training data comes from **OpenWebText** — an open-source recreation of the dataset used to train GPT-2.

### How OpenWebText Was Created

1. **Source**: All URLs shared on Reddit that received at least 3 upvotes (karma)
2. **Download**: Each URL was downloaded and the main text was extracted
3. **Language filter**: Facebook's fastText language model filtered out non-English pages
4. **Deduplication**: Near-duplicate documents were removed using a technique called LSH (Locality-Sensitive Hashing)
5. **Length filter**: Documents shorter than 128 tokens were removed

The Reddit upvote requirement is clever — it uses millions of Reddit users as a distributed quality filter. If a link got upvoted, at least some people found it worth reading.

### OpenWebText Statistics

| Property | Value |
|---|---|
| Total documents | ~8 million |
| Total size (raw text) | ~38 GB |
| Source | Reddit-shared URLs |
| Language | English (fastText filtered) |
| Quality signal | Reddit karma ≥ 3 |

## Our Subset

We use a 10 GB subset of OpenWebText, not the full 38 GB. This is intentional:

- 10 GB is enough to train an 85M parameter model well (the Chinchilla scaling law says we need ~1.7 billion tokens; we have 5.1 billion)
- Smaller data means faster preparation and iteration
- We can always scale up later using the `full_dataset_60gb` preset in config

## What the Data Actually Contains

Since the data comes from Reddit links, it is biased toward certain content types:

| Content Type | How Common | Quality |
|---|---|---|
| News articles | Very common | Generally well-written |
| Blog posts | Very common | Variable quality |
| Reddit discussions | Common | Informal, slang |
| Technical articles | Medium | Usually good |
| Opinion pieces | Medium | Variable |
| Fiction/creative writing | Rare | Limited |
| Academic papers | Rare | Usually summarized, not full papers |
| Code/programming | Occasional | Mixed with text |

This composition affects what the model learns. Since news articles are very common, the model gets good at writing in a news-article style. Since fiction is rare, the model is weaker at creative writing.
