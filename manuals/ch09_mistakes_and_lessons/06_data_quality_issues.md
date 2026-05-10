# Chapter 9.6 — Data Quality Issues

## The Hidden Problem

You can build a perfect Transformer architecture, use the best optimizer, tune your learning rate perfectly — and still get a bad model if your data is bad. Data quality is the single most impactful factor in language model performance.

## Issues We Found in OpenWebText

### Issue 1: Non-English Content

**Problem:** Despite the fastText language filter in OpenWebText's original pipeline, non-English content leaked through:
- Arabic transliterations in news articles about the Middle East
- Mixed-language pages (mostly English with foreign sections)
- Transliterated text using only ASCII characters (bypasses character filters)

**Impact:** The model memorized non-English token sequences, which surfaced as gibberish during generation.

**Our fix:** Character-based heuristic filters (MIN_ASCII_ALPHA_RATIO, MAX_NON_ASCII_CHAR_RATIO, stopword check). Effective but not perfect.

**Recommended improvement:** Add fastText language detection at the paragraph level, not just document level.

### Issue 2: Repetitive Content

**Problem:** Some web pages contain large amounts of repeated text — navigation menus, copyright notices, template boilerplate.

**Impact:** The model learns these repetitive patterns and can reproduce them during generation, contributing to the repetition problem.

**Our fix:** MAX_LINE_REPEAT_RATIO = 0.30 — reject documents where more than 30% of lines are duplicates.

### Issue 3: Low-Quality Writing

**Problem:** Reddit-sourced URLs include blog posts, forum threads, and opinion pieces of varying quality. Not all web text is well-written.

**Impact:** The model's writing quality is limited by the average quality of its training data.

**Our fix:** MIN_WORD_COUNT = 50, MIN_ALPHA_CHAR_RATIO = 0.55. These remove very short or non-text pages, but do not judge writing quality.

### Issue 4: Topical Bias

**Problem:** OpenWebText is sourced from Reddit, which skews toward certain topics:
- Technology and gaming (very overrepresented)
- US politics (overrepresented)
- Sports (well represented)
- Academic topics (underrepresented)
- Non-Western topics (underrepresented)

**Impact:** The model is better at generating text about technology and US politics than about, say, South Asian history or molecular biology.

**No fix applied:** This is inherent to the dataset. A production model would use a more diverse data mixture.

## Data Quality Checklist

Before any training run, verify:

- [ ] Language filtering is on (`FILTER_TO_ENGLISH = True`)
- [ ] Quality filtering is on (`FILTER_FOR_QUALITY = True`)
- [ ] Non-ASCII ratio is reasonable (≤ 0.15 recommended)
- [ ] Stopword ratio check is active (≥ 0.02)
- [ ] Minimum document length is enforced (≥ 200 chars)
- [ ] Repetition check is active (≤ 0.30 line repeat ratio)
- [ ] Sample 100 random documents from the filtered set and read them manually

That last point — manually reading samples — is the most important quality check. Automated filters catch obvious problems, but only human review catches subtle issues.

## The Data Scaling Question

Do we need more data or better data?

**For our 118M model:** We have more than enough data. The Chinchilla scaling law recommends ~1.7B tokens for 85M parameters. We have 5.1B tokens — 3× the recommended amount. **Better data helps more than more data at our scale.**

**For a larger model (1B+ parameters):** More data would be needed. The current 5.1B tokens would only support a ~250M parameter model at Chinchilla-optimal ratios.

## Key Lesson

Data quality compounds: a 1% contamination rate in 5 billion tokens means 50 million bad tokens. The model will see these bad tokens thousands of times during training. Even a small amount of bad data can have an outsized effect on generation quality.

**Always invest in data quality before investing in model size.**
