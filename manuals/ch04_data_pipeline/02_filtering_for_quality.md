# Chapter 4.2 — Filtering for Quality

## Why Filter?

Not all web text is good. The internet contains spam, machine-generated text, duplicate content, non-English pages, and low-quality writing. If we train on this junk, the model learns junk.

Our `prepare_data.py` script applies multiple quality filters before any text enters the training data.

## The Filters We Use

### 1. Document Length

| Filter | Threshold | Why |
|---|---|---|
| Minimum characters | 200 | Very short documents are usually boilerplate, navigation menus, or error pages |
| Maximum characters | 50,000 | Extremely long documents are often data dumps, not readable text |
| Minimum words | 50 | Documents with fewer than 50 words rarely contain useful language patterns |

### 2. Character Composition

| Filter | Threshold | What It Catches |
|---|---|---|
| Minimum alphabetic ratio | 55% | Rejects pages full of numbers, symbols, or code |
| Minimum ASCII alpha ratio | 85% | Rejects pages with too many non-English characters |
| Maximum digit ratio | 20% | Rejects number-heavy pages (tables, statistics) |
| Maximum non-ASCII ratio | 20% | Rejects pages with heavy use of non-Latin characters |

### 3. English Language Check

We check if a document is actually English using two methods:

**Method 1: Language column** — If the parquet file has a language column (e.g., "en"), we use it directly.

**Method 2: Stopword heuristic** — We count how many English function words ("the", "and", "that", "have", "for", "not", etc.) appear. English text naturally has at least 2% of its words as stopwords. Non-English text has very few English stopwords.

### 4. Repetition Check

| Filter | Threshold | What It Catches |
|---|---|---|
| Maximum line repeat ratio | 30% | Rejects documents where more than 30% of lines are duplicates |
| Character run check | 9+ identical chars | Rejects "aaaaaaaaa" spam patterns |
| Maximum URL count | 10 | Rejects pages that are mostly links |

## How Much Gets Filtered?

During data preparation, the filter logs how many documents were kept vs rejected:

```
Filtering summary: kept 2,341,567 docs | rejected 845,231 docs
```

Roughly 26% of documents get rejected. Most rejections are due to:
- Too short (navigation pages, error pages)
- Non-English (despite OpenWebText's own filtering, some slip through)
- High repetition (copy-paste content, auto-generated pages)

## The Non-English Problem

Despite these filters, some non-English text still leaks through. This is documented in detail in Chapter 9 (Mistakes and Lessons). The main culprit: documents that are mostly English but contain non-English names, quotes, or sections.

For example, a news article about Middle Eastern politics might be 90% English but contain Arabic names and transliterated words. Our filters pass it because the overall ASCII ratio is above 85%, but the model memorizes the non-English fragments.

This is what produced the "ibn nimy" degeneration pattern we observed. Solutions are discussed in Chapter 9.
