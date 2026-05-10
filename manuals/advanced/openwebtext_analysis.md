# OpenWebText Data Quality Analysis

## What OpenWebText Actually Contains

OpenWebText is an open-source recreation of the WebText dataset used to train GPT-2. Understanding its composition is essential because the training data directly determines what the model learns — and what it fails to learn.

### Source and Curation Pipeline

```
Reddit submissions (all URLs with >= 3 karma)
        ↓
    URL extraction and deduplication
        ↓
    Web page download (newspaper library)
        ↓
    fastText English language filter
        ↓
    LSH near-deduplication (5-gram Jaccard > 0.5 removed)
        ↓
    Minimum length filter (< 128 tokens removed)
        ↓
    OpenWebText corpus (~38 GB, ~8M documents)
```

### Corpus Statistics

| Metric | Value |
|---|---|
| Total documents | ~8,013,769 |
| Total size (raw text) | ~38 GB |
| Source | Web pages linked from Reddit |
| Language filter | fastText (English only) |
| Deduplication | LSH with 5-gram Jaccard similarity |
| Minimum length | 128 tokens per document |
| Average document length | ~4.7 KB |

### Content Distribution

Because the data comes from Reddit, the corpus is biased toward:

| Content Type | Prevalence | Quality |
|---|---|---|
| News articles | High | Generally good grammar and structure |
| Blog posts | High | Variable quality, opinion-heavy |
| Forum discussions | Medium | Informal, may contain slang |
| Wikipedia-style articles | Low (excluded) | Not present |
| Technical/academic | Low | Limited depth |
| Fiction/creative | Low | Minimal |
| Non-English pages | Low (filtered) | Should be zero, but leaks through |

## The Non-English Contamination Problem

### Why Non-English Text Leaks Through

Despite fastText language filtering, some non-English content makes it into the corpus:

1. **Mixed-language documents**: Articles that are mostly English but contain foreign-language sections (names, quotes, titles)
2. **Code-switching**: Documents that alternate between English and another language
3. **Transliterated text**: Non-English words written in Latin characters that fastText classifies as English
4. **Short non-English fragments**: Below the detection threshold of the language model
5. **Misclassified languages**: fastText has error rates, especially for related languages (Dutch, Afrikaans, etc.)

### Impact on the Model

When the model encounters non-English n-grams during training, it learns them as valid sequences. During generation, if the model's confidence is high (late training, low entropy), these memorized non-English patterns can surface as gibberish:

- `"ibn nimy ibn nimy"` — appears to be Arabic transliterations memorized from the training data
- `"ipsamas"` — possibly a memorized non-English word or malformed token

### Quantifying the Problem

The project's `prepare_data.py` already includes quality filtering:

```python
# Current filtering thresholds in config.py
FILTER_TO_ENGLISH = True
MIN_ASCII_ALPHA_RATIO = 0.85     # At least 85% of alpha chars must be ASCII
MAX_NON_ASCII_CHAR_RATIO = 0.20  # At most 20% non-ASCII characters
MIN_ENGLISH_STOPWORD_RATIO = 0.02 # At least 2% English stopwords
```

These heuristics catch many non-English documents but miss:
- Documents that are 80% English with 20% non-English segments
- Transliterated foreign text that uses only ASCII characters
- Documents with high English stopword frequency but non-English body text

### Strengthening the Language Filter

#### Option 1: Use fastText Language Detection in the Pipeline

Install the `fasttext` library and add a language detection step to `prepare_data.py`:

```python
import fasttext

# Download the language identification model:
# https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
LANG_MODEL = fasttext.load_model("lid.176.bin")

def is_english_fasttext(text, threshold=0.8):
    """
    Use fastText to check if text is English.
    
    Args:
        text: Document text.
        threshold: Minimum confidence for English classification.
    
    Returns:
        True if document is likely English.
    """
    # fastText requires single-line input
    clean_text = text.replace("\n", " ")[:5000]  # Limit length
    predictions = LANG_MODEL.predict(clean_text, k=1)
    label, confidence = predictions[0][0], predictions[1][0]
    
    # fastText labels are like "__label__en"
    detected_lang = label.replace("__label__", "")
    return detected_lang == "en" and confidence >= threshold
```

#### Option 2: Paragraph-Level Filtering

Instead of checking the entire document, check each paragraph independently:

```python
def filter_non_english_paragraphs(text, threshold=0.7):
    """
    Remove paragraphs that are not English.
    Keeps the document structure but removes foreign-language sections.
    """
    paragraphs = text.split("\n\n")
    english_paragraphs = []
    
    for para in paragraphs:
        if len(para.strip()) < 20:
            english_paragraphs.append(para)  # Keep short separators
            continue
        if is_english_fasttext(para, threshold):
            english_paragraphs.append(para)
    
    filtered = "\n\n".join(english_paragraphs)
    
    # Only keep if at least 50% of content survived filtering
    if len(filtered) < len(text) * 0.5:
        return None
    return filtered
```

#### Option 3: Raise Existing Thresholds

A simpler approach that requires no new dependencies:

```python
# Stricter config values
MIN_ASCII_ALPHA_RATIO = 0.92   # Was 0.85 — stricter ASCII requirement
MAX_NON_ASCII_CHAR_RATIO = 0.08 # Was 0.20 — reject docs with >8% non-ASCII
MIN_ENGLISH_STOPWORD_RATIO = 0.05 # Was 0.02 — require more English function words
```

This will reject more borderline documents but may also filter some valid English text with names, technical terms, or URLs containing non-ASCII characters.

## Scaling: More Data vs Better Data

### Chinchilla Scaling Laws

The Chinchilla paper (Hoffmann et al., 2022) established that compute-optimal training requires approximately **20 tokens per parameter**:

$$
N_{\text{tokens, optimal}} \approx 20 \times N_{\text{params}}
$$

For the 85M parameter model:

$$
N_{\text{tokens, optimal}} = 20 \times 85{,}000{,}000 = 1.7 \text{ billion tokens}
$$

### Current Data Situation

| Metric | Value |
|---|---|
| Model parameters | ~85M |
| Chinchilla-optimal tokens | 1.7B |
| train.bin tokens | 5.1B |
| val.bin tokens | 0.27B |
| Total available tokens | 5.37B |
| Tokens per parameter ratio | 63× (over Chinchilla optimal) |

The model has **more than enough data** from a Chinchilla perspective. The issue is not data quantity but potentially:

1. **Data quality** — non-English contamination, low-quality documents
2. **Multi-epoch exposure** — the model sees the same data multiple times at 100K+ steps
3. **Diversity** — OpenWebText is Reddit-biased and may lack domain diversity

### Tokens Seen During Training

At each training step, the model processes:

$$
\text{tokens per step} = \text{batch\_size} \times \text{block\_size} = 20 \times 384 = 7{,}680
$$

Total tokens seen after $N$ steps:

$$
\text{total tokens seen} = N \times 7{,}680
$$

| Checkpoint | Steps | Tokens Seen | Epochs Over 5.1B Train Tokens |
|---|---|---|---|
| 5,000 | 5K | 38.4M | 0.008 |
| 40,000 | 40K | 307.2M | 0.060 |
| 100,000 | 100K | 768.0M | 0.151 |
| 150,000 | 150K | 1.15B | 0.226 |

At 100K steps the model has seen 768M tokens — less than one full pass through the 5.1B-token training set. This means **the model is not running out of data**. The degeneration at late steps is likely due to learning rate decay and model confidence, not data exhaustion.

### When More Data Helps

More data would help if:
- You increase model size beyond 85M parameters
- You want to train for longer than 150K steps
- You observe significant train-val loss gap (overfitting to training data)

### When Better Data Helps More

Better data helps when:
- The model generates non-English fragments (→ improve language filtering)
- The model produces low-quality or incoherent text (→ filter for higher quality)
- The model echoes specific data patterns (→ improve deduplication)

### Recommendations for This Project

1. **Keep the 10GB data target** — it is sufficient for an 85M model
2. **Improve filtering** — add fastText language detection to `prepare_data.py`
3. **Raise quality thresholds** — tighten the existing heuristic filters
4. **Monitor data exposure** — log how many unique documents the model has seen
5. **Consider supplementary data** — if increasing model size, add data from:
   - Books (Project Gutenberg, BookCorpus)
   - Wikipedia dumps
   - Curated web datasets (RefinedWeb, DCLM)

## Data Quality Checklist

Before starting a new training run, verify:

- [ ] Language filtering is active (`FILTER_TO_ENGLISH = True`)
- [ ] Quality heuristics are active (`FILTER_FOR_QUALITY = True`)
- [ ] Non-ASCII ratio threshold is reasonable (`MAX_NON_ASCII_CHAR_RATIO <= 0.15`)
- [ ] English stopword check is active (`MIN_ENGLISH_STOPWORD_RATIO >= 0.02`)
- [ ] Documents are long enough (`MIN_DOC_CHARS >= 200`)
- [ ] Near-duplicates have been removed (check unique document count)
- [ ] Training data size matches model size (20× tokens per parameter minimum)

## References

- Gokaslan, A. & Cohen, V. (2019). "OpenWebText Corpus."
- Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla)
- Penedo et al. (2023). "The RefinedWeb Dataset for Falcon LLM." (Advanced web data filtering)
- Li et al. (2024). "DataComp-LM: In Search of the Next Generation of Training Sets for Language Models." (DCLM)
- Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." (Data mixture strategies)
- Zhao et al. (2023). "A Survey of Large Language Models." (Comprehensive data pipeline overview)
