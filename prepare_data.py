import glob
import os
import random
import re

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

import config
from tokenizer import BytePairTokenizer

# --- Configuration ---
DATASET_PATH = config.DATASET_PATH
TRAIN_BIN = config.TRAIN_BIN
VAL_BIN = config.VAL_BIN
TOKENIZER_PATH = config.TOKENIZER_PATH
VAL_PERCENT = getattr(config, "VAL_PERCENT", 0.05)
SAMPLE_SIZE_MB = getattr(config, "TOKENIZER_SAMPLE_SIZE_MB", 100)
TARGET_SIZE_GB = getattr(config, "DATASET_TARGET_SIZE_GB", None)
PREP_RANDOM_SEED = getattr(config, "PREP_RANDOM_SEED", 1337)
TOKENIZATION_BATCH_SIZE = getattr(config, "TOKENIZATION_BATCH_SIZE", 128)
SKIP_FULL_ROW_COUNT_SCAN = getattr(config, "SKIP_FULL_ROW_COUNT_SCAN", True)
SHUFFLE_PARQUET_FILES = getattr(config, "SHUFFLE_PARQUET_FILES", True)
MAX_PARQUET_FILES = getattr(config, "MAX_PARQUET_FILES", None)

TEXT_COLUMN_CANDIDATES = ("text", "content", "document", "body")
LANGUAGE_COLUMN_CANDIDATES = ("language", "lang", "language_code")
QUALITY_COLUMN_CANDIDATES = ("quality_score", "score", "quality", "rank", "rating")

ENGLISH_STOPWORDS = {
    "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
    "his", "from", "they", "say", "her", "she", "will", "one", "all", "would",
    "there", "their", "what", "about", "which", "when", "make", "can", "like",
    "time", "just", "know", "take", "people", "into", "year", "your", "good",
    "some", "could", "them", "see", "other", "than", "then", "now", "look",
    "only", "come", "its", "over", "think", "also", "back", "after", "use",
    "two", "how", "our", "work", "first", "well", "way", "even", "new",
}
WORD_RE = re.compile(r"[A-Za-z']+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
CHAR_RUN_RE = re.compile(r"(.)\1{8,}")


def print_stage(title):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")


def sample_text_subset(dataset, target_size_mb):
    current_size = 0
    sampled_text = []
    print(f"Sampling {target_size_mb}MB for tokenizer training...")
    for doc in dataset:
        text = doc["text"]
        sampled_text.append(text)
        current_size += len(text.encode("utf-8"))
        if current_size >= target_size_mb * 1024 * 1024:
            break
    return sampled_text


def get_parquet_files(dataset_path):
    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_path}")
    if SHUFFLE_PARQUET_FILES:
        random.shuffle(parquet_files)
    if MAX_PARQUET_FILES is not None:
        parquet_files = parquet_files[:MAX_PARQUET_FILES]
    return parquet_files


def detect_column(parquet_path, candidates, required=False):
    schema_names = pq.ParquetFile(parquet_path).schema.names
    for candidate in candidates:
        if candidate in schema_names:
            return candidate

    if required:
        raise KeyError(
            f"Could not find any of {candidates} in {parquet_path}. "
            f"Available columns: {', '.join(schema_names)}"
        )
    return None


def parse_quality_score(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def english_heuristic_passes(text):
    char_count = len(text)
    if char_count < getattr(config, "MIN_DOC_CHARS", 200):
        return False
    if char_count > getattr(config, "MAX_DOC_CHARS", 50000):
        return False

    words = WORD_RE.findall(text.lower())
    if len(words) < getattr(config, "MIN_WORD_COUNT", 50):
        return False

    alpha_chars = sum(ch.isalpha() for ch in text)
    ascii_alpha_chars = sum(("a" <= ch.lower() <= "z") for ch in text if ch.isalpha())
    digit_chars = sum(ch.isdigit() for ch in text)
    non_ascii_chars = sum(ord(ch) > 127 for ch in text)

    alpha_ratio = alpha_chars / char_count
    digit_ratio = digit_chars / char_count
    non_ascii_ratio = non_ascii_chars / char_count
    ascii_alpha_ratio = ascii_alpha_chars / max(alpha_chars, 1)

    if alpha_ratio < getattr(config, "MIN_ALPHA_CHAR_RATIO", 0.55):
        return False
    if ascii_alpha_ratio < getattr(config, "MIN_ASCII_ALPHA_RATIO", 0.85):
        return False
    if digit_ratio > getattr(config, "MAX_DIGIT_CHAR_RATIO", 0.20):
        return False
    if non_ascii_ratio > getattr(config, "MAX_NON_ASCII_CHAR_RATIO", 0.20):
        return False

    stopword_hits = sum(word in ENGLISH_STOPWORDS for word in words)
    stopword_ratio = stopword_hits / max(len(words), 1)
    if stopword_ratio < getattr(config, "MIN_ENGLISH_STOPWORD_RATIO", 0.02):
        return False

    if len(URL_RE.findall(text)) > getattr(config, "MAX_URL_COUNT", 10):
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 5:
        unique_line_ratio = len(set(lines)) / len(lines)
        repeated_line_ratio = 1.0 - unique_line_ratio
        if repeated_line_ratio > getattr(config, "MAX_LINE_REPEAT_RATIO", 0.30):
            return False

    if CHAR_RUN_RE.search(text):
        return False

    return True


def document_passes_filters(doc):
    text = doc["text"]
    needs_quality_check = getattr(config, "FILTER_FOR_QUALITY", True)

    if getattr(config, "FILTER_TO_ENGLISH", True):
        language_value = doc.get("language")
        if language_value is not None:
            if str(language_value).strip().lower() not in {"en", "eng", "english"}:
                return False
            if needs_quality_check and not english_heuristic_passes(text):
                return False
        elif not english_heuristic_passes(text):
            return False
    elif needs_quality_check and not english_heuristic_passes(text):
        return False

    min_quality_score = getattr(config, "MIN_QUALITY_SCORE", None)
    quality_score = doc.get("quality_score")
    if min_quality_score is not None and quality_score is not None and quality_score < min_quality_score:
        return False

    return True


def iter_parquet_documents(parquet_files, text_column, language_column=None, quality_column=None):
    selected_columns = [text_column]
    if language_column and language_column not in selected_columns:
        selected_columns.append(language_column)
    if quality_column and quality_column not in selected_columns:
        selected_columns.append(quality_column)

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=selected_columns)
            text_values = table.column(text_column).to_pylist()
            language_values = (
                table.column(language_column).to_pylist()
                if language_column else [None] * len(text_values)
            )
            quality_values = (
                table.column(quality_column).to_pylist()
                if quality_column else [None] * len(text_values)
            )

            for text_value, language_value, quality_value in zip(text_values, language_values, quality_values):
                if text_value is None:
                    continue

                if isinstance(text_value, bytes):
                    text_value = text_value.decode("utf-8", errors="ignore")
                else:
                    text_value = str(text_value)

                if not text_value:
                    continue

                yield {
                    "text": text_value,
                    "language": language_value,
                    "quality_score": parse_quality_score(quality_value),
                }


def iter_filtered_documents(parquet_files, text_column, language_column=None, quality_column=None):
    kept_docs = 0
    rejected_docs = 0

    for doc in iter_parquet_documents(parquet_files, text_column, language_column, quality_column):
        if document_passes_filters(doc):
            kept_docs += 1
            yield doc
        else:
            rejected_docs += 1

    print(f"Filtering summary: kept {kept_docs:,} docs | rejected {rejected_docs:,} docs")


def count_parquet_rows(parquet_files):
    total_rows = 0
    for parquet_path in parquet_files:
        total_rows += pq.ParquetFile(parquet_path).metadata.num_rows
    return total_rows


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare():
    random.seed(PREP_RANDOM_SEED)

    print_stage("STEP 1/3 - Scanning Parquet Dataset")
    print(f"Loading Parquet files from {DATASET_PATH}...")
    parquet_files = get_parquet_files(DATASET_PATH)
    text_column = detect_column(parquet_files[0], TEXT_COLUMN_CANDIDATES, required=True)
    language_column = detect_column(parquet_files[0], LANGUAGE_COLUMN_CANDIDATES, required=False)
    quality_column = detect_column(parquet_files[0], QUALITY_COLUMN_CANDIDATES, required=False)
    total_docs = None if SKIP_FULL_ROW_COUNT_SCAN else count_parquet_rows(parquet_files)

    if total_docs is None:
        print(
            f"Found {len(parquet_files)} parquet shards using text column '{text_column}'. "
            "Skipping full row count scan for faster startup."
        )
    else:
        print(
            f"Found {len(parquet_files)} parquet shards, about {total_docs:,} rows, "
            f"using text column '{text_column}'."
        )
    print(
        f"Optional metadata: language={language_column or 'none'} | "
        f"quality={quality_column or 'none'}"
    )
    if TARGET_SIZE_GB is not None:
        print(f"Target output size: {TARGET_SIZE_GB:.2f} GB across train+val binaries")
    print(f"Tokenization batch size: {TOKENIZATION_BATCH_SIZE}")
    if MAX_PARQUET_FILES is not None:
        print(f"Using only the first {len(parquet_files)} parquet shards after shuffling")

    print_stage("STEP 2/3 - Preparing Tokenizer")
    if not os.path.exists(TOKENIZER_PATH):
        training_docs = sample_text_subset(
            iter_filtered_documents(parquet_files, text_column, language_column, quality_column),
            SAMPLE_SIZE_MB,
        )
        tokenizer = BytePairTokenizer()
        tokenizer.train(training_docs, vocab_size=32000)
        tokenizer.save(TOKENIZER_PATH)
        print(f"Tokenizer saved to {TOKENIZER_PATH}")
    else:
        tokenizer = BytePairTokenizer.load(TOKENIZER_PATH)
        print("Loaded existing tokenizer.")

    eos_id = tokenizer.special_to_id["<eos>"]
    train_token_count = 0
    val_token_count = 0
    target_total_bytes = None

    if TARGET_SIZE_GB is not None:
        target_total_bytes = int(TARGET_SIZE_GB * 1024 * 1024 * 1024)

    print_stage("STEP 3/3 - Tokenizing And Writing train.bin / val.bin")
    if total_docs is None:
        print("Processing documents until the target size is reached...")
    else:
        print(f"Processing up to {total_docs:,} documents...")

    with open(TRAIN_BIN, "wb") as train_f, open(VAL_BIN, "wb") as val_f:
        filtered_docs = iter_filtered_documents(parquet_files, text_column, language_column, quality_column)
        progress = tqdm(filtered_docs, total=total_docs, desc="Tokenizing documents", unit="doc")
        for doc_batch in batched(progress, TOKENIZATION_BATCH_SIZE):
            text_batch = [doc["text"] for doc in doc_batch]
            token_batches = tokenizer.encode_batch(text_batch)

            for tokens in token_batches:
                tokens.append(eos_id)

                arr = np.asarray(tokens, dtype=np.uint16)
                if arr.size <= 1:
                    continue

                current_total_bytes = (train_token_count + val_token_count) * 2
                if target_total_bytes is not None and current_total_bytes + arr.nbytes > target_total_bytes:
                    remaining_bytes = target_total_bytes - current_total_bytes
                    remaining_tokens = remaining_bytes // 2
                    if remaining_tokens <= 1:
                        progress.close()
                        print(f"Complete! Train: {train_token_count:,} tokens | Val: {val_token_count:,} tokens")
                        return
                    arr = arr[:remaining_tokens]

                if random.random() < VAL_PERCENT:
                    val_f.write(arr.tobytes())
                    val_token_count += len(arr)
                else:
                    train_f.write(arr.tobytes())
                    train_token_count += len(arr)

                if target_total_bytes is not None and (train_token_count + val_token_count) * 2 >= target_total_bytes:
                    progress.close()
                    print(f"Complete! Train: {train_token_count:,} tokens | Val: {val_token_count:,} tokens")
                    return

    print(f"Complete! Train: {train_token_count:,} tokens | Val: {val_token_count:,} tokens")


if __name__ == "__main__":
    prepare()
