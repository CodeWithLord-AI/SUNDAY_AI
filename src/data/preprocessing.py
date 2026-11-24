"""
SUNDAY LLM - Text Preprocessing
==================================

Phase 5 Implementation.

Preprocessing pipeline for raw text data before tokenization.

Steps:
  1. Text cleaning (remove noise, normalize whitespace)
  2. Tokenization (via SundayTokenizer)
  3. Corpus flattening (join all sequences into one flat token list)
  4. Train/val split (before tokenization for text, or after for token IDs)
  5. Save as binary (.bin) for efficient loading during training

Why flatten into one big sequence?
  During training we sample random windows of max_seq_len tokens.
  By flattening, we maximize data utilization — no gaps between documents.
  We separate documents with an <eos> token so the model learns document
  boundaries naturally.
"""

import os
import re
import unicodedata
from typing import List, Optional, Tuple

import numpy as np


def clean_text(text: str) -> str:
    """
    Basic text cleaning for LLM pretraining.

    Performs:
      - Unicode normalization (NFC)
      - Strip control characters (except newlines/tabs)
      - Normalize whitespace (collapse multiple spaces/newlines)
      - Remove HTML tags if present

    Args:
        text: Raw input text string.

    Returns:
        Cleaned text string.
    """
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove control characters (keep \n, \t)
    text = "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C")
    )

    # Normalize multiple spaces → single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize multiple newlines → max 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_texts(texts: List[str], verbose: bool = True) -> List[str]:
    """
    Clean a list of text strings.

    Args:
        texts: List of raw text strings.
        verbose: Whether to print progress.

    Returns:
        List of cleaned text strings.
    """
    if verbose:
        print(f"[Preprocessing] Cleaning {len(texts):,} texts...")

    cleaned = []
    skipped = 0

    for text in texts:
        c = clean_text(text)
        if len(c) > 20:  # Skip very short texts
            cleaned.append(c)
        else:
            skipped += 1

    if verbose:
        print(f"[Preprocessing] Done: {len(cleaned):,} kept, {skipped:,} skipped")

    return cleaned


def tokenize_corpus(
    texts: List[str],
    tokenizer,
    add_eos: bool = True,
    verbose: bool = True,
) -> List[int]:
    """
    Tokenize a list of texts and flatten into one continuous token sequence.

    Each text is tokenized and joined with <eos> tokens between documents.
    This creates a continuous stream suitable for causal LM training.

    Args:
        texts: List of text strings to tokenize.
        tokenizer: A SundayTokenizer instance.
        add_eos: Whether to add <eos> token at end of each document.
        verbose: Whether to print progress.

    Returns:
        Flat list of token IDs (the entire corpus as one sequence).

    Example:
        >>> texts = ["Hello world", "Another doc"]
        >>> ids = tokenize_corpus(texts, tokenizer)
        >>> # ids = [2, 15496, 995, 3, 2, 7652, 10412, 3]
        >>> #        ^bos ^Hello ^world ^eos ...
    """
    if verbose:
        print(f"[Preprocessing] Tokenizing {len(texts):,} documents...")

    all_token_ids = []
    total_tokens = 0

    for i, text in enumerate(texts):
        # Tokenize with <bos> and optionally <eos>
        ids = tokenizer.encode(text, add_bos=True, add_eos=add_eos)
        all_token_ids.extend(ids)
        total_tokens += len(ids)

        if verbose and (i + 1) % 10000 == 0:
            print(f"  [{i+1:,}/{len(texts):,}] {total_tokens:,} tokens so far...")

    if verbose:
        print(f"[Preprocessing] Total tokens: {total_tokens:,}")

    return all_token_ids


def save_token_ids_binary(
    token_ids: List[int],
    filepath: str,
    dtype=np.uint16,
) -> None:
    """
    Save token IDs to a binary file for efficient training loading.

    Uses uint16 (2 bytes/token) since vocab_size=32768 fits in uint16.
    This is 2x more compact than int32.

    For 1B tokens: uint16 → 2 GB file
                   int32  → 4 GB file

    Args:
        token_ids: List of integer token IDs.
        filepath: Output .bin file path.
        dtype: NumPy dtype to use. Default: np.uint16.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    arr = np.array(token_ids, dtype=dtype)
    arr.tofile(filepath)
    print(
        f"[Preprocessing] Saved {len(token_ids):,} tokens to {filepath} "
        f"({os.path.getsize(filepath) / 1e6:.1f} MB)"
    )


def load_token_ids_binary(filepath: str, dtype=np.uint16) -> np.ndarray:
    """
    Load token IDs from a binary file (memory-mapped for efficiency).

    Args:
        filepath: Path to the .bin file.
        dtype: NumPy dtype used when saving.

    Returns:
        Memory-mapped numpy array of token IDs.
    """
    data = np.memmap(filepath, dtype=dtype, mode="r")
    print(f"[Preprocessing] Loaded {len(data):,} tokens from {filepath}")
    return data


def prepare_text_corpus(
    texts: List[str],
    tokenizer,
    output_dir: str = "data",
    train_split: float = 0.9,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[str, str]:
    """
    Full preprocessing pipeline: clean → tokenize → split → save.

    Args:
        texts: Raw text strings.
        tokenizer: Trained SundayTokenizer.
        output_dir: Directory to save train.bin and val.bin.
        train_split: Fraction for training (rest is validation).
        seed: Random seed for reproducible split.
        verbose: Print progress.

    Returns:
        (train_path, val_path): Paths to the saved binary files.
    """
    import random
    random.seed(seed)

    # Step 1: Clean
    texts = clean_texts(texts, verbose=verbose)

    # Step 2: Shuffle documents
    random.shuffle(texts)

    # Step 3: Split
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    if verbose:
        print(f"[Preprocessing] Train docs: {len(train_texts):,}")
        print(f"[Preprocessing] Val docs:   {len(val_texts):,}")

    # Step 4: Tokenize
    train_ids = tokenize_corpus(train_texts, tokenizer, verbose=verbose)
    val_ids = tokenize_corpus(val_texts, tokenizer, verbose=verbose)

    # Step 5: Save
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    save_token_ids_binary(train_ids, train_path)
    save_token_ids_binary(val_ids, val_path)

    return train_path, val_path
