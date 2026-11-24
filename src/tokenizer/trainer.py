"""
SUNDAY LLM - BPE Tokenizer Trainer
====================================

This module trains a Byte-Pair Encoding (BPE) tokenizer using
HuggingFace's `tokenizers` library (written in Rust, very fast).

BPE Algorithm:
  1. Start with all individual bytes/characters as the vocabulary
  2. Count all adjacent token pairs in the training corpus
  3. Merge the most frequent pair into a new token
  4. Repeat steps 2-3 until desired vocab_size is reached

Why BPE?
  - Handles ANY text (no out-of-vocabulary problem)
  - Balances between character-level and word-level tokenization
  - Common words → single token (efficient)
  - Rare words → split into subwords (still representable)
  - Used by GPT-2, GPT-3, GPT-4, LLaMA, etc.

Training Pipeline:
  1. Normalizer: Unicode NFC normalization
  2. Pre-tokenizer: Split on whitespace + punctuation (ByteLevel)
  3. BPE Model: Learn merges from frequency statistics
  4. Decoder: ByteLevel decoder to reconstruct text

Usage:
  >>> from src.tokenizer.trainer import train_tokenizer
  >>> tokenizer = train_tokenizer(
  ...     data_files=["data/sample.txt"],
  ...     vocab_size=32768,
  ...     save_dir="tokenizer_files"
  ... )
"""

import os
from typing import List, Optional

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFC


def train_tokenizer(
    data_files: List[str],
    vocab_size: int = 32768,
    min_frequency: int = 2,
    save_dir: str = "tokenizer_files",
    save_name: str = "sunday_tokenizer.json",
    special_tokens: Optional[List[str]] = None,
) -> Tokenizer:
    """
    Train a BPE tokenizer from text files.

    Args:
        data_files: List of text file paths for training.
        vocab_size: Target vocabulary size (default: 32768 = 32K).
        min_frequency: Minimum frequency for a pair to be merged.
                       Higher = fewer rare tokens in vocab.
        save_dir: Directory to save the trained tokenizer.
        save_name: Filename for the saved tokenizer JSON.
        special_tokens: List of special tokens. If None, uses defaults.

    Returns:
        Trained HuggingFace Tokenizer object.

    Example:
        >>> tokenizer = train_tokenizer(
        ...     data_files=["data/train.txt"],
        ...     vocab_size=32768,
        ... )
        >>> encoded = tokenizer.encode("Hello, world!")
        >>> print(encoded.ids)    # [15496, 11, 995, 0]
        >>> print(encoded.tokens) # ['Hello', ',', 'Ġworld', '!']
    """
    # --- Step 1: Define special tokens ---
    if special_tokens is None:
        special_tokens = [
            "<pad>",   # Padding token (ID: 0)
            "<unk>",   # Unknown token (ID: 1)
            "<bos>",   # Beginning of sequence (ID: 2)
            "<eos>",   # End of sequence (ID: 3)
            "<sep>",   # Separator for instruction tuning (ID: 4)
        ]

    print(f"[SUNDAY Tokenizer] Training BPE tokenizer...")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Min frequency: {min_frequency}")
    print(f"  Training files: {data_files}")
    print(f"  Special tokens: {special_tokens}")

    # --- Step 2: Initialize BPE model ---
    # ByteLevel BPE: works on bytes, can represent ANY text
    # (no unknown characters possible — worst case, falls back to bytes)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # --- Step 3: Set up normalizer ---
    # NFC: Unicode Canonical Decomposition + Canonical Composition
    # Ensures consistent representation of accented characters etc.
    # e.g., "é" can be encoded as 1 char or 2 chars — NFC normalizes this
    tokenizer.normalizer = NFC()

    # --- Step 4: Set up pre-tokenizer ---
    # ByteLevel pre-tokenizer:
    # - Converts text to bytes first (handles any encoding)
    # - Splits on word boundaries
    # - Adds 'Ġ' prefix for tokens after whitespace
    #   e.g., "Hello world" → ["Hello", "Ġworld"]
    #   The 'Ġ' tells us this token had a space before it
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # --- Step 5: Set up decoder ---
    # ByteLevel decoder reverses the byte encoding
    tokenizer.decoder = decoders.ByteLevel()

    # --- Step 6: Set up post-processor ---
    # ByteLevel post-processor handles offset tracking
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # --- Step 7: Configure the BPE trainer ---
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # --- Step 8: Train! ---
    print(f"\n[SUNDAY Tokenizer] Starting training...")
    tokenizer.train(files=data_files, trainer=trainer)

    # --- Step 9: Verify ---
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"\n[SUNDAY Tokenizer] Training complete!")
    print(f"  Final vocab size: {actual_vocab_size:,}")

    # Verify special tokens have correct IDs
    for i, token in enumerate(special_tokens):
        token_id = tokenizer.token_to_id(token)
        print(f"  {token}: ID = {token_id}")
        assert token_id is not None, f"Special token '{token}' not found in vocabulary!"

    # --- Step 10: Save ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    tokenizer.save(save_path)
    print(f"\n[SUNDAY Tokenizer] Saved to: {save_path}")

    return tokenizer


def train_tokenizer_from_texts(
    texts: List[str],
    vocab_size: int = 32768,
    min_frequency: int = 2,
    save_dir: str = "tokenizer_files",
    save_name: str = "sunday_tokenizer.json",
    special_tokens: Optional[List[str]] = None,
) -> Tokenizer:
    """
    Train a BPE tokenizer from a list of text strings (instead of files).
    Useful for quick experiments or when data is already in memory.

    Args:
        texts: List of text strings to train on.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum pair frequency for merging.
        save_dir: Directory to save the trained tokenizer.
        save_name: Filename for the saved tokenizer.
        special_tokens: List of special tokens.

    Returns:
        Trained Tokenizer object.
    """
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]

    print(f"[SUNDAY Tokenizer] Training from {len(texts):,} text samples...")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train from iterator instead of files
    tokenizer.train_from_iterator(texts, trainer=trainer)

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"[SUNDAY Tokenizer] Done! Vocab size: {actual_vocab_size:,}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    tokenizer.save(save_path)
    print(f"[SUNDAY Tokenizer] Saved to: {save_path}")

    return tokenizer
