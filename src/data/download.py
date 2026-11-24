"""
SUNDAY LLM - Dataset Download Utilities
==========================================

Phase 5 Implementation.

Utilities to download and prepare training data:
  - OpenWebText (via HuggingFace datasets)
  - The Pile (via HuggingFace datasets)
  - Stanford Alpaca (for fine-tuning)
  - Any custom text files

Usage:
    # Download a small subset for testing
    texts = download_openwebtext(max_samples=10000)

    # Or use custom files
    texts = load_text_files(["data/my_corpus.txt"])
"""

import os
from typing import List, Optional


def download_openwebtext(
    max_samples: Optional[int] = None,
    split: str = "train",
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Download OpenWebText from HuggingFace Datasets.

    OpenWebText is an open-source recreation of WebText (GPT-2's training data).
    It contains ~40GB of web text, filtered for quality.

    Args:
        max_samples: Maximum number of documents to load. None = all (~8M docs).
                     For testing, use 10_000 - 100_000.
        split: Dataset split to use. 'train' is the only split.
        cache_dir: Directory to cache downloaded data.
        verbose: Print progress.

    Returns:
        List of text strings.

    Example:
        >>> texts = download_openwebtext(max_samples=50_000)
        >>> print(len(texts))  # 50000
        >>> print(texts[0][:100])  # First 100 chars of first document
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    if verbose:
        print("[Download] Loading OpenWebText from HuggingFace...")

    # NOTE: 'openwebtext' was renamed to 'Skylion007/openwebtext' on HF Hub
    dataset = load_dataset(
        "Skylion007/openwebtext",
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    texts = dataset["text"]

    if verbose:
        total_chars = sum(len(t) for t in texts)
        print(f"[Download] Loaded {len(texts):,} documents ({total_chars/1e9:.2f} GB)")

    return texts


def download_alpaca(
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Download Stanford Alpaca dataset for instruction fine-tuning.

    Returns 52K instruction-following examples in the format:
      {
        "instruction": "What is the capital of France?",
        "input": "",  # Optional extra context
        "output": "The capital of France is Paris."
      }

    Args:
        cache_dir: Directory to cache downloaded data.
        verbose: Print progress.

    Returns:
        List of dicts with 'instruction', 'input', 'output' keys.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    if verbose:
        print("[Download] Loading Stanford Alpaca from HuggingFace...")

    dataset = load_dataset(
        "tatsu-lab/alpaca",
        split="train",
        cache_dir=cache_dir,
    )

    examples = [
        {
            "instruction": row["instruction"],
            "input": row.get("input", ""),
            "output": row["output"],
        }
        for row in dataset
    ]

    if verbose:
        print(f"[Download] Loaded {len(examples):,} Alpaca examples")

    return examples


def load_text_files(
    filepaths: List[str],
    encoding: str = "utf-8",
    verbose: bool = True,
) -> List[str]:
    """
    Load text from local files. Each line = one document.

    Args:
        filepaths: List of .txt file paths.
        encoding: File encoding.
        verbose: Print progress.

    Returns:
        List of text strings (one per non-empty line).
    """
    all_texts = []

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"[Warning] File not found: {filepath}")
            continue

        with open(filepath, "r", encoding=encoding) as f:
            texts = [line.strip() for line in f if line.strip()]
            all_texts.extend(texts)

        if verbose:
            print(f"[Load] {filepath}: {len(texts):,} lines")

    if verbose:
        print(f"[Load] Total: {len(all_texts):,} documents")

    return all_texts


def create_sample_dataset(
    n_samples: int = 1000,
    min_length: int = 50,
    max_length: int = 500,
    seed: int = 42,
) -> List[str]:
    """
    Create a tiny synthetic dataset for testing the pipeline.

    Generates random sentences — not useful for real training,
    but great for verifying the pipeline works end-to-end.

    Args:
        n_samples: Number of synthetic texts to generate.
        min_length: Minimum text length in characters.
        max_length: Maximum text length in characters.
        seed: Random seed.

    Returns:
        List of synthetic text strings.
    """
    import random
    import string

    random.seed(seed)

    # Simple vocabulary for synthetic text
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "language", "model", "transformer", "attention",
        "neural", "network", "training", "data", "token", "embed", "gradient",
        "loss", "batch", "optimizer", "learning", "rate", "epoch", "checkpoint",
        "SUNDAY", "model", "GPT", "decoder", "encoder", "head", "layer",
        "is", "a", "an", "this", "that", "with", "and", "or", "but", "for",
        "in", "on", "at", "to", "from", "by", "of", "can", "will", "has",
    ]

    texts = []
    for _ in range(n_samples):
        length = random.randint(min_length // 5, max_length // 5)
        sentence_words = [random.choice(words) for _ in range(length)]
        text = " ".join(sentence_words).capitalize() + "."
        texts.append(text)

    print(f"[Dataset] Created {n_samples:,} synthetic texts for testing")
    return texts
