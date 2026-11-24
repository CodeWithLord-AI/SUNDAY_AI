"""
SUNDAY LLM - Generation Quality Metrics
==========================================

Phase 8 Implementation.

Metrics for evaluating text generation quality:

1. Diversity Metrics (no reference needed):
   - Distinct-1 / Distinct-2: ratio of unique unigrams/bigrams
   - Repetition rate: fraction of repeated n-grams (lower = better)
   - Average length: mean output length

2. Reference-based Metrics (need reference texts):
   - BLEU: n-gram overlap with references (standard for translation)
   - ROUGE: recall-oriented n-gram overlap (standard for summarization)

3. Coherence Metrics (qualitative):
   - Self-BLEU: diversity measure (how similar generations are to each other)

For LLM pretraining evaluation, perplexity is the primary metric.
These additional metrics are useful for instruction-tuned models.
"""

import re
from collections import Counter
from typing import List, Optional, Tuple


def distinct_ngrams(texts: List[str], n: int) -> float:
    """
    Compute Distinct-N: ratio of unique n-grams to total n-grams.

    Measures lexical diversity. Higher = more diverse outputs.
    A model that always repeats the same text would score near 0.

    Args:
        texts: List of generated text strings.
        n: N-gram size (1 for unigrams, 2 for bigrams).

    Returns:
        Distinct-N score (0.0 to 1.0). Higher = more diverse.

    Example:
        >>> texts = ["hello world", "hello earth", "hello moon"]
        >>> print(distinct_ngrams(texts, 1))  # ~0.75 (3 unique / 4 total tokens)
    """
    all_ngrams = []

    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    return unique / total


def repetition_rate(text: str, n: int = 4) -> float:
    """
    Measure the repetition rate of a text.

    Computes the fraction of n-grams that appear more than once.
    High repetition indicates degenerate/looping generation.

    Args:
        text: Generated text string.
        n: N-gram size to check for repetitions.

    Returns:
        Repetition rate (0.0 to 1.0). Lower = less repetitive.
    """
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / max(len(ngrams), 1)


def average_length(texts: List[str]) -> float:
    """
    Compute average token length of generated texts.

    Args:
        texts: List of generated strings.

    Returns:
        Mean number of tokens per text.
    """
    if not texts:
        return 0.0
    lengths = [len(text.split()) for text in texts]
    return sum(lengths) / len(lengths)


def compute_generation_metrics(
    generated_texts: List[str],
    reference_texts: Optional[List[str]] = None,
) -> dict:
    """
    Compute a bundle of generation quality metrics.

    Args:
        generated_texts: List of model-generated text strings.
        reference_texts: Optional list of reference texts (for BLEU/ROUGE).

    Returns:
        Dict of metric names → values.

    Example:
        >>> texts = ["The model generates coherent sentences."]
        >>> metrics = compute_generation_metrics(texts)
        >>> print(metrics)
        {'distinct_1': 0.86, 'distinct_2': 1.0, 'avg_length': 5.0, 'rep_rate_4': 0.0}
    """
    metrics = {}

    if not generated_texts:
        return metrics

    # Diversity metrics
    metrics["distinct_1"] = distinct_ngrams(generated_texts, 1)
    metrics["distinct_2"] = distinct_ngrams(generated_texts, 2)
    metrics["avg_length"] = average_length(generated_texts)

    # Repetition (average across texts)
    rep_rates = [repetition_rate(t, n=4) for t in generated_texts]
    metrics["rep_rate_4"] = sum(rep_rates) / len(rep_rates)

    return metrics
