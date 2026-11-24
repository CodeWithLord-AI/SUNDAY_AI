"""
SUNDAY LLM - Tokenizer Utilities
==================================

Helper functions for tokenizer analysis and visualization.

These utilities help you understand:
  - How text gets split into tokens
  - Token frequency distribution
  - Vocabulary coverage
  - Tokenization efficiency (tokens per word ratio)
"""

from typing import List, Dict, Optional
from collections import Counter


def analyze_tokenization(
    tokenizer,
    texts: List[str],
    top_n: int = 20,
) -> Dict:
    """
    Analyze how a tokenizer handles a set of texts.

    Returns statistics about token usage, efficiency, and coverage.

    Args:
        tokenizer: SundayTokenizer instance.
        texts: List of text strings to analyze.
        top_n: Number of top tokens to show.

    Returns:
        Dictionary with analysis results.
    """
    all_token_ids = []
    all_tokens = []
    total_chars = 0
    total_words = 0

    for text in texts:
        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        tokens = tokenizer.get_tokens(text)
        all_token_ids.extend(ids)
        all_tokens.extend(tokens)
        total_chars += len(text)
        total_words += len(text.split())

    total_tokens = len(all_token_ids)

    # Token frequency
    token_counts = Counter(all_tokens)
    most_common = token_counts.most_common(top_n)

    # Efficiency metrics
    tokens_per_word = total_tokens / max(total_words, 1)
    chars_per_token = total_chars / max(total_tokens, 1)
    unique_tokens_used = len(set(all_token_ids))

    results = {
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_chars": total_chars,
        "unique_tokens_used": unique_tokens_used,
        "vocab_coverage": unique_tokens_used / tokenizer.vocab_size * 100,
        "tokens_per_word": round(tokens_per_word, 2),
        "chars_per_token": round(chars_per_token, 2),
        "most_common_tokens": most_common,
    }

    return results


def print_tokenization(tokenizer, text: str) -> None:
    """
    Visualize how a text gets tokenized.

    Shows each token with its ID and boundaries.

    Example output:
      Text: "The cat sat on the mat"
      Tokens: ['The', 'Ġcat', 'Ġsat', 'Ġon', 'Ġthe', 'Ġmat']
      IDs:    [464,   3797,   3332,  319,   262,    2603]
      Total:  6 tokens for 6 words (1.0 tokens/word)
    """
    tokens = tokenizer.get_tokens(text)
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)

    word_count = len(text.split())

    print(f"\n{'='*60}")
    print(f"  Text: \"{text}\"")
    print(f"{'='*60}")
    print(f"  Tokens ({len(tokens)}): {tokens}")
    print(f"  IDs    ({len(ids)}): {ids}")
    print(f"  Words: {word_count} | Tokens/word: {len(tokens)/max(word_count,1):.2f}")
    print(f"{'='*60}\n")


def print_analysis(analysis: Dict) -> None:
    """Pretty-print the results from analyze_tokenization()."""
    print(f"\n{'='*50}")
    print(f"  SUNDAY Tokenizer Analysis")
    print(f"{'='*50}")
    print(f"  Total tokens:      {analysis['total_tokens']:,}")
    print(f"  Total words:       {analysis['total_words']:,}")
    print(f"  Unique tokens:     {analysis['unique_tokens_used']:,}")
    print(f"  Vocab coverage:    {analysis['vocab_coverage']:.1f}%")
    print(f"  Tokens per word:   {analysis['tokens_per_word']}")
    print(f"  Chars per token:   {analysis['chars_per_token']}")
    print(f"\n  Top {len(analysis['most_common_tokens'])} tokens:")
    for token, count in analysis["most_common_tokens"]:
        print(f"    {token!r:20s} → {count:,}")
    print(f"{'='*50}\n")
