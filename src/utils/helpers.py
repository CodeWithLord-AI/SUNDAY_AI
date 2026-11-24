"""
SUNDAY LLM - Helper Utilities
===============================

Miscellaneous helper functions used across the project.
"""

import os
import json
import time
from typing import Any, Dict


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters (total, trainable, non-trainable).

    Args:
        model: PyTorch model (nn.Module)

    Returns:
        Dictionary with parameter counts.

    Example:
        >>> stats = count_parameters(model)
        >>> print(f"Total: {stats['total']:,}")
        Total: 49,500,000
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total_millions": round(total / 1e6, 2),
    }


def format_number(n: int) -> str:
    """
    Format a large number with appropriate suffix.

    Example:
        >>> format_number(49500000)
        '49.5M'
        >>> format_number(1200)
        '1.2K'
    """
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """
    Simple context manager for timing code blocks.

    Example:
        >>> with Timer("Training step"):
        ...     loss = train_step(batch)
        [Timer] Training step: 0.42s
    """

    def __init__(self, name: str = "Block"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"[Timer] {self.name}: {elapsed:.2f}s")

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
