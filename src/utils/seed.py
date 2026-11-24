"""
SUNDAY LLM - Reproducibility Utilities
========================================

Why do we need reproducibility?
  Machine learning involves randomness: weight initialization, data shuffling,
  dropout masks, etc. Setting seeds ensures that experiments can be reproduced
  exactly. This is critical for:
  - Debugging (same input → same output)
  - Comparing experiments (only hyperparams change, not randomness)
  - Scientific rigor (others can verify your results)

What this module does:
  Sets random seeds for Python, NumPy, PyTorch (CPU + CUDA) to ensure
  deterministic behavior across runs.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value. Default is 42 (a classic choice).

    Example:
        >>> set_seed(42)
        >>> torch.randn(3)  # Will always produce the same values
        tensor([-0.4621,  0.1816, -0.6297])
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Make CuDNN deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[SUNDAY] Random seed set to {seed}")
