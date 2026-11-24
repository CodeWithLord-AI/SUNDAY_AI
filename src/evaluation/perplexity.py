"""
SUNDAY LLM - Perplexity Metric
================================

Phase 8 Implementation.

Perplexity is the primary evaluation metric for language models.

Perplexity = exp(average_cross_entropy_loss)

Intuition:
  - Perplexity measures how "surprised" the model is by the text.
  - Lower perplexity = model assigns higher probability to the text.
  - A model with perplexity K behaves like uniform choice over K tokens.
  - GPT-2 (117M): ~29 on WikiText-103
  - GPT-3 (175B): ~20 on Penn Treebank
  - Our target: perplexity < 50 on our val set (ambitious for 50M model)

Bit-per-character (BPC) is an alternative metric:
  BPC = average_cross_entropy_loss / log(2)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_amp: bool = False,
) -> float:
    """
    Compute perplexity of a model on a dataset.

    Perplexity = exp(mean cross-entropy loss)

    Args:
        model: The SUNDAY model (or any causal LM).
        data_loader: DataLoader yielding (input_ids, target_ids) pairs.
        device: Computation device.
        max_batches: Limit evaluation to first N batches (for speed).
                     None = evaluate on entire dataset.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Perplexity (float). Lower = better.

    Example:
        >>> ppl = compute_perplexity(model, val_loader, device)
        >>> print(f"Validation Perplexity: {ppl:.2f}")
        Validation Perplexity: 45.3
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch_idx, (input_ids, target_ids) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, _ = model(input_ids, targets=target_ids)

        # Accumulate loss weighted by sequence length
        # (handles variable-length sequences correctly)
        batch_tokens = target_ids.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        n_batches += 1

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens

    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    return perplexity


def bits_per_character(loss: float) -> float:
    """
    Convert cross-entropy loss (nats) to bits-per-character.

    BPC = loss / ln(2)

    Args:
        loss: Cross-entropy loss in nats.

    Returns:
        Bits per character (lower = better).
    """
    return loss / math.log(2)
