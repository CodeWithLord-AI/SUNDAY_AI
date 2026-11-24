"""
SUNDAY LLM - Sampling Strategies
===================================

Phase 10 Implementation.

Different decoding strategies for text generation:

1. Greedy Decoding:
   Always pick the highest-probability next token.
   Fast, deterministic, but often repetitive.

2. Temperature Sampling:
   Scale logits by 1/T before sampling:
   - T < 1: sharper distribution (more confident, less diverse)
   - T = 1: standard sampling
   - T > 1: flatter distribution (more random, more diverse)

3. Top-K Sampling:
   Sample only from the K most likely tokens.
   Prevents selecting very unlikely tokens.

4. Top-P (Nucleus) Sampling:
   Sample from the smallest set of tokens whose
   cumulative probability ≥ p.
   Adapts dynamically to the distribution width.

5. Repetition Penalty:
   Penalize tokens that have already appeared in the context.
   Reduces repetitive/looping outputs.

Best practice: Temperature + Top-P (nucleus) sampling
  temperature=0.8, top_p=0.9 — widely used default for chat models.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy decoding: always pick the highest probability token.

    Args:
        logits: Unnormalized log-probabilities, shape (batch, vocab_size).

    Returns:
        Selected token IDs, shape (batch, 1).
    """
    return logits.argmax(dim=-1, keepdim=True)


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Apply temperature scaling and sample.

    temperature < 1.0 → more deterministic (sharpens distribution)
    temperature > 1.0 → more random (flattens distribution)
    temperature = 1.0 → no change (standard sampling)

    Args:
        logits: Unnormalized log-probabilities, shape (batch, vocab_size).
        temperature: Scaling factor. Must be > 0.

    Returns:
        Sampled token IDs, shape (batch, 1).
    """
    assert temperature > 0, "Temperature must be positive"

    # Scale logits
    scaled_logits = logits / temperature

    # Sample from distribution
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def apply_top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filter logits to keep only the top-K values.

    Sets all logits outside the top-K to -inf.

    Args:
        logits: Shape (batch, vocab_size).
        k: Number of top tokens to keep.

    Returns:
        Filtered logits with the same shape.
    """
    if k == 0:
        return logits  # No filtering

    k = min(k, logits.size(-1))

    # Get the k-th largest logit value
    kth_values, _ = torch.topk(logits, k, dim=-1)
    min_values = kth_values[:, -1].unsqueeze(-1)  # (batch, 1)

    # Zero out everything below the k-th value
    filtered = logits.masked_fill(logits < min_values, float("-inf"))
    return filtered


def apply_top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability ≥ p.

    Args:
        logits: Shape (batch, vocab_size).
        p: Cumulative probability threshold (0.0 to 1.0).

    Returns:
        Filtered logits with the same shape.
    """
    if p >= 1.0:
        return logits  # No filtering

    # Sort in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens where cumulative prob exceeds p
    # (shift right by 1 to keep the token that pushes us over p)
    mask = cumulative_probs - sorted_probs > p
    sorted_logits.masked_fill_(mask, float("-inf"))

    # Scatter back to original ordering
    filtered = sorted_logits.scatter(1, sorted_indices, sorted_logits)
    return filtered


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float = 1.3,
) -> torch.Tensor:
    """
    Apply repetition penalty to reduce repeated tokens.

    Penalizes tokens that have already appeared in the generated context.
    Positive logits are divided by penalty; negative logits are multiplied.

    Args:
        logits: Shape (1, vocab_size) — single sample only.
        generated_ids: List of already-generated token IDs.
        penalty: Penalty factor. 1.0 = no penalty. >1.0 = more penalty.

    Returns:
        Penalized logits, shape (1, vocab_size).
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    score = logits.clone()

    for token_id in set(generated_ids):
        if token_id < score.shape[-1]:
            if score[0, token_id] < 0:
                score[0, token_id] *= penalty
            else:
                score[0, token_id] /= penalty

    return score


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    generated_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Sample the next token using the configured strategy.

    Applies: repetition penalty → top-k → top-p → temperature → sample

    Args:
        logits: Shape (batch, vocab_size).
        temperature: Sampling temperature.
        top_k: Top-K filtering (0 = disabled).
        top_p: Nucleus sampling (1.0 = disabled).
        greedy: If True, ignore temperature and always pick argmax.
        generated_ids: Previously generated IDs for repetition penalty.
        repetition_penalty: Repetition penalty factor.

    Returns:
        Next token IDs, shape (batch, 1).
    """
    if greedy:
        return sample_greedy(logits)

    # Apply repetition penalty
    if repetition_penalty != 1.0 and generated_ids:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Apply top-k
    if top_k > 0:
        logits = apply_top_k_filter(logits, top_k)

    # Apply top-p (nucleus)
    if top_p < 1.0:
        logits = apply_top_p_filter(logits, top_p)

    # Apply temperature and sample
    return sample_with_temperature(logits, temperature)
