"""
SUNDAY LLM - Optimizer & LR Scheduler
========================================

Phase 6/7 Implementation.

Optimizer: AdamW (Adam with Weight Decay)
  - Standard choice for LLM training
  - β1=0.9, β2=0.95 (GPT-3 style, slightly higher β2 than default)
  - eps=1e-8 for numerical stability
  - weight_decay=0.1 applied to 2D params (weights), not 1D (biases, LayerNorm)

Learning Rate Schedule: Cosine Decay with Linear Warmup
  Phase 1 (warmup):  LR linearly increases from 0 → max_lr
  Phase 2 (cosine):  LR decays from max_lr → min_lr following cosine curve

  Formula:
    warmup:  lr = max_lr * (step / warmup_steps)
    cosine:  lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
    after:   lr = min_lr

GPT-3 Reference Values (scaled for our model):
  max_lr = 6e-4 (for 125M), 1e-3 (for 50M), 3e-3 (for Tiny)
  min_lr = max_lr / 10
  warmup = 2000 steps
  total  = 100K - 500K steps
"""

import math
import torch
import torch.optim as optim
from typing import Optional


def configure_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 6e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    verbose: bool = True,
) -> optim.AdamW:
    """
    Configure AdamW optimizer with correct weight decay settings.

    Key rule: Apply weight decay only to 2D (weight) parameters,
    NOT to 1D parameters (biases, LayerNorm weight/bias).

    Why no weight decay on 1D?
      Weight decay pushes parameters toward 0.
      LayerNorm parameters should not be pushed to 0 — they scale activations.
      Biases are small scalars — weight decay doesn't help them.

    Args:
        model: The SUNDAY model (or any nn.Module).
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization coefficient.
        beta1: Adam β₁ (exponential decay for first moment).
        beta2: Adam β₂ (exponential decay for second moment).
        eps: Adam epsilon for numerical stability.
        verbose: Print parameter group info.

    Returns:
        Configured AdamW optimizer.
    """
    # Separate parameters into two groups
    decay_params = []     # 2D: nn.Linear weight matrices
    no_decay_params = []  # 1D: biases, LayerNorm weight/bias, embeddings

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            # Weight matrices: apply weight decay
            decay_params.append(param)
        else:
            # Biases, LayerNorm params: no weight decay
            no_decay_params.append(param)

    # Count parameters in each group
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)

    if verbose:
        print(f"[Optimizer] Decay params:    {n_decay:>12,} ({len(decay_params)} tensors)")
        print(f"[Optimizer] No-decay params: {n_no_decay:>12,} ({len(no_decay_params)} tensors)")
        print(f"[Optimizer] lr={learning_rate}, wd={weight_decay}, β=({beta1},{beta2})")

    optimizer = optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )

    return optimizer


def get_cosine_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """
    Compute learning rate for the current step using cosine decay with warmup.

    Schedule:
      Steps 0..warmup_steps:         Linear warmup  (0 → max_lr)
      Steps warmup_steps..max_steps: Cosine decay   (max_lr → min_lr)
      Steps > max_steps:             Constant min_lr

    Args:
        step: Current training step (0-indexed).
        max_lr: Peak learning rate.
        min_lr: Minimum learning rate (floor).
        warmup_steps: Number of steps for linear warmup.
        max_steps: Total training steps.

    Returns:
        Learning rate for this step (float).

    Example:
        >>> lr = get_cosine_lr(500, 6e-4, 6e-5, 2000, 100000)
        >>> print(f"{lr:.6f}")  # 0.000150 (in warmup phase)
    """
    # Phase 1: Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Phase 3: After training (shouldn't normally happen)
    if step >= max_steps:
        return min_lr

    # Phase 2: Cosine decay
    # progress: 0.0 → 1.0 over the cosine decay portion
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    # Cosine: 1.0 → 0.0 as progress goes 0.0 → 1.0
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + cosine_factor * (max_lr - min_lr)


def update_lr(optimizer: optim.Optimizer, lr: float) -> None:
    """
    Update learning rate for all parameter groups in an optimizer.

    Args:
        optimizer: PyTorch optimizer.
        lr: New learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
