"""
SUNDAY LLM - Model Utilities
==============================

Phase 4 Implementation.

Utility functions for the SUNDAY model:
  - Weight initialization (GPT-2 style)
  - Parameter counting
  - Model info printing
"""

import math
import torch
import torch.nn as nn
from typing import Dict


def init_weights(module: nn.Module, n_layers: int) -> None:
    """
    Initialize model weights following GPT-2 convention.

    Rules:
      - Linear layers: Normal(mean=0, std=0.02)
      - Embeddings:    Normal(mean=0, std=0.02)
      - LayerNorm:     weight=1, bias=0
      - Residual projections (out_proj, fc2): scaled by 1/sqrt(2 * n_layers)
        This prevents the variance from growing with depth.

    Args:
        module: The nn.Module to initialize.
        n_layers: Total number of transformer layers (for residual scaling).
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def scale_residual_projections(model: nn.Module, n_layers: int) -> None:
    """
    Scale down weights of residual projection layers by 1/sqrt(2 * n_layers).

    GPT-2 paper applies this to the output projections of attention
    and FFN, so the residual stream doesn't grow in variance with depth.

    Targeted layers:
      - attention.out_proj  (CausalSelfAttention output)
      - ffn.fc2             (FeedForward output)

    Args:
        model: The full SundayModel.
        n_layers: Number of transformer blocks.
    """
    scale = 1.0 / math.sqrt(2 * n_layers)

    for name, param in model.named_parameters():
        if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
            with torch.no_grad():
                param.mul_(scale)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters by component.

    Args:
        model: The SUNDAY model (or any nn.Module).

    Returns:
        Dictionary with parameter counts:
          {
            "total": int,
            "trainable": int,
            "non_trainable": int,
            "embeddings": int,
            "attention": int,
            "ffn": int,
            "layer_norms": int,
            "lm_head": int,
          }

    Example:
        >>> counts = count_parameters(model)
        >>> print(f"Total: {counts['total']:,}")
        Total: 49,500,000
    """
    counts = {
        "total": 0,
        "trainable": 0,
        "non_trainable": 0,
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "layer_norms": 0,
        "lm_head": 0,
    }

    for name, param in model.named_parameters():
        n = param.numel()
        counts["total"] += n

        if param.requires_grad:
            counts["trainable"] += n
        else:
            counts["non_trainable"] += n

        # Categorize by name
        if "token_embedding" in name or "position_embedding" in name:
            counts["embeddings"] += n
        elif "attention" in name:
            counts["attention"] += n
        elif "ffn" in name:
            counts["ffn"] += n
        elif "ln" in name or "layer_norm" in name:
            counts["layer_norms"] += n
        elif "lm_head" in name:
            counts["lm_head"] += n

    return counts


def print_model_info(model: nn.Module, config: dict) -> None:
    """
    Print a summary of the model architecture and parameter counts.

    Args:
        model: The SUNDAY model.
        config: Model configuration dictionary.
    """
    counts = count_parameters(model)

    print("=" * 60)
    print(f"  SUNDAY Model: {config.get('model_name', 'SUNDAY')}")
    print("=" * 60)
    print(f"  Architecture:")
    print(f"    vocab_size  : {config.get('vocab_size', '?'):>10,}")
    print(f"    max_seq_len : {config.get('max_seq_len', '?'):>10,}")
    print(f"    d_model     : {config.get('d_model', '?'):>10,}")
    print(f"    n_heads     : {config.get('n_heads', '?'):>10,}")
    print(f"    n_layers    : {config.get('n_layers', '?'):>10,}")
    print(f"    d_ff        : {config.get('d_ff', '?'):>10,}")
    print(f"    dropout     : {config.get('dropout', '?'):>10}")
    print(f"    bias        : {str(config.get('bias', '?')):>10}")
    print(f"    weight_tying: {str(config.get('weight_tying', '?')):>10}")
    print()
    print(f"  Parameter Counts:")
    print(f"    Total       : {counts['total']:>12,}")
    print(f"    Trainable   : {counts['trainable']:>12,}")
    print(f"    Non-trainable:{counts['non_trainable']:>11,}")
    print(f"    Embeddings  : {counts['embeddings']:>12,}")
    print(f"    Attention   : {counts['attention']:>12,}")
    print(f"    FFN         : {counts['ffn']:>12,}")
    print(f"    LayerNorms  : {counts['layer_norms']:>12,}")
    print(f"    LM Head     : {counts['lm_head']:>12,}")
    print(f"    (~{counts['total'] / 1e6:.1f}M params)")
    print("=" * 60)
