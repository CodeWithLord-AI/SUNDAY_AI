"""
SUNDAY LLM - LoRA (Low-Rank Adaptation)
==========================================

Phase 9 Implementation.

LoRA is a parameter-efficient fine-tuning method that:
  1. Freezes all original model weights
  2. Adds small trainable rank-decomposition matrices to attention layers
  3. Only trains the new matrices (much fewer parameters!)

The key insight:
  Instead of updating W (d_model × d_model),
  we update: ΔW = A @ B
  Where: A is (d_model × rank), B is (rank × d_model)
  rank << d_model  → much fewer parameters!

Example for 125M model, rank=16:
  Original attention: 4 × 768 × 768 = 2,359,296 params per layer
  LoRA matrices:      4 × 2 × (768 × 16) = 98,304 params per layer
  Compression:        ~24x fewer parameters to train!

Formula:
  h = W₀x + ΔWx = W₀x + (B @ A)x
  Where A is initialized from N(0, σ) and B from zeros.
  (B starts at zero so ΔW = 0 at initialization — no initial disruption)

Scaling:
  ΔW is scaled by α/rank, where α is a hyperparameter.
  Convention: set α = rank (so α/rank = 1.0).

Reference: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional


class LoRALinear(nn.Module):
    """
    A linear layer with LoRA adaptation.

    Replaces nn.Linear with: out = W₀ @ x + (B @ A) @ x * (alpha/rank)

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        rank (int): LoRA rank. Typical values: 4, 8, 16, 32.
        alpha (float): LoRA scaling. Convention: set to rank.
        dropout (float): Dropout on LoRA path.
        bias (bool): Whether the original linear has bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # --- Original (frozen) linear layer ---
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # --- LoRA matrices ---
        # A: (rank, in_features) — initialized from Gaussian
        # B: (out_features, rank) — initialized to zero (so ΔW=0 initially)
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        # Dropout on LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze the base linear weights
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base output + LoRA correction.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base forward pass (frozen)
        base_out = self.linear(x)

        # LoRA path: x → dropout → A → B → scale
        # x: (..., in_features)
        lora_out = self.lora_dropout(x) @ self.lora_A.T  # (..., rank)
        lora_out = lora_out @ self.lora_B.T               # (..., out_features)
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    def merge_weights(self) -> None:
        """
        Merge LoRA weights into the base weights for inference efficiency.

        After merging: W_merged = W₀ + B @ A * scaling
        This eliminates the extra computation at inference time.

        WARNING: Once merged, you cannot separate the adapters again.
        """
        with torch.no_grad():
            delta_W = (self.lora_B @ self.lora_A) * self.scaling
            self.linear.weight.data += delta_W

        print(f"[LoRA] Merged LoRA weights (rank={self.rank})")


def apply_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: float = 16.0,
    lora_dropout: float = 0.1,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.

    Replaces target linear layers with LoRALinear wrappers.
    Freezes all non-LoRA parameters.

    Args:
        model: The model to apply LoRA to.
        target_modules: List of module name substrings to target.
                        Common: ['qkv_proj', 'out_proj'] for attention layers.
        rank: LoRA rank.
        alpha: LoRA scaling.
        lora_dropout: Dropout on LoRA path.
        verbose: Print modification info.

    Returns:
        The model with LoRA applied.

    Example:
        >>> model = SundayModel.from_config("50m")
        >>> model = apply_lora(model, target_modules=['qkv_proj', 'out_proj'], rank=16)
        >>> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        >>> print(f"LoRA trainable params: {trainable:,}")
    """
    # --- First, freeze all parameters ---
    for param in model.parameters():
        param.requires_grad = False

    replaced = []

    # --- Replace target linear layers with LoRALinear ---
    def _replace_module(parent: nn.Module, name: str, child: nn.Module) -> None:
        """Recursively find and replace target modules."""
        for attr_name in dir(parent):
            if attr_name.startswith("_"):
                continue
        for child_name, child_module in parent.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this module should be replaced
            if isinstance(child_module, nn.Linear):
                for target in target_modules:
                    if target in child_name:
                        # Replace with LoRALinear
                        lora_layer = LoRALinear(
                            in_features=child_module.in_features,
                            out_features=child_module.out_features,
                            rank=rank,
                            alpha=alpha,
                            dropout=lora_dropout,
                            bias=child_module.bias is not None,
                        )
                        # Copy existing weights
                        lora_layer.linear.weight.data.copy_(child_module.weight.data)
                        if child_module.bias is not None:
                            lora_layer.linear.bias.data.copy_(child_module.bias.data)

                        setattr(parent, child_name, lora_layer)
                        replaced.append(full_name)
                        break
            else:
                _replace_module(child_module, full_name, child_module)

    _replace_module(model, "", model)

    if verbose:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[LoRA] Applied to {len(replaced)} layers: {replaced}")
        print(f"[LoRA] Trainable params: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.2f}%)")

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into their base layers for efficient inference.

    Args:
        model: Model with LoRA adapters applied.

    Returns:
        The model with merged weights (inference-ready).
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()

    return model
