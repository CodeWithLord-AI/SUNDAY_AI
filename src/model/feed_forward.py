"""
SUNDAY LLM - Feed-Forward Network
====================================

Phase 4 Implementation.

The Feed-Forward Network (FFN) is the second sub-block of each
Transformer layer. It applies two linear transformations with a
GELU activation in between.

Formula:
  FFN(x) = Dropout(W_2 * GELU(W_1 * x + b_1) + b_2)

Dimensions:
  Input:  (batch, seq_len, d_model)
  Hidden: (batch, seq_len, d_ff)   ← 4x expansion
  Output: (batch, seq_len, d_model) ← project back

SUNDAY Config (125M):
  d_model = 768, d_ff = 3072 (4x)
  Parameters: 2 × 768 × 3072 = 4,718,592 per layer

Why GELU?
  Gaussian Error Linear Unit — smooth, probabilistic gating.
  Outperforms ReLU for language models (used in GPT, BERT).
  GELU(x) ≈ x * Φ(x) where Φ is the Gaussian CDF.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applied independently to each token position (hence "position-wise").
    Two linear layers with GELU activation and dropout.

    Args:
        d_model (int): Input/output dimension.
        d_ff (int): Hidden (expanded) dimension. Typically 4 × d_model.
        dropout (float): Dropout rate after first activation.
        bias (bool): Whether to use bias in linear layers.
        activation (str): Activation function. 'gelu' or 'relu'.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = False,
        activation: str = "gelu",
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_p = dropout

        # --- First linear: d_model → d_ff (expansion) ---
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)

        # --- Second linear: d_ff → d_model (projection back) ---
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

        # --- Dropout ---
        self.dropout = nn.Dropout(dropout)

        # --- Activation ---
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation '{activation}'. Use 'gelu' or 'relu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FFN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)

        Steps:
            1. Expand: d_model → d_ff  (W_1 @ x)
            2. Activate: GELU
            3. Dropout
            4. Project: d_ff → d_model (W_2 @ h)
        """
        # (B, T, d_model) → (B, T, d_ff)
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout(h)

        # (B, T, d_ff) → (B, T, d_model)
        out = self.fc2(h)

        return out

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, "
            f"dropout={self.dropout_p}"
        )
