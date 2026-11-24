"""
SUNDAY LLM - Transformer Block
=================================

Phase 4 Implementation.

One Transformer block = the repeating unit of the model.
SUNDAY stacks N of these blocks (8 for 50M, 12 for 125M).

Architecture (Pre-Norm style):
  a = x + MultiHeadAttention(LayerNorm(x))     # Sub-block 1
  h = a + FeedForward(LayerNorm(a))              # Sub-block 2

Pre-Norm vs Post-Norm:
  - Pre-Norm: LayerNorm BEFORE attention/FFN (modern, more stable)
  - Post-Norm: LayerNorm AFTER residual add (original 2017 paper)
  SUNDAY uses Pre-Norm (like GPT-2, LLaMA).

Residual Connections:
  output = Layer(x) + x
  Allows gradients to flow directly through skip connections,
  enabling training of deep networks.

Parameters per block (125M config):
  Attention: 2,359,296
  FFN:       4,718,592
  LayerNorm: 3,072
  Total:     ~7.08M
"""

import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention
from src.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block (Pre-Norm style).

    Applies:
      1. LayerNorm → CausalSelfAttention → Residual add
      2. LayerNorm → FeedForward → Residual add

    Args:
        d_model (int): Embedding dimension.
        n_heads (int): Number of attention heads.
        d_ff (int): Feed-forward hidden dimension (typically 4 × d_model).
        dropout (float): Dropout rate.
        bias (bool): Whether to use bias in linear layers.
        max_seq_len (int): Maximum sequence length.
        norm_eps (float): Epsilon for LayerNorm numerical stability.
        activation (str): Activation in FFN ('gelu' or 'relu').
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 512,
        norm_eps: float = 1e-5,
        activation: str = "gelu",
    ):
        super().__init__()

        # --- LayerNorm before Attention (Pre-Norm) ---
        self.ln1 = nn.LayerNorm(d_model, eps=norm_eps)

        # --- Multi-Head Causal Self-Attention ---
        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            max_seq_len=max_seq_len,
        )

        # --- LayerNorm before FFN (Pre-Norm) ---
        self.ln2 = nn.LayerNorm(d_model, eps=norm_eps)

        # --- Feed-Forward Network ---
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)

        Flow:
            x → LN → Attention → + x  (residual)  →  a
            a → LN → FFN       → + a  (residual)  →  h
        """
        # --- Sub-block 1: Attention with residual ---
        # Pre-norm: normalize BEFORE the sublayer
        a = x + self.attention(self.ln1(x))

        # --- Sub-block 2: FFN with residual ---
        h = a + self.ffn(self.ln2(a))

        return h
