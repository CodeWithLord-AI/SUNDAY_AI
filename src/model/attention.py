"""
SUNDAY LLM - Multi-Head Causal Self-Attention
===============================================

Phase 4 Implementation.

This is the CORE component of the Transformer.

Mathematical Formula:
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

  Where:
    Q = X @ W_Q    (queries: what am I looking for?)
    K = X @ W_K    (keys: what do I contain?)
    V = X @ W_V    (values: what information do I provide?)
    d_k = d_model / n_heads  (dimension per head)

Multi-Head:
  Split Q, K, V into n_heads parallel attention computations,
  concatenate results, and project through W_O.

Causal Masking:
  Apply upper-triangular mask (-inf) to prevent tokens from
  attending to future positions. Essential for autoregressive
  text generation.

SUNDAY Config (125M):
  d_model = 768, n_heads = 12, d_k = 64
  Parameters: 4 × 768 × 768 = 2,359,296 per layer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    This is the scaled dot-product attention with:
      - Multiple parallel attention heads (multi-head)
      - Causal mask (each token only attends to past tokens)
      - Linear projections for Q, K, V, and Output

    Args:
        d_model (int): Model embedding dimension.
        n_heads (int): Number of attention heads. Must divide d_model evenly.
        dropout (float): Dropout rate applied to attention weights.
        bias (bool): Whether to use bias in linear layers (False = GPT-2 modern style).
        max_seq_len (int): Maximum sequence length for pre-building the causal mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 512,
    ):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.dropout_p = dropout

        # --- Q, K, V projections (all in one matrix for efficiency) ---
        # W_Q, W_K, W_V: each (d_model, d_model)
        # Combined into a single (d_model, 3 * d_model) for one matmul
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        # --- Output projection ---
        # Projects concatenated heads back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # --- Dropout ---
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # --- Causal mask ---
        # Pre-build upper-triangular mask: True where we should MASK (future positions)
        # Shape: (1, 1, max_seq_len, max_seq_len) for broadcasting
        # Using register_buffer so it's part of model state but not a parameter
        mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1  # diagonal=1: mask upper triangle (future positions)
        )
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))
        # Shape: (1, 1, max_seq_len, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)

        Steps:
            1. Project x → Q, K, V
            2. Reshape into multi-head format
            3. Compute scaled dot-product attention with causal mask
            4. Concatenate heads
            5. Project output
        """
        B, T, C = x.shape  # batch_size, seq_len, d_model

        # --- Step 1: Compute Q, K, V in one shot ---
        # qkv shape: (B, T, 3 * d_model)
        qkv = self.qkv_proj(x)

        # Split into Q, K, V; each: (B, T, d_model)
        Q, K, V = qkv.split(self.d_model, dim=2)

        # --- Step 2: Reshape for multi-head ---
        # (B, T, d_model) → (B, T, n_heads, d_k) → (B, n_heads, T, d_k)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # All: (B, n_heads, T, d_k)

        # --- Step 3: Scaled Dot-Product Attention ---
        # scores: (B, n_heads, T, T)
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Apply causal mask: positions we want to mask get -inf
        # causal_mask[:, :, :T, :T] selects the right submatrix for current seq len
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T],
            float("-inf")
        )

        # Softmax to get attention weights
        # attn_weights: (B, n_heads, T, T)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        # out: (B, n_heads, T, d_k)
        out = torch.matmul(attn_weights, V)

        # --- Step 4: Concatenate heads ---
        # (B, n_heads, T, d_k) → (B, T, n_heads, d_k) → (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # --- Step 5: Output projection ---
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_k={self.d_k}, dropout={self.dropout_p}"
        )
