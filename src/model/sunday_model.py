"""
SUNDAY LLM - Full Model Assembly
===================================

Phase 4 Implementation.

The SundayModel class assembles all components into the complete
GPT-style decoder-only Transformer:

  Input IDs → Embeddings → N × TransformerBlock → LayerNorm → LM Head → Logits

Components:
  1. Token + Positional Embeddings (embeddings.py)
  2. N × Transformer Blocks (transformer_block.py)
     Each block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
  3. Final Layer Normalization
  4. Language Modeling Head (linear projection to vocab_size)
     Weight-tied with token embeddings

Weight Initialization:
  - Linear layers: Normal(0, 0.02)
  - Embeddings: Normal(0, 0.02)
  - LayerNorm: gamma=1, beta=0
  - Residual projections scaled by 1/sqrt(2*n_layers)
  (Following GPT-2 initialization scheme)

Usage:
  model = SundayModel(config)
  logits = model(input_ids)  # (batch, seq_len, vocab_size)
  loss = model(input_ids, targets=target_ids)  # returns loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from configs.model_config import get_model_config
from src.model.embeddings import SundayEmbeddings
from src.model.transformer_block import TransformerBlock
from src.model.utils import init_weights, scale_residual_projections, print_model_info


class SundayModel(nn.Module):
    """
    SUNDAY GPT-style Decoder-Only Language Model.

    Takes token IDs as input and outputs logits over the vocabulary.
    Optionally computes cross-entropy loss when targets are provided.

    Args:
        config (dict): Model hyperparameters. Must contain:
            vocab_size, max_seq_len, d_model, n_heads, n_layers,
            d_ff, dropout, bias, weight_tying, norm_eps, activation.

    Example:
        >>> from configs.model_config import get_model_config
        >>> config = get_model_config("tiny")
        >>> model = SundayModel(config)
        >>> input_ids = torch.randint(0, 32768, (2, 64))
        >>> logits = model(input_ids)             # (2, 64, 32768)
        >>> loss, logits = model(input_ids, targets=input_ids[:, 1:])
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # --- Extract config values ---
        vocab_size  = config["vocab_size"]
        max_seq_len = config["max_seq_len"]
        d_model     = config["d_model"]
        n_heads     = config["n_heads"]
        n_layers    = config["n_layers"]
        d_ff        = config["d_ff"]
        dropout     = config["dropout"]
        bias        = config["bias"]
        norm_eps    = config.get("norm_eps", 1e-5)
        activation  = config.get("activation", "gelu")
        self.weight_tying = config.get("weight_tying", True)

        # =========================================================
        # 1. Embeddings
        # =========================================================
        self.embeddings = SundayEmbeddings(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            dropout=dropout,
        )

        # =========================================================
        # 2. Transformer Blocks (stacked N times)
        # =========================================================
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                bias=bias,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
                activation=activation,
            )
            for _ in range(n_layers)
        ])

        # =========================================================
        # 3. Final Layer Normalization
        # =========================================================
        self.final_ln = nn.LayerNorm(d_model, eps=norm_eps)

        # =========================================================
        # 4. Language Modeling Head
        # =========================================================
        # Projects d_model → vocab_size (no bias is standard in GPT)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # =========================================================
        # 5. Weight Tying
        # =========================================================
        # Share token embedding weights with the LM head.
        # This reduces parameters by vocab_size × d_model (25M for 125M model!)
        # and often improves performance (Press & Wolf, 2017).
        if self.weight_tying:
            # lm_head.weight = token embedding weights (same tensor)
            self.lm_head.weight = self.embeddings.token_embedding.weight

        # =========================================================
        # 6. Initialize Weights (GPT-2 style)
        # =========================================================
        self.apply(lambda m: init_weights(m, n_layers))
        scale_residual_projections(self, n_layers)

        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[SUNDAY] Model initialized: {total_params/1e6:.2f}M parameters")

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the full SUNDAY model.

        Args:
            input_ids: Integer token IDs, shape (batch_size, seq_len).
                       Values in range [0, vocab_size).
            targets: Optional target token IDs for loss computation.
                     Shape (batch_size, seq_len).
                     If provided, returns (loss, logits).
                     If None, returns logits only.

        Returns:
            If targets is None:
                logits: (batch_size, seq_len, vocab_size)
            If targets is provided:
                (loss, logits): (scalar, (batch_size, seq_len, vocab_size))

        Example:
            # Inference (no targets)
            logits = model(input_ids)       # (B, T, V)

            # Training (with targets)
            loss, logits = model(input_ids, targets=shifted_ids)
        """
        # ---- Step 1: Embeddings ----
        # (B, T) → (B, T, d_model)
        x = self.embeddings(input_ids)

        # ---- Step 2: Transformer Blocks ----
        for block in self.blocks:
            x = block(x)
        # x: (B, T, d_model)

        # ---- Step 3: Final LayerNorm ----
        x = self.final_ln(x)
        # x: (B, T, d_model)

        # ---- Step 4: LM Head → Logits ----
        logits = self.lm_head(x)
        # logits: (B, T, vocab_size)

        # ---- Step 5: Loss (if targets provided) ----
        if targets is not None:
            # Cross-entropy loss
            # logits: (B, T, V) → flatten to (B*T, V)
            # targets: (B, T)   → flatten to (B*T,)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
                ignore_index=-1,  # -1 = padding positions to ignore
            )
            return loss, logits

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Auto-regressively generate new tokens.

        A simple greedy/top-k sampler built into the model for quick testing.
        (Full sampling logic lives in src/inference/sampling.py)

        Args:
            input_ids: Starting token IDs, shape (1, prompt_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (1.0 = no change, <1 = sharper).
            top_k: If set, sample from top-k most probable tokens.
            eos_token_id: If set, stop generation when this token is produced.

        Returns:
            Tensor of shape (1, prompt_len + generated_len).
        """
        self.eval()
        max_seq_len = self.config["max_seq_len"]

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            context = input_ids[:, -max_seq_len:]

            # Forward pass
            logits = self(context)              # (1, T, vocab_size)
            logits = logits[:, -1, :]           # Last token: (1, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                # Zero out all logits below the k-th largest
                values, _ = torch.topk(logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_val, float("-inf"))

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS token generated
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids

    @classmethod
    def from_config(cls, size: str = "50m") -> "SundayModel":
        """
        Create a SundayModel from a named configuration.

        Args:
            size: One of 'tiny', '50m', '125m'.

        Returns:
            Initialized SundayModel.

        Example:
            >>> model = SundayModel.from_config("tiny")
        """
        config = get_model_config(size)
        return cls(config)

    def count_parameters(self) -> dict:
        """
        Count and return parameter statistics.

        Returns:
            Dict with 'total', 'trainable', 'non_trainable'.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
        }

    def print_info(self) -> None:
        """Print model architecture and parameter summary."""
        print_model_info(self, self.config)
