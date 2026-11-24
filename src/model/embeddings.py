"""
SUNDAY LLM - Embeddings
=========================

Phase 4 Implementation.

Two types of embeddings in SUNDAY:

1. Token Embeddings:
   - Converts token IDs → dense vectors
   - nn.Embedding(vocab_size=32768, d_model=768)
   - Each token has a unique 768-dimensional vector
   - Parameters: 32768 × 768 = 25,165,824

2. Positional Embeddings:
   - Tells the model WHERE each token is in the sequence
   - nn.Embedding(max_seq_len=512, d_model=768)
   - Learned (not sinusoidal) — GPT-2 style
   - Parameters: 512 × 768 = 393,216

Combined:
  h = TokenEmbed(token_ids) + PosEmbed(positions) + Dropout

Why addition (not concatenation)?
  Addition keeps the dimension at d_model.
  The model learns to combine word meaning + position
  information in the same vector space.
"""

import torch
import torch.nn as nn


class SundayEmbeddings(nn.Module):
    """
    Token + Learned Positional Embeddings for SUNDAY LLM.

    Takes integer token IDs and returns continuous embedding vectors
    with position information added.

    Args:
        vocab_size (int): Number of tokens in the vocabulary (32768).
        max_seq_len (int): Maximum sequence length the model supports.
        d_model (int): Embedding dimension.
        dropout (float): Dropout rate applied after combining embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # --- Token Embedding Table ---
        # Maps each token ID → d_model-dimensional vector
        # Shape: (vocab_size, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # --- Positional Embedding Table ---
        # Maps each position index (0, 1, ..., max_seq_len-1) → d_model vector
        # GPT-2 uses LEARNED positional embeddings (not sinusoidal)
        # Shape: (max_seq_len, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # --- Dropout ---
        # Applied after combining token + position embeddings
        self.dropout = nn.Dropout(dropout)

        # --- Pre-build position indices ---
        # We register position indices as a buffer (not a parameter)
        # This avoids creating them on every forward pass and
        # ensures they move to the right device automatically
        positions = torch.arange(max_seq_len).unsqueeze(0)  # (1, max_seq_len)
        self.register_buffer("position_indices", positions)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embedded vectors with position information.

        Args:
            token_ids: Integer tensor of shape (batch_size, seq_len)
                       Values should be in range [0, vocab_size)

        Returns:
            Embedding tensor of shape (batch_size, seq_len, d_model)

        Example:
            >>> embed = SundayEmbeddings(32768, 512, 768)
            >>> ids = torch.randint(0, 32768, (2, 64))  # batch=2, seq=64
            >>> out = embed(ids)  # (2, 64, 768)
        """
        B, T = token_ids.shape

        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        )

        # --- Token Embeddings ---
        # (B, T) → (B, T, d_model)
        tok_emb = self.token_embedding(token_ids)

        # --- Positional Embeddings ---
        # Use only the first T positions: (1, T, d_model)
        pos_emb = self.position_embedding(self.position_indices[:, :T])
        # pos_emb broadcasts over batch dimension automatically

        # --- Combine & Dropout ---
        # Addition: both tensors are (B, T, d_model) after broadcasting
        embeddings = tok_emb + pos_emb
        embeddings = self.dropout(embeddings)

        return embeddings

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, "
            f"max_seq_len={self.max_seq_len}, "
            f"d_model={self.d_model}"
        )
