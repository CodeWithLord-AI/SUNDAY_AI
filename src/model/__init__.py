"""
SUNDAY Model Package
====================

Exports the main model components for easy import.

Usage:
    from src.model import SundayModel
    from src.model import CausalSelfAttention, FeedForward, SundayEmbeddings, TransformerBlock
    from src.model.utils import count_parameters, print_model_info
"""

from src.model.attention import CausalSelfAttention
from src.model.feed_forward import FeedForward
from src.model.embeddings import SundayEmbeddings
from src.model.transformer_block import TransformerBlock
from src.model.sunday_model import SundayModel
from src.model.utils import count_parameters, print_model_info, init_weights

__all__ = [
    "SundayModel",
    "CausalSelfAttention",
    "FeedForward",
    "SundayEmbeddings",
    "TransformerBlock",
    "count_parameters",
    "print_model_info",
    "init_weights",
]
