"""
SUNDAY Inference Package
=========================

Exports for text generation and chat.

Usage:
    from src.inference import Generator, ChatSession
    from src.inference.sampling import sample_next_token
"""

from src.inference.generator import Generator
from src.inference.chat import ChatSession
from src.inference.sampling import sample_next_token

__all__ = [
    "Generator",
    "ChatSession",
    "sample_next_token",
]
