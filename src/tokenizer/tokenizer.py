"""
SUNDAY LLM - Tokenizer Wrapper
================================

A clean wrapper around the HuggingFace tokenizer that provides
easy-to-use encode/decode methods for the SUNDAY model.

Why a wrapper?
  The raw HuggingFace tokenizer returns Encoding objects with many
  fields. Our model only needs token IDs. This wrapper simplifies
  the interface and handles:
  - Loading the trained tokenizer
  - Adding <bos>/<eos> tokens
  - Padding sequences to max_seq_len
  - Batch encoding for DataLoader
  - Decoding back to human-readable text

Usage:
  >>> from src.tokenizer.tokenizer import SundayTokenizer
  >>> tok = SundayTokenizer.from_file("tokenizer_files/sunday_tokenizer.json")
  >>> ids = tok.encode("Hello world!")
  >>> text = tok.decode(ids)
  >>> print(text)  # "Hello world!"
"""

import os
from typing import List, Optional, Union

from tokenizers import Tokenizer


class SundayTokenizer:
    """
    Wrapper around a trained BPE tokenizer for SUNDAY LLM.

    Attributes:
        tokenizer: The underlying HuggingFace Tokenizer object.
        vocab_size: Size of the vocabulary.
        pad_id: Token ID for <pad>.
        unk_id: Token ID for <unk>.
        bos_id: Token ID for <bos> (beginning of sequence).
        eos_id: Token ID for <eos> (end of sequence).
        sep_id: Token ID for <sep> (separator).
    """

    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize from a HuggingFace Tokenizer object.
        Usually you'll use from_file() instead.
        """
        self.tokenizer = tokenizer

        # Cache special token IDs
        self.pad_id = self._get_token_id("<pad>")
        self.unk_id = self._get_token_id("<unk>")
        self.bos_id = self._get_token_id("<bos>")
        self.eos_id = self._get_token_id("<eos>")
        self.sep_id = self._get_token_id("<sep>")

        # Enable padding
        self.tokenizer.enable_padding(
            pad_id=self.pad_id,
            pad_token="<pad>",
        )

    def _get_token_id(self, token: str) -> Optional[int]:
        """Get the ID for a token, or None if not found."""
        return self.tokenizer.token_to_id(token)

    @classmethod
    def from_file(cls, path: str) -> "SundayTokenizer":
        """
        Load a trained tokenizer from a JSON file.

        Args:
            path: Path to the tokenizer JSON file.

        Returns:
            SundayTokenizer instance.

        Example:
            >>> tok = SundayTokenizer.from_file("tokenizer_files/sunday_tokenizer.json")
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Encode text to a list of token IDs.

        Args:
            text: Input text string.
            add_bos: Whether to prepend <bos> token.
            add_eos: Whether to append <eos> token.

        Returns:
            List of integer token IDs.

        Example:
            >>> tok.encode("Hello world!")
            [2, 15496, 995, 0, 3]
            #  ^bos   ^tokens     ^eos
        """
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids

        # Add special tokens
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of integer token IDs.
            skip_special_tokens: Whether to remove <bos>, <eos>, <pad>, etc.

        Returns:
            Decoded text string.

        Example:
            >>> tok.decode([2, 15496, 995, 0, 3])
            "Hello world!"
        """
        # Filter out special tokens if requested
        if skip_special_tokens:
            special_ids = {self.pad_id, self.bos_id, self.eos_id, self.sep_id}
            special_ids.discard(None)  # Remove None values
            ids = [i for i in ids if i not in special_ids]

        return self.tokenizer.decode(ids)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Encode a batch of texts to token IDs.

        Args:
            texts: List of text strings.
            add_bos: Whether to prepend <bos> token.
            add_eos: Whether to append <eos> token.
            max_length: Maximum sequence length (truncates if longer).

        Returns:
            List of lists of token IDs.
        """
        results = []
        for text in texts:
            ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            if max_length is not None:
                ids = ids[:max_length]
            results.append(ids)
        return results

    def decode_batch(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token ID sequences back to text.

        Args:
            batch_ids: List of lists of token IDs.
            skip_special_tokens: Whether to remove special tokens.

        Returns:
            List of decoded text strings.
        """
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]

    def get_tokens(self, text: str) -> List[str]:
        """
        Get the actual token strings (not IDs) for a text.
        Useful for debugging and understanding tokenization.

        Example:
            >>> tok.get_tokens("Hello world!")
            ['Hello', 'Ġworld', '!']
        """
        return self.tokenizer.encode(text).tokens

    def token_to_id(self, token: str) -> Optional[int]:
        """Get the ID for a specific token."""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        """Get the token string for a specific ID."""
        return self.tokenizer.id_to_token(id)

    def get_vocab(self) -> dict:
        """Get the full vocabulary as a {token: id} dictionary."""
        return self.tokenizer.get_vocab()

    def __len__(self) -> int:
        """Return vocab size (for convenience)."""
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"SundayTokenizer("
            f"vocab_size={self.vocab_size:,}, "
            f"pad_id={self.pad_id}, "
            f"bos_id={self.bos_id}, "
            f"eos_id={self.eos_id})"
        )
