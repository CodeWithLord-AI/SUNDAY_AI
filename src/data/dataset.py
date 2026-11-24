"""
SUNDAY LLM - PyTorch Dataset
===============================

Phase 5 Implementation.

Custom PyTorch Dataset class for causal language modeling.

The dataset takes tokenized text and creates (input, target) pairs:
  Input:  [token_1, token_2, ..., token_n-1]
  Target: [token_2, token_3, ..., token_n]

Each sample is a fixed-length chunk of max_seq_len tokens.

The key insight for causal LM training:
  Given a sequence of N tokens, we can extract N-1 (input, target) pairs,
  because each token predicts the next one. This is the auto-regressive
  training objective.

Example (max_seq_len=4):
  Full token sequence: [1, 2, 3, 4, 5, 6, 7, 8]

  Chunk 0 → input: [1, 2, 3, 4], target: [2, 3, 4, 5]
  Chunk 1 → input: [5, 6, 7, 8], target: [6, 7, 8, ?] ← dropped if incomplete
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


class SundayDataset(Dataset):
    """
    PyTorch Dataset for causal language model training.

    Takes a flat list of token IDs (the entire tokenized corpus) and
    divides it into fixed-length chunks for training.

    For each chunk of length max_seq_len+1:
      - input  = chunk[:-1]  (tokens 0..n-1)
      - target = chunk[1:]   (tokens 1..n)   ← shifted by 1

    Args:
        token_ids (List[int]): Flat list of all token IDs in the corpus.
        max_seq_len (int): Length of each training sequence.
        stride (Optional[int]): Step size between chunks. If None, uses
                                max_seq_len (no overlap). Set to smaller
                                values for overlapping windows.

    Example:
        >>> token_ids = list(range(1000))
        >>> dataset = SundayDataset(token_ids, max_seq_len=128)
        >>> len(dataset)
        7
        >>> x, y = dataset[0]
        >>> x.shape, y.shape
        (torch.Size([128]), torch.Size([128]))
    """

    def __init__(
        self,
        token_ids: List[int],
        max_seq_len: int,
        stride: Optional[int] = None,
    ):
        self.max_seq_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len

        # Store token IDs as a tensor for efficient slicing
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        total_tokens = len(self.token_ids)

        # Pre-compute all valid start indices
        # We need at least max_seq_len+1 tokens per chunk (input + target)
        self.start_indices = list(
            range(0, total_tokens - max_seq_len, self.stride)
        )

        print(
            f"[SundayDataset] {total_tokens:,} tokens → "
            f"{len(self.start_indices):,} chunks "
            f"(seq_len={max_seq_len}, stride={self.stride})"
        )

    def __len__(self) -> int:
        """Return number of training chunks."""
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (input, target) training pair.

        Args:
            idx: Index of the chunk.

        Returns:
            Tuple of (input_ids, target_ids), each of shape (max_seq_len,).

        The input is shifted by 1 position from the target, implementing
        the causal language modeling objective: predict next token from context.
        """
        start = self.start_indices[idx]
        end = start + self.max_seq_len + 1  # +1 to get both input and target

        chunk = self.token_ids[start:end]

        input_ids = chunk[:-1]   # Tokens 0..n-1
        target_ids = chunk[1:]   # Tokens 1..n (shifted right by 1)

        return input_ids, target_ids


class SundayDatasetFromFile(Dataset):
    """
    Load a pre-tokenized binary file (.bin) for efficient training.

    For large corpora, it's efficient to pre-tokenize and store as a
    memory-mapped binary file. This avoids loading everything into RAM.

    Format: flat array of uint16 token IDs (since vocab_size=32768 fits in uint16).

    Usage with prepare_data.py:
        # Prepare:
        token_ids = tokenizer.encode_batch(texts)
        flat = [t for seq in token_ids for t in seq]
        arr = np.array(flat, dtype=np.uint16)
        arr.tofile("data/train.bin")

        # Load:
        dataset = SundayDatasetFromFile("data/train.bin", max_seq_len=512)
    """

    def __init__(
        self,
        filepath: str,
        max_seq_len: int,
        stride: Optional[int] = None,
    ):
        import numpy as np

        self.max_seq_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len

        # Memory-mapped array (doesn't load all data into RAM)
        self.data = np.memmap(filepath, dtype=np.uint16, mode="r")
        total_tokens = len(self.data)

        self.start_indices = list(
            range(0, total_tokens - max_seq_len, self.stride)
        )

        print(
            f"[SundayDatasetFromFile] {filepath}: "
            f"{total_tokens:,} tokens → "
            f"{len(self.start_indices):,} chunks"
        )

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        start = self.start_indices[idx]
        end = start + self.max_seq_len + 1

        # Read from memory map and convert to torch tensor
        chunk = torch.from_numpy(
            self.data[start:end].astype(np.int64)
        )

        input_ids = chunk[:-1]
        target_ids = chunk[1:]

        return input_ids, target_ids
