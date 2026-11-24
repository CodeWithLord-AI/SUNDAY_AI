"""
SUNDAY LLM - DataLoader Factory
==================================

Phase 5 Implementation.

Creates PyTorch DataLoaders for training and validation.

Key concerns:
  - Shuffling: Shuffle training data each epoch (not validation)
  - Batching: Group sequences into batches for parallel processing
  - num_workers: Parallel data loading (set 0 on Windows, 2-4 on Linux)
  - pin_memory: Faster CPU→GPU transfer (enable when using CUDA)
  - prefetch_factor: Pre-load batches in background

The DataLoader returns tensors of shape:
  input_ids:  (batch_size, max_seq_len)
  target_ids: (batch_size, max_seq_len)
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional, Tuple


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split a dataset into train/val and create DataLoaders.

    Args:
        dataset: A SundayDataset instance.
        batch_size: Number of sequences per batch.
        val_split: Fraction of data to use for validation (0.0-1.0).
        num_workers: Parallel workers for data loading.
                     Use 0 on Windows, 2-4 on Linux/Colab.
        pin_memory: Pin tensors in CPU memory for faster GPU transfer.
                    Enable when training on GPU.
        seed: Random seed for reproducible train/val split.

    Returns:
        (train_loader, val_loader): Both are DataLoader instances.

    Example:
        >>> train_loader, val_loader = create_dataloaders(
        ...     dataset, batch_size=32, val_split=0.1
        ... )
        >>> for input_ids, target_ids in train_loader:
        ...     loss, logits = model(input_ids, targets=target_ids)
    """
    # --- Split into train / val ---
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    print(
        f"[DataLoader] Split: {train_size:,} train / {val_size:,} val "
        f"(batch_size={batch_size})"
    )

    # --- Training DataLoader (shuffle=True) ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for consistent shapes
    )

    # --- Validation DataLoader (shuffle=False) ---
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def create_dataloader_from_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders from pre-split train and val datasets.

    Use this when you have separate train.bin and val.bin files.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        batch_size: Sequences per batch.
        num_workers: Worker processes for data loading.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    print(
        f"[DataLoader] Train: {len(train_dataset):,} samples → "
        f"{n_train_batches:,} batches"
    )
    print(
        f"[DataLoader] Val:   {len(val_dataset):,} samples → "
        f"{n_val_batches:,} batches"
    )

    return train_loader, val_loader
