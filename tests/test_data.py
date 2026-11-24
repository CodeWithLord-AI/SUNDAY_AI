"""
SUNDAY LLM - Data Pipeline Tests
===================================

Unit tests for the dataset, dataloader, and preprocessing modules (Phase 5).

Run with:
  cd e:\\AI&ML\\Project\\SUNDAY_AI
  .venv\\Scripts\\python.exe -m pytest tests/test_data.py -v
"""

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# SundayDataset
# ---------------------------------------------------------------------------

class TestSundayDataset:
    def test_import(self):
        from src.data.dataset import SundayDataset  # noqa

    def test_len(self):
        from src.data.dataset import SundayDataset
        tokens = list(range(1000))
        ds = SundayDataset(tokens, max_seq_len=128)
        assert len(ds) > 0

    def test_item_shape(self):
        from src.data.dataset import SundayDataset
        tokens = list(range(1000))
        ds = SundayDataset(tokens, max_seq_len=64)
        x, y = ds[0]
        assert x.shape == torch.Size([64]), f"x.shape={x.shape}"
        assert y.shape == torch.Size([64]), f"y.shape={y.shape}"

    def test_target_is_shifted(self):
        from src.data.dataset import SundayDataset
        tokens = list(range(200))
        ds = SundayDataset(tokens, max_seq_len=10)
        x, y = ds[0]
        # y should be x shifted left by 1 (target = next token)
        assert torch.all(y[:-1] == x[1:]), "Target is not correctly shifted from input"

    def test_no_overlap_stride(self):
        from src.data.dataset import SundayDataset
        seq_len = 32
        tokens = list(range(seq_len * 5))
        ds = SundayDataset(tokens, max_seq_len=seq_len)
        # With stride=seq_len, there should be ~5 chunks (last may be dropped)
        assert len(ds) >= 4

    def test_dtype(self):
        from src.data.dataset import SundayDataset
        tokens = list(range(500))
        ds = SundayDataset(tokens, max_seq_len=32)
        x, y = ds[0]
        assert x.dtype == torch.long, f"Expected torch.long, got {x.dtype}"


# ---------------------------------------------------------------------------
# SundayDatasetFromFile (binary)
# ---------------------------------------------------------------------------

class TestSundayDatasetFromFile:
    def test_roundtrip(self):
        """Save tokens as binary, load them back, check shapes."""
        from src.data.dataset import SundayDatasetFromFile
        import numpy as np
        import tempfile

        tokens = list(range(1000))
        # Use a named temp file instead of a directory to avoid Windows locking issues
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            bin_path = f.name

        try:
            arr = np.array(tokens, dtype=np.uint16)
            arr.tofile(bin_path)

            ds = SundayDatasetFromFile(bin_path, max_seq_len=32)
            assert len(ds) > 0
            x, y = ds[0]
            assert x.shape == torch.Size([32])
            assert y.shape == torch.Size([32])
            # Close the memmap explicitly before cleanup
            del ds
        finally:
            try:
                os.unlink(bin_path)
            except Exception:
                pass  # Ignore cleanup errors on Windows


# ---------------------------------------------------------------------------
# create_dataloaders
# ---------------------------------------------------------------------------

class TestCreateDataloaders:
    def test_basic(self):
        from src.data.dataset import SundayDataset
        from src.data.dataloader import create_dataloaders

        tokens = list(range(5000))
        seq_len = 32
        # create_dataloaders takes a single dataset and splits it (val_split=0.1)
        ds = SundayDataset(tokens, max_seq_len=seq_len)
        train_loader, val_loader = create_dataloaders(
            dataset=ds,
            batch_size=4,
            val_split=0.1,
            num_workers=0,
            pin_memory=False,
        )
        assert train_loader is not None
        assert val_loader is not None

    def test_batch_shape(self):
        from src.data.dataset import SundayDataset
        from src.data.dataloader import create_dataloaders

        tokens = list(range(5000))
        seq_len = 32
        ds = SundayDataset(tokens, max_seq_len=seq_len)
        loader, _ = create_dataloaders(
            dataset=ds,
            batch_size=4,
            val_split=0.1,
            num_workers=0,
        )
        x, y = next(iter(loader))
        assert x.shape[0] <= 4          # batch size
        assert x.shape[1] == seq_len    # seq len
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_import(self):
        from src.data.preprocessing import clean_text  # noqa

    def test_clean_text_strips_whitespace(self):
        from src.data.preprocessing import clean_text
        result = clean_text("  Hello   World  ")
        assert "  " not in result
        assert result.strip() == result

    def test_clean_text_returns_string(self):
        from src.data.preprocessing import clean_text
        result = clean_text("Hello world")
        assert isinstance(result, str)

    def test_clean_text_handles_empty(self):
        from src.data.preprocessing import clean_text
        result = clean_text("")
        assert isinstance(result, str)
