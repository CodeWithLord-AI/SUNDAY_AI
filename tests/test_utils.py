"""
SUNDAY LLM - Utility Tests
=============================

Tests for utility functions. These run NOW (Phase 2).
"""

import pytest
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.helpers import format_number, Timer


class TestSeed:
    """Test reproducibility utilities."""

    def test_set_seed_runs_without_error(self):
        """set_seed should execute without raising."""
        set_seed(42)

    def test_set_seed_produces_reproducible_results(self):
        """Same seed should produce same random numbers."""
        import torch
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert torch.equal(a, b), "Same seed should produce same tensors"


class TestDevice:
    """Test device detection utilities."""

    def test_get_device_auto(self):
        """Auto device should return cpu or cuda without error."""
        device = get_device("auto")
        assert device.type in ("cpu", "cuda")

    def test_get_device_cpu(self):
        """Explicitly requesting CPU should work."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_invalid(self):
        """Invalid device should raise ValueError."""
        with pytest.raises(ValueError):
            get_device("tpu")


class TestHelpers:
    """Test helper functions."""

    def test_format_number_millions(self):
        assert format_number(49_500_000) == "49.5M"

    def test_format_number_billions(self):
        assert format_number(1_500_000_000) == "1.5B"

    def test_format_number_thousands(self):
        assert format_number(1200) == "1.2K"

    def test_format_number_small(self):
        assert format_number(42) == "42"

    def test_timer_context_manager(self):
        """Timer should measure elapsed time."""
        import time
        with Timer("test") as t:
            time.sleep(0.1)
        assert t.elapsed >= 0.1
