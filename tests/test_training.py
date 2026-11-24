"""
SUNDAY LLM - Training Pipeline Tests
======================================

Unit tests for optimizer, LR schedule, and checkpoint management (Phase 6).

Run with:
  cd e:\\AI&ML\\Project\\SUNDAY_AI
  .venv\\Scripts\\python.exe -m pytest tests/test_training.py -v
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_linear():
    """A trivially small linear model for optimizer tests."""
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.LayerNorm(64),
        nn.Linear(64, 32),
    )
    return model


@pytest.fixture
def tiny_model_and_config():
    from src.model.sunday_model import SundayModel
    config = {
        "model_name": "SUNDAY-TestTiny",
        "vocab_size": 256,
        "max_seq_len": 16,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,
        "activation": "gelu",
        "norm_type": "pre_norm",
        "bias": False,
        "weight_tying": True,
        "norm_eps": 1e-5,
    }
    model = SundayModel(config)
    return model, config


# ---------------------------------------------------------------------------
# configure_optimizer
# ---------------------------------------------------------------------------

class TestConfigureOptimizer:
    def test_import(self):
        from src.training.optimizer import configure_optimizer  # noqa

    def test_returns_adamw(self, tiny_linear):
        from src.training.optimizer import configure_optimizer
        opt = configure_optimizer(
            model=tiny_linear,
            learning_rate=1e-3,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
        )
        assert isinstance(opt, torch.optim.AdamW)

    def test_param_groups(self, tiny_linear):
        """Should have 2 groups: decay and no-decay."""
        from src.training.optimizer import configure_optimizer
        opt = configure_optimizer(
            model=tiny_linear,
            learning_rate=1e-3,
            weight_decay=0.1,
        )
        assert len(opt.param_groups) == 2

    def test_no_decay_for_layernorm(self, tiny_linear):
        """LayerNorm parameters must be in the no-decay group."""
        from src.training.optimizer import configure_optimizer
        opt = configure_optimizer(tiny_linear, learning_rate=1e-3, weight_decay=0.1)
        no_decay_group = opt.param_groups[1]  # convention: no-decay is second
        # LayerNorm has 2 params (weight + bias). They should land in no-decay.
        assert no_decay_group["weight_decay"] == 0.0


# ---------------------------------------------------------------------------
# get_cosine_lr
# ---------------------------------------------------------------------------

class TestGetCosineLR:
    def test_import(self):
        from src.training.optimizer import get_cosine_lr  # noqa

    def test_warmup_linear_increase(self):
        from src.training.optimizer import get_cosine_lr
        warmup = 100
        # During warmup, LR should increase linearly
        lr_0  = get_cosine_lr(step=0,  max_lr=1e-3, min_lr=1e-5, warmup_steps=warmup, max_steps=1000)
        lr_50 = get_cosine_lr(step=50, max_lr=1e-3, min_lr=1e-5, warmup_steps=warmup, max_steps=1000)
        lr_100= get_cosine_lr(step=100,max_lr=1e-3, min_lr=1e-5, warmup_steps=warmup, max_steps=1000)
        assert lr_0 <= lr_50 <= lr_100, "LR should increase linearly during warmup"

    def test_peak_at_warmup_end(self):
        from src.training.optimizer import get_cosine_lr
        warmup, max_lr = 100, 1e-3
        lr = get_cosine_lr(step=warmup, max_lr=max_lr, min_lr=1e-5, warmup_steps=warmup, max_steps=1000)
        assert abs(lr - max_lr) < 1e-8, f"LR at warmup end should be max_lr, got {lr}"

    def test_cosine_decay_after_warmup(self):
        from src.training.optimizer import get_cosine_lr
        warmup, max_lr, min_lr = 100, 1e-3, 1e-5
        lr_200 = get_cosine_lr(500,  max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup, max_steps=1000)
        lr_800 = get_cosine_lr(900,  max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup, max_steps=1000)
        assert lr_200 > lr_800, "LR should decrease after warmup"

    def test_min_lr_at_end(self):
        from src.training.optimizer import get_cosine_lr
        min_lr = 1e-5
        lr = get_cosine_lr(1000, max_lr=1e-3, min_lr=min_lr, warmup_steps=100, max_steps=1000)
        assert lr >= min_lr - 1e-9, "LR should not go below min_lr"


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_import(self):
        from src.training.checkpoint import save_checkpoint, load_checkpoint  # noqa

    def test_save_and_load_roundtrip(self, tiny_model_and_config):
        from src.training.checkpoint import save_checkpoint, load_checkpoint

        model, config = tiny_model_and_config
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=42,
                epoch=1,
                config=config,
                train_loss=1.5,
                val_loss=1.8,
                best_val_loss=1.8,
                save_dir=tmpdir,
                keep_last_n=3,
            )
            assert os.path.exists(path), f"Checkpoint file not found at {path}"

            # Load into fresh model
            from src.model.sunday_model import SundayModel
            model2  = SundayModel(config)
            opt2    = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            state   = load_checkpoint(path, model=model2, optimizer=opt2, device="cpu")

            assert state["step"] == 42
            assert abs(state.get("train_loss", state.get("loss", 0)) - 1.5) < 1e-5

    def test_model_weights_restored(self, tiny_model_and_config):
        from src.training.checkpoint import save_checkpoint, load_checkpoint
        from src.model.sunday_model import SundayModel

        model, config = tiny_model_and_config

        # Grab weights before save
        before = {k: v.clone() for k, v in model.state_dict().items()}

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(
                model=model, optimizer=optimizer,
                step=1, epoch=0, config=config,
                train_loss=0.0, val_loss=0.0, best_val_loss=0.0,
                save_dir=tmpdir,
            )
            # Fresh model (random weights)
            model2 = SundayModel(config)
            opt2   = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            load_checkpoint(path, model=model2, optimizer=opt2, device="cpu")

            after = model2.state_dict()
            for k in before:
                assert torch.allclose(before[k].float(), after[k].float(), atol=1e-6), \
                    f"Weight mismatch after load for key: {k}"

    def test_find_latest_checkpoint(self, tiny_model_and_config):
        from src.training.checkpoint import save_checkpoint, find_latest_checkpoint
        from src.model.sunday_model import SundayModel

        model, config = tiny_model_and_config
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [100, 200, 300]:
                save_checkpoint(
                    model=model, optimizer=optimizer,
                    step=step, epoch=0, config=config,
                    train_loss=1.0, val_loss=1.0, best_val_loss=1.0,
                    save_dir=tmpdir, keep_last_n=5,
                )
            latest = find_latest_checkpoint(tmpdir)
            assert latest is not None
            assert "300" in os.path.basename(latest)
