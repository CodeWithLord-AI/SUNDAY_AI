"""
SUNDAY LLM - Model Tests
===========================

Unit tests for the Transformer model components (Phase 4).

Run with:
  cd e:\\AI&ML\\Project\\SUNDAY_AI
  .venv\\Scripts\\python.exe -m pytest tests/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_config():
    """Minimal model config for fast unit tests."""
    return {
        "model_name": "SUNDAY-TestTiny",
        "vocab_size": 256,
        "max_seq_len": 32,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "dropout": 0.0,       # 0.0 for deterministic tests
        "activation": "gelu",
        "norm_type": "pre_norm",
        "bias": False,
        "weight_tying": True,
        "norm_eps": 1e-5,
    }


@pytest.fixture(scope="module")
def model(tiny_config):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.model.sunday_model import SundayModel
    m = SundayModel(tiny_config)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# CausalSelfAttention
# ---------------------------------------------------------------------------

class TestCausalSelfAttention:
    def test_import(self):
        from src.model.attention import CausalSelfAttention  # noqa

    def test_output_shape(self, tiny_config):
        from src.model.attention import CausalSelfAttention
        attn = CausalSelfAttention(
            d_model=tiny_config["d_model"],
            n_heads=tiny_config["n_heads"],
            dropout=0.0,
            bias=False,
        )
        attn.eval()
        B, T, D = 2, 16, tiny_config["d_model"]
        x = torch.randn(B, T, D)
        out = attn(x)
        assert out.shape == (B, T, D), f"Expected {(B,T,D)}, got {out.shape}"

    def test_causal_mask(self, tiny_config):
        """Future tokens must not attend to current position."""
        from src.model.attention import CausalSelfAttention
        torch.manual_seed(0)
        attn = CausalSelfAttention(
            d_model=tiny_config["d_model"],
            n_heads=tiny_config["n_heads"],
            dropout=0.0,
            bias=False,
        )
        attn.eval()
        T = 8
        x = torch.randn(1, T, tiny_config["d_model"])
        out1 = attn(x)

        # Corrupt future tokens — output at position 0 must not change
        x_corrupt = x.clone()
        x_corrupt[0, 1:, :] = torch.randn(T - 1, tiny_config["d_model"])
        out2 = attn(x_corrupt)

        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5), (
            "Position 0 output changed when future tokens were modified — causal mask broken!"
        )


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class TestFeedForward:
    def test_import(self):
        from src.model.feed_forward import FeedForward  # noqa

    def test_output_shape(self, tiny_config):
        from src.model.feed_forward import FeedForward
        ff = FeedForward(
            d_model=tiny_config["d_model"],
            d_ff=tiny_config["d_ff"],
            dropout=0.0,
            bias=False,
            activation="gelu",
        )
        ff.eval()
        B, T, D = 2, 16, tiny_config["d_model"]
        x = torch.randn(B, T, D)
        out = ff(x)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# SundayEmbeddings
# ---------------------------------------------------------------------------

class TestSundayEmbeddings:
    def test_output_shape(self, tiny_config):
        from src.model.embeddings import SundayEmbeddings
        emb = SundayEmbeddings(
            vocab_size=tiny_config["vocab_size"],
            max_seq_len=tiny_config["max_seq_len"],
            d_model=tiny_config["d_model"],
            dropout=0.0,
        )
        emb.eval()
        ids = torch.randint(0, tiny_config["vocab_size"], (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, tiny_config["d_model"])


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def test_output_shape(self, tiny_config):
        from src.model.transformer_block import TransformerBlock
        block = TransformerBlock(
            d_model=tiny_config["d_model"],
            n_heads=tiny_config["n_heads"],
            d_ff=tiny_config["d_ff"],
            dropout=0.0,
            bias=False,
            norm_eps=1e-5,
            activation="gelu",
        )
        block.eval()
        B, T, D = 2, 16, tiny_config["d_model"]
        x = torch.randn(B, T, D)
        out = block(x)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# SundayModel (full model)
# ---------------------------------------------------------------------------

class TestSundayModel:
    def test_instantiation(self, model):
        assert model is not None

    def test_parameter_count(self, model):
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n > 0, "Model has no parameters"

    def test_forward_logits_shape(self, model, tiny_config):
        B, T = 2, 16
        ids = torch.randint(0, tiny_config["vocab_size"], (B, T))
        with torch.no_grad():
            out = model(ids)
        # forward may return (loss, logits) or just logits depending on implementation
        if isinstance(out, tuple):
            logits = out[1]
        else:
            logits = out
        assert logits.shape == (B, T, tiny_config["vocab_size"]), \
            f"Expected ({B}, {T}, {tiny_config['vocab_size']}), got {logits.shape}"

    def test_forward_with_targets(self, model, tiny_config):
        B, T = 2, 16
        ids     = torch.randint(0, tiny_config["vocab_size"], (B, T))
        targets = ids.clone()
        with torch.no_grad():
            result = model(ids, targets=targets)
        assert isinstance(result, tuple) and len(result) == 2, \
            "With targets, model should return (loss, logits)"
        loss, logits = result
        assert loss.ndim == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"

    def test_weight_tying(self, model):
        """LM head weights must share storage with token embedding."""
        emb_weight  = model.embeddings.token_embedding.weight
        head_weight = model.lm_head.weight
        assert emb_weight.data_ptr() == head_weight.data_ptr(), \
            "Weight tying broken — embedding and LM head use different tensors"

    def test_generate_shape(self, model, tiny_config):
        ids = torch.randint(0, tiny_config["vocab_size"], (1, 4))
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=5)
        assert out.shape[0] == 1
        assert out.shape[1] == 4 + 5, \
            f"Expected {4+5} tokens after generate, got {out.shape[1]}"

    def test_gradient_flow(self, tiny_config):
        """Gradients should flow to all parameter groups."""
        from src.model.sunday_model import SundayModel
        model_grad = SundayModel(tiny_config)
        model_grad.train()

        ids     = torch.randint(0, tiny_config["vocab_size"], (1, 8))
        targets = ids.clone()
        loss, _ = model_grad(ids, targets=targets)
        loss.backward()

        for name, p in model_grad.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
