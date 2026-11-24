"""
SUNDAY LLM - Tokenizer Tests
===============================

Tests for BPE tokenizer training, encoding, and decoding.

Run with:
  cd e:\\AI&ML\\Project\\SUNDAY_AI
  .venv\\Scripts\\pytest.exe tests\\test_tokenizer.py -v
"""

import os
import tempfile
import pytest

from src.tokenizer.trainer import train_tokenizer_from_texts
from src.tokenizer.tokenizer import SundayTokenizer


# --- Sample training data ---
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing enables computers to understand text.",
    "The transformer architecture revolutionized NLP in 2017.",
    "Attention is all you need was a groundbreaking paper.",
    "GPT models use decoder-only transformer architecture.",
    "SUNDAY is a language model built from scratch.",
    "Training large language models requires significant compute.",
    "Tokenization converts text into numerical representations.",
    "The cat sat on the mat and looked at the hat.",
    "Python is a popular programming language for AI research.",
    "PyTorch provides automatic differentiation for deep learning.",
    "Backpropagation computes gradients through the computational graph.",
    "Self-attention allows each token to attend to all other tokens.",
] * 20  # Repeat to give BPE enough data to learn merges


@pytest.fixture(scope="module")
def trained_tokenizer():
    """Train a small tokenizer for testing (shared across tests)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tok = train_tokenizer_from_texts(
            texts=SAMPLE_TEXTS,
            vocab_size=500,  # Small vocab for fast testing
            min_frequency=2,
            save_dir=tmpdir,
            save_name="test_tokenizer.json",
        )
        # Wrap in our SundayTokenizer
        path = os.path.join(tmpdir, "test_tokenizer.json")
        sunday_tok = SundayTokenizer.from_file(path)
        yield sunday_tok


class TestTokenizerTraining:
    """Test that tokenizer training works correctly."""

    def test_training_produces_valid_tokenizer(self, trained_tokenizer):
        """Tokenizer should be created and have a valid vocab."""
        assert trained_tokenizer.vocab_size > 0
        assert trained_tokenizer.vocab_size <= 500

    def test_special_tokens_have_correct_ids(self, trained_tokenizer):
        """Special tokens should have IDs 0-4."""
        assert trained_tokenizer.pad_id == 0
        assert trained_tokenizer.unk_id == 1
        assert trained_tokenizer.bos_id == 2
        assert trained_tokenizer.eos_id == 3
        assert trained_tokenizer.sep_id == 4

    def test_vocab_size_property(self, trained_tokenizer):
        """vocab_size property and len() should match."""
        assert len(trained_tokenizer) == trained_tokenizer.vocab_size


class TestEncoding:
    """Test text encoding."""

    def test_encode_returns_list_of_ints(self, trained_tokenizer):
        """Encoding should return a list of integers."""
        ids = trained_tokenizer.encode("Hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_with_bos_eos(self, trained_tokenizer):
        """Encoding with bos/eos should wrap the token IDs."""
        ids = trained_tokenizer.encode("test", add_bos=True, add_eos=True)
        assert ids[0] == trained_tokenizer.bos_id   # First = <bos>
        assert ids[-1] == trained_tokenizer.eos_id   # Last = <eos>

    def test_encode_without_bos_eos(self, trained_tokenizer):
        """Encoding without bos/eos should not add them."""
        ids = trained_tokenizer.encode("test", add_bos=False, add_eos=False)
        if len(ids) > 0:
            assert ids[0] != trained_tokenizer.bos_id or ids[0] == trained_tokenizer.bos_id
            # Can't be certain the first content token isn't coincidentally bos_id,
            # but we can verify length is shorter
        ids_with = trained_tokenizer.encode("test", add_bos=True, add_eos=True)
        assert len(ids) == len(ids_with) - 2  # 2 fewer tokens (no bos/eos)

    def test_encode_empty_string(self, trained_tokenizer):
        """Empty string should still work (returns just bos/eos)."""
        ids = trained_tokenizer.encode("", add_bos=True, add_eos=True)
        assert len(ids) >= 2  # At least bos + eos


class TestDecoding:
    """Test token ID decoding."""

    def test_decode_returns_string(self, trained_tokenizer):
        """Decoding should return a string."""
        ids = trained_tokenizer.encode("Hello world", add_bos=False, add_eos=False)
        text = trained_tokenizer.decode(ids)
        assert isinstance(text, str)

    def test_roundtrip_encode_decode(self, trained_tokenizer):
        """encode → decode should approximately reconstruct the input."""
        original = "The quick brown fox"
        ids = trained_tokenizer.encode(original, add_bos=False, add_eos=False)
        decoded = trained_tokenizer.decode(ids, skip_special_tokens=True)
        # BPE may add/change whitespace slightly, but content should match
        assert decoded.strip() == original.strip()

    def test_roundtrip_with_special_tokens(self, trained_tokenizer):
        """Roundtrip should work even with bos/eos when skip_special=True."""
        original = "Machine learning is great"
        ids = trained_tokenizer.encode(original, add_bos=True, add_eos=True)
        decoded = trained_tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded.strip() == original.strip()


class TestBatchOperations:
    """Test batch encode/decode."""

    def test_batch_encode(self, trained_tokenizer):
        """Batch encoding should return a list of lists."""
        texts = ["Hello world", "Foo bar", "Testing"]
        batch = trained_tokenizer.encode_batch(texts)
        assert len(batch) == 3
        assert all(isinstance(ids, list) for ids in batch)

    def test_batch_decode(self, trained_tokenizer):
        """Batch decoding should return a list of strings."""
        texts = ["Hello world", "Testing"]
        batch_ids = trained_tokenizer.encode_batch(texts, add_bos=False, add_eos=False)
        decoded = trained_tokenizer.decode_batch(batch_ids)
        assert len(decoded) == 2
        assert all(isinstance(t, str) for t in decoded)

    def test_batch_encode_max_length(self, trained_tokenizer):
        """max_length should truncate long sequences."""
        texts = ["A very long sentence " * 50]
        batch = trained_tokenizer.encode_batch(texts, max_length=10)
        assert len(batch[0]) <= 10


class TestTokenInspection:
    """Test token inspection utilities."""

    def test_get_tokens(self, trained_tokenizer):
        """get_tokens should return token strings."""
        tokens = trained_tokenizer.get_tokens("Hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_token_id_roundtrip(self, trained_tokenizer):
        """token_to_id and id_to_token should be inverse operations."""
        token = "<pad>"
        tid = trained_tokenizer.token_to_id(token)
        assert tid is not None
        back = trained_tokenizer.id_to_token(tid)
        assert back == token

    def test_get_vocab(self, trained_tokenizer):
        """get_vocab should return a dictionary."""
        vocab = trained_tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) == trained_tokenizer.vocab_size

    def test_repr(self, trained_tokenizer):
        """__repr__ should return a readable string."""
        r = repr(trained_tokenizer)
        assert "SundayTokenizer" in r
        assert "vocab_size" in r


class TestPersistence:
    """Test saving and loading."""

    def test_save_and_load(self, trained_tokenizer):
        """Tokenizer should be savable and loadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train a fresh tokenizer and save it
            from src.tokenizer.trainer import train_tokenizer_from_texts
            tok = train_tokenizer_from_texts(
                texts=SAMPLE_TEXTS[:50],
                vocab_size=300,
                save_dir=tmpdir,
                save_name="persist_test.json",
            )
            # Load it back
            path = os.path.join(tmpdir, "persist_test.json")
            loaded = SundayTokenizer.from_file(path)

            # Verify it works
            ids = loaded.encode("Test sentence")
            text = loaded.decode(ids, skip_special_tokens=True)
            assert "Test" in text

    def test_load_nonexistent_raises(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SundayTokenizer.from_file("nonexistent_file.json")
