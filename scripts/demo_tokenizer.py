"""
SUNDAY LLM - Tokenizer Demo
===============================

Run this script to train a small BPE tokenizer and see how it works. 

Usage:
  cd e:\\AI&ML\\Project\\SUNDAY_AI
  .venv\\Scripts\\python.exe scripts/demo_tokenizer.py
"""

from src.tokenizer.trainer import train_tokenizer_from_texts
from src.tokenizer.tokenizer import SundayTokenizer
from src.tokenizer.utils import print_tokenization, analyze_tokenization, print_analysis


def main():
    # === STEP 1: Prepare sample training data ===
    print("\n" + "=" * 60)
    print("  SUNDAY LLM — Tokenizer Demo")
    print("=" * 60)

    sample_texts = [
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
        "The embedding layer converts token IDs to dense vectors.",
        "Layer normalization stabilizes training in deep networks.",
        "Residual connections allow gradients to flow through the network.",
        "The feed-forward network processes each token independently.",
        "Cross-entropy loss measures the prediction quality.",
    ] * 50  # Repeat for more data

    # === STEP 2: Train the tokenizer ===
    print("\n📝 Step 1: Training BPE tokenizer on sample data...\n")

    train_tokenizer_from_texts(
        texts=sample_texts,
        vocab_size=1000,  # Small vocab for demo
        min_frequency=2,
        save_dir="tokenizer_files",
        save_name="sunday_tokenizer_demo.json",
    )

    # === STEP 3: Load and use the tokenizer ===
    print("\n📦 Step 2: Loading trained tokenizer...\n")
    tok = SundayTokenizer.from_file("tokenizer_files/sunday_tokenizer_demo.json")
    print(f"  {tok}")

    # === STEP 4: Demonstrate encoding/decoding ===
    print("\n🔤 Step 3: Encoding & Decoding Examples\n")

    test_sentences = [
        "Hello world!",
        "The cat sat on the mat.",
        "SUNDAY is my own language model.",
        "Machine learning is amazing.",
        "Supercalifragilisticexpialidocious",  # Rare word → subwords
    ]

    for sentence in test_sentences:
        print_tokenization(tok, sentence)

    # === STEP 5: Roundtrip test ===
    print("\n🔄 Step 4: Roundtrip Test (encode → decode)\n")
    for sentence in test_sentences[:3]:
        ids = tok.encode(sentence, add_bos=True, add_eos=True)
        decoded = tok.decode(ids, skip_special_tokens=True)
        match = "✅" if decoded.strip() == sentence.strip() else "⚠️"
        print(f"  {match} Original: {sentence!r}")
        print(f"     Decoded:  {decoded!r}")
        print(f"     IDs:      {ids}")
        print()

    # === STEP 6: Analyze tokenization efficiency ===
    print("\n📊 Step 5: Tokenization Analysis\n")
    analysis = analyze_tokenization(tok, sample_texts[:20])
    print_analysis(analysis)

    print("\n✅ Tokenizer demo complete!")
    print(f"   Tokenizer saved to: tokenizer_files/sunday_tokenizer_demo.json")
    print(f"   Vocab size: {tok.vocab_size:,}")
    print(f"\n   Next step: Train with real data in Phase 5 (Colab)")


if __name__ == "__main__":
    main()
