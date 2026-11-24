"""
SUNDAY LLM - Main Evaluator
==============================

Phase 8 Implementation.

Comprehensive evaluation pipeline that runs all metrics and
produces a summary report.

Evaluation modes:
  1. Perplexity evaluation: on a held-out val set
  2. Text generation samples: generate and display sample completions
  3. Generation quality metrics: diversity, repetition, length
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
from torch.utils.data import DataLoader

from src.evaluation.perplexity import compute_perplexity
from src.evaluation.generation_quality import compute_generation_metrics


class Evaluator:
    """
    Comprehensive model evaluator for SUNDAY LLM.

    Runs all evaluation metrics and produces a readable summary.

    Args:
        model: The SUNDAY model (or any causal LM).
        tokenizer: The SundayTokenizer instance.
        device: Computation device.
        use_amp: Whether to use automatic mixed precision.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        use_amp: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_amp = use_amp

    def evaluate(
        self,
        val_loader: DataLoader,
        prompts: Optional[List[str]] = None,
        max_batches: Optional[int] = 100,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Dict:
        """
        Run full evaluation and return results dict.

        Args:
            val_loader: Validation DataLoader for perplexity computation.
            prompts: Optional list of text prompts for generation samples.
            max_batches: Max validation batches for perplexity.
            max_new_tokens: Tokens to generate per prompt.
            temperature: Sampling temperature.
            top_k: Top-K sampling.

        Returns:
            Dict with keys: 'perplexity', 'generation_metrics', 'samples'
        """
        results = {}

        print("\n[Evaluator] Running evaluation...")
        print("=" * 55)

        # --- 1. Perplexity ---
        print("[Evaluator] Computing perplexity...")
        ppl = compute_perplexity(
            self.model, val_loader, self.device,
            max_batches=max_batches, use_amp=self.use_amp
        )
        results["perplexity"] = ppl
        print(f"[Evaluator] Perplexity: {ppl:.2f}")

        # --- 2. Text Generation Samples ---
        if prompts:
            print(f"\n[Evaluator] Generating {len(prompts)} samples...")
            generated_texts = self._generate_samples(
                prompts, max_new_tokens, temperature, top_k
            )
            results["samples"] = list(zip(prompts, generated_texts))

            # Generation quality metrics
            gen_metrics = compute_generation_metrics(generated_texts)
            results["generation_metrics"] = gen_metrics

            print(f"[Evaluator] Distinct-1: {gen_metrics.get('distinct_1', 0):.3f}")
            print(f"[Evaluator] Distinct-2: {gen_metrics.get('distinct_2', 0):.3f}")
            print(f"[Evaluator] Avg length: {gen_metrics.get('avg_length', 0):.1f} tokens")

        print("=" * 55)
        return results

    def _generate_samples(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> List[str]:
        """Generate text continuations for a list of prompts."""
        self.model.eval()
        generated = []

        for prompt in prompts:
            # Encode prompt
            ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=self.tokenizer.eos_id,
                )

            # Decode (only the generated part, not the prompt)
            new_ids = output_ids[0, len(ids):].tolist()
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            generated.append(text)

        return generated

    def print_samples(self, prompts: List[str], n: int = 3) -> None:
        """
        Print generated text samples for quick qualitative evaluation.

        Args:
            prompts: List of prompt strings.
            n: Number of samples to print.
        """
        samples = self._generate_samples(
            prompts[:n], max_new_tokens=100, temperature=0.8, top_k=50
        )

        print("\n[Evaluator] --- Generation Samples ---")
        for prompt, continuation in zip(prompts[:n], samples):
            print(f"\n  Prompt: {prompt!r}")
            print(f"  Output: {continuation!r}")
        print("[Evaluator] --- End Samples ---\n")


def evaluate_checkpoint(
    checkpoint_path: str,
    val_loader: DataLoader,
    tokenizer,
    model_size: str = "50m",
    device_str: str = "cpu",
) -> Dict:
    """
    Load a checkpoint and evaluate it.

    Convenience function for evaluating saved checkpoints.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        val_loader: Validation DataLoader.
        tokenizer: SundayTokenizer instance.
        model_size: Model size ('tiny', '50m', '125m').
        device_str: Device to use.

    Returns:
        Evaluation results dict.
    """
    from src.model.sunday_model import SundayModel
    from src.training.checkpoint import load_checkpoint

    device = torch.device(device_str)

    # Create and load model
    model = SundayModel.from_config(model_size)
    load_checkpoint(checkpoint_path, model, device=device_str)
    model.to(device)

    evaluator = Evaluator(model, tokenizer, device)

    sample_prompts = [
        "The transformer model",
        "Once upon a time",
        "Machine learning is",
    ]

    return evaluator.evaluate(val_loader, prompts=sample_prompts)
