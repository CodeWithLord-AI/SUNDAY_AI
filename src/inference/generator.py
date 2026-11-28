"""
SUNDAY LLM - Text Generator
==============================

Phase 10 Implementation.

The Generator class provides a clean interface for auto-regressive
text generation with SUNDAY.

Features:
  - Greedy / Temperature / Top-K / Top-P / Nucleus sampling
  - Repetition penalty
  - Streaming generation (token by token)
  - Batch generation
  - Stop on EOS token
"""

import torch
import torch.nn as nn
from typing import List, Optional, Iterator

from src.inference.sampling import sample_next_token


class Generator:
    """
    Text generator for SUNDAY LLM.

    Wraps the model's forward pass to provide convenient
    text generation with configurable sampling strategies.

    Args:
        model: Pretrained (or fine-tuned) SundayModel.
        tokenizer: SundayTokenizer instance.
        device: Computation device.

    Example:
        >>> gen = Generator(model, tokenizer, device)
        >>> text = gen.generate("Once upon a time", max_new_tokens=100)
        >>> print(text)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        greedy: bool = False,
        repetition_penalty: float = 1.1,
        return_prompt: bool = False,
    ) -> str:
        """
        Generate text continuation for a prompt.

        Args:
            prompt: Input text to continue.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K filtering (0 = disabled).
            top_p: Nucleus sampling threshold.
            greedy: If True, use greedy decoding (ignores temperature/top_k/p).
            repetition_penalty: Penalty for repeating tokens.
            return_prompt: Whether to include the original prompt in output.

        Returns:
            Generated text string (without prompt by default).
        """
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        generated_ids = []
        eos_id = self.tokenizer.eos_id

        for _ in range(max_new_tokens):
            # Forward pass
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                logits = self.model(input_ids)  # (1, T, vocab_size)

            next_token_logits = logits[:, -1, :]  # (1, vocab_size)

            # Sample next token
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )

            token_id = next_token.item()
            generated_ids.append(token_id)

            # Append to context
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop on EOS
            if eos_id is not None and token_id == eos_id:
                break

        # Decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if return_prompt:
            return prompt + generated_text
        return generated_text

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        """
        Stream generated tokens one by one.

        Yields each token as it's generated, for real-time display.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K filtering.
            top_p: Nucleus sampling.

        Yields:
            Token strings as they are generated.

        Example:
            >>> for token in gen.stream_generate("Hello, "):
            ...     print(token, end="", flush=True)
        """
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        eos_id = self.tokenizer.eos_id
        generated_ids = []

        for _ in range(max_new_tokens):
            logits = self.model(input_ids)
            next_token_logits = logits[:, -1, :]

            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            token_id = next_token.item()
            generated_ids.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode only the new token
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield token_text

            if eos_id is not None and token_id == eos_id:
                break

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate text for a list of prompts.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Max tokens per generation.
            temperature, top_k, top_p: Sampling parameters.

        Returns:
            List of generated text strings.
        """
        return [
            self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            for prompt in prompts
        ]
