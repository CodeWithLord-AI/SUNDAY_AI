"""
SUNDAY LLM - Chat Interface
==============================

Phase 10 Implementation.

Manages a conversation with the SUNDAY model.

For instruction-tuned models, formats each turn using the Alpaca template:
  ### Instruction: {user_message}
  ### Response: {model_response}

For base (non-instruction-tuned) models, just concatenates turns.

Chat History:
  Maintains a list of (role, content) tuples.
  Truncates old history when it exceeds max_context_len.
"""

from typing import List, Tuple, Optional
import torch

from src.inference.generator import Generator
from src.finetuning.data_formatter import format_inference_prompt


class ChatSession:
    """
    Manages a multi-turn conversation with SUNDAY LLM.

    Maintains conversation history and formats prompts appropriately
    for instruction-tuned or base models.

    Args:
        generator: A Generator instance wrapping the model.
        instruction_tuned: Whether the model has been fine-tuned on instructions.
        system_prompt: Optional system context (prepended to all conversations).
        max_history_turns: Maximum number of previous turns to include.

    Example:
        >>> session = ChatSession(generator, instruction_tuned=True)
        >>> response = session.chat("What is machine learning?")
        >>> print(response)
        Machine learning is a branch of AI...
    """

    def __init__(
        self,
        generator: Generator,
        instruction_tuned: bool = True,
        system_prompt: Optional[str] = None,
        max_history_turns: int = 5,
    ):
        self.generator = generator
        self.instruction_tuned = instruction_tuned
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns

        # History: list of (role, content) where role is 'user' or 'assistant'
        self.history: List[Tuple[str, str]] = []

    def chat(
        self,
        user_message: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """
        Send a message and get a response.

        Args:
            user_message: The user's input message.
            max_new_tokens: Max tokens for the response.
            temperature: Sampling temperature.
            top_k: Top-K filtering.
            top_p: Nucleus sampling.

        Returns:
            The model's response as a string.
        """
        # Build the prompt from history + current message
        prompt = self._build_prompt(user_message)

        # Generate response
        response = self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Clean up the response
        response = self._clean_response(response)

        # Add to history
        self.history.append(("user", user_message))
        self.history.append(("assistant", response))

        # Truncate old history if needed
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-self.max_history_turns * 2:]

        return response

    def stream_chat(
        self,
        user_message: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        """
        Stream a response token by token.

        Yields tokens as they are generated. Useful for real-time display
        in the Streamlit UI.

        Args:
            user_message: The user's input.

        Yields:
            Token strings as they are generated.
        """
        prompt = self._build_prompt(user_message)

        full_response = ""
        for token in self.generator.stream_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ):
            full_response += token
            yield token

        # Clean and store in history
        clean_response = self._clean_response(full_response)
        self.history.append(("user", user_message))
        self.history.append(("assistant", clean_response))

    def _build_prompt(self, user_message: str) -> str:
        """
        Build the full prompt including history and current message.

        For instruction-tuned models: uses Alpaca template.
        For base models: simple concatenation.
        """
        if self.instruction_tuned:
            # For instruction-tuned models, format as Alpaca prompt
            # Include recent history for context
            prompt_parts = []

            if self.system_prompt:
                prompt_parts.append(f"### System:\n{self.system_prompt}\n")

            # Add recent history
            for role, content in self.history[-self.max_history_turns * 2:]:
                if role == "user":
                    prompt_parts.append(f"### Instruction:\n{content}\n\n### Response:\n")
                else:
                    prompt_parts.append(f"{content}\n\n")

            # Add current message
            prompt_parts.append(format_inference_prompt(user_message))

            return "".join(prompt_parts)
        else:
            # Base model: simple concatenation
            parts = []
            if self.system_prompt:
                parts.append(self.system_prompt + "\n\n")

            for role, content in self.history[-self.max_history_turns * 2:]:
                prefix = "Human: " if role == "user" else "Assistant: "
                parts.append(f"{prefix}{content}\n")

            parts.append(f"Human: {user_message}\nAssistant: ")

            return "".join(parts)

    def _clean_response(self, response: str) -> str:
        """Remove artifacts and trim the response."""
        # Stop at next instruction marker (for instruction-tuned models)
        for stop_token in ["### Instruction:", "### Human:", "Human:", "\n\n\n"]:
            if stop_token in response:
                response = response[:response.index(stop_token)]

        return response.strip()

    def reset(self) -> None:
        """Clear the conversation history."""
        self.history = []
        print("[Chat] Conversation reset.")

    def print_history(self) -> None:
        """Print the full conversation history."""
        print("\n[Chat] --- Conversation History ---")
        for role, content in self.history:
            prefix = "You: " if role == "user" else "SUNDAY: "
            print(f"{prefix}{content}")
        print("[Chat] --- End History ---\n")
