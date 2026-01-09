"""Base provider interface and common types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    """A message in the conversation."""

    role: Literal["system", "user", "assistant"]
    content: str
    actor_id: str | None = None  # Which participant sent this


class Provider(ABC):
    """Abstract base class for AI model providers.

    Each provider implements a different way to get responses:
    - OpenRouter: Standard chat completions API
    - GeminiPrefill: Prefill/continuation mode (predicts next text, not "assistant" persona)
    - Human: Input from stdin
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this provider instance."""
        ...

    @abstractmethod
    def respond(self, messages: list[Message], actor_id: str) -> str:
        """Generate a response given conversation history.

        Args:
            messages: Full conversation history
            actor_id: The identifier for this participant (e.g., "Actor 1")

        Returns:
            The response text (without actor prefix - that's added by game logic)
        """
        ...
    
    def respond_vote(self, messages: list[Message], actor_id: str) -> str:
        """Generate a vote response. 
        
        Override this for providers that need different behavior for voting
        (e.g., GeminiPrefill uses normal API for voting, not prefill mode).
        
        Default implementation just calls respond().
        """
        return self.respond(messages, actor_id)


class HumanProvider(Provider):
    """Provider that gets input from a human player."""

    def __init__(self, player_name: str = "Human"):
        self._name = player_name

    @property
    def name(self) -> str:
        return self._name

    def respond(self, messages: list[Message], actor_id: str) -> str:
        # Show recent context
        print(f"\n--- Your turn as {actor_id} ---")
        return input("> ").strip()
