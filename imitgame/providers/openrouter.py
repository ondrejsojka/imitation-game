"""OpenRouter provider using OpenAI-compatible API."""

from openai import OpenAI

from .base import Message, Provider


class OpenRouterProvider(Provider):
    """Provider using OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or self._get_api_key())

    def _get_api_key(self) -> str:
        import os

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        return key

    @property
    def name(self) -> str:
        # Extract short name from model string like "openai/gpt-4o" -> "gpt-4o"
        return self.model.split("/")[-1] if "/" in self.model else self.model

    def respond(self, messages: list[Message], actor_id: str) -> str:
        # Convert to OpenAI format
        # Include actor_id in content for multi-party chat simulation
        # (otherwise models see assistant messages and try to continue them)
        openai_messages = []

        for msg in messages:
            if msg.role == "system":
                openai_messages.append({"role": "system", "content": msg.content})
            elif msg.actor_id:
                # Multi-party: prefix with actor name, use user role
                # This prevents models from "continuing" other actors' messages
                openai_messages.append({
                    "role": "user",
                    "content": f"{msg.actor_id}: {msg.content}"
                })
            else:
                openai_messages.append({"role": msg.role, "content": msg.content})

        response = self.client.chat.completions.create(
            model=self.model, messages=openai_messages, max_tokens=512
        )

        if response.choices is None:
            raise RuntimeError(f"No response from {self.model}")

        return response.choices[0].message.content or ""
