"""Gemini provider in prefill/continuation mode.

Instead of responding as an "assistant", this mode makes the model
predict how the conversation would continue - closer to a base model's
behavior. This gives a fundamentally different "epistemology of humanness".
"""

import os

import requests

from .base import Message, Provider


class GeminiPrefillProvider(Provider):
    """Provider using Gemini's prefill mode for continuation-style responses.

    The model receives the conversation as if it's predicting text continuation,
    not responding as a helpful assistant. This removes the "instruct tuning"
    persona and tests a different kind of humanness.
    """

    def __init__(
        self, model: str = "gemini-2.5-flash-preview-05-20", api_key: str | None = None
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

    @property
    def name(self) -> str:
        return f"{self.model}:prefill"

    def respond(self, messages: list[Message], actor_id: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        # Build conversation as a text transcript for prefill
        transcript_lines = []
        for msg in messages:
            if msg.role == "system":
                continue  # Skip system messages in transcript
            prefix = msg.actor_id or ("User" if msg.role == "user" else "Assistant")
            transcript_lines.append(f"{prefix}: {msg.content}")

        # Add the start of this actor's response for prefill
        transcript = "\n\n".join(transcript_lines)
        prefill = f"{transcript}\n\n{actor_id}:"

        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": "You are continuing a group conversation transcript. "
                        "Predict what the next participant would naturally say. "
                        "Be concise and match the conversational style."
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Continue this conversation transcript:"}],
                },
                {"role": "model", "parts": [{"text": prefill}]},
            ],
            "generationConfig": {"maxOutputTokens": 512},
        }

        response = requests.post(url, json=payload)
        data = response.json()

        if "candidates" not in data:
            raise RuntimeError(f"Gemini error: {data}")

        return data["candidates"][0]["content"]["parts"][0]["text"]
