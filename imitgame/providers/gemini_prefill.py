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

    # Ondřej from Brno - gives the prefill model a concrete persona to embody
    PERSONA = (
        "Ondřej z Brna, studuje chemii a filozofii na MUNI. "
        "Rád pije Braník, má skeptický pohled na věci, občas trochu cynický. "
        "Mluví přirozeně, používá čeglish (mix češtiny a angličtiny). "
        "Má rád debaty a občas zbytečně overanalyzuje."
    )

    def __init__(
        self, model: str = "gemini-3-flash-preview", api_key: str | None = None
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

        # Build conversation as a text transcript - this goes ENTIRELY in the model block
        # The trick: model "continues" the transcript as if predicting text, not responding
        transcript_lines = []

        # Add persona header so the model knows WHO this actor is
        transcript_lines.append(f"[{actor_id} is {self.PERSONA}]")

        for msg in messages:
            if msg.role == "system":
                continue  # Skip system messages in transcript
            prefix = msg.actor_id or ("User" if msg.role == "user" else "Assistant")
            transcript_lines.append(f"{prefix}: {msg.content}")

        # The ENTIRE transcript + start of this actor's turn goes in model prefill
        transcript = "\n\n".join(transcript_lines)
        prefill = f"{transcript}\n\n{actor_id}:"

        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": "You are in transcript simulation mode. "
                        "You output continuations of group chat transcripts. "
                        "Output ONLY one participant's response, then stop."
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "cat transcript.txt"}],
                },
                # The entire conversation lives HERE - model "continues" from this
                {"role": "model", "parts": [{"text": prefill}]},
            ],
            "generationConfig": {
                "maxOutputTokens": 256,
                "stopSequences": ["\n\nActor", "\nActor "],  # Stop before next actor
            },
        }

        response = requests.post(url, json=payload)
        data = response.json()

        if "candidates" not in data:
            raise RuntimeError(f"Gemini error: {data}")

        return data["candidates"][0]["content"]["parts"][0]["text"]

    def respond_vote(self, messages: list[Message], actor_id: str) -> str:
        """For voting, use standard Gemini API with proper reasoning - no prefill tricks."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        # Build conversation context for the model
        context_lines = []
        for msg in messages:
            if msg.role == "system":
                # Include system message content as context
                context_lines.append(f"[System]: {msg.content}")
            else:
                prefix = msg.actor_id or ("User" if msg.role == "user" else "Assistant")
                context_lines.append(f"{prefix}: {msg.content}")

        context = "\n\n".join(context_lines)

        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are a judge in a social deduction game. "
                            "You need to identify which participant is the HUMAN among AI participants. "
                            "Respond with ONLY valid JSON in this exact format:\n"
                            '{"reasoning": "your analysis here", "vote": "Actor N"}'
                        )
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"Here is the conversation. Who is the human?\n\n{context}"
                        }
                    ],
                }
            ],
            "generationConfig": {"maxOutputTokens": 512},
        }

        response = requests.post(url, json=payload)
        data = response.json()

        if "candidates" not in data:
            raise RuntimeError(f"Gemini error: {data}")

        return data["candidates"][0]["content"]["parts"][0]["text"]
