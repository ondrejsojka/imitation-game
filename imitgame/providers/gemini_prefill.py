"""Gemini provider in prefill/continuation mode.

Instead of responding as an "assistant", this mode makes the model
predict how the conversation would continue - closer to a base model's
behavior. This gives a fundamentally different "epistemology of humanness".
"""

import json
import os
import re

import requests

from .base import Message, Provider


def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts/ directory."""
    prompts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts"
    )
    path = os.path.join(prompts_dir, f"{name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class GeminiPrefillProvider(Provider):
    """Provider using Gemini's prefill mode for continuation-style responses.

    The model receives the conversation as if it's predicting text continuation,
    not responding as a helpful assistant. This removes the "instruct tuning"
    persona and tests a different kind of humanness.
    """

    PERSONA = _load_prompt("persona_ondrej")

    def __init__(
        self, model: str = "gemini-3-flash-preview", api_key: str | None = None
    ):
        self.model = model
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set"
            )

    @property
    def name(self) -> str:
        return f"{self.model}:prefill"

    def respond(self, messages: list[Message], actor_id: str) -> str:
        # here I hardcode pro on purpose; it's just so much better.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key={self.api_key}"

        # Build conversation as a text transcript - this goes ENTIRELY in the model block
        # The trick: model "continues" the transcript as if predicting text, not responding
        transcript_lines = []

        # Find the last system message that defines the current actor's persona/instructions
        # In our new architecture, it's the first message in the list
        persona_info = self.PERSONA
        for msg in messages:
            if msg.role == "system":
                # Extract identity if provided in system message
                if f"You are {actor_id}" in msg.content:
                    # We keep the custom persona but acknowledge the system instruction
                    pass

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
                "parts": [{"text": _load_prompt("system_transcript_sim")}]
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
                "stopSequences": ["\n\nActor", "\nActor ", "\n\nSystem", "\nSystem:"],
            },
        }

        response = requests.post(url, json=payload)
        data = response.json()

        if "candidates" not in data:
            raise RuntimeError(f"Gemini error: {data}")

        return data["candidates"][0]["content"]["parts"][0]["text"]

    def respond_vote(self, messages: list[Message], actor_id: str) -> str:
        """Use official google-genai SDK for the final judge vote.

        Prefill mode is intentionally not used here.
        """

        # Import lazily so the rest of the provider works even if google-genai
        # is not installed (though it is a project dependency).
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "google-genai is required for judge voting; install project dependencies"
            ) from exc

        # Build conversation context for the model
        context_lines: list[str] = []
        for msg in messages:
            if msg.role == "system":
                continue
            prefix = msg.actor_id or ("User" if msg.role == "user" else "Assistant")
            context_lines.append(f"{prefix}: {msg.content}")

        context = "\n\n".join(context_lines)

        judge_instruction = _load_prompt("judge_vote")

        prompt = (
            "Here is the conversation transcript. Analyze it carefully and identify the human.\n\n"
            f"{context}"
        )

        def extract_json_object(text: str) -> str:
            candidate = text.strip()
            if "```" in candidate:
                match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL
                )
                if match:
                    candidate = match.group(1).strip()

            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate

            match = re.search(r"(\{.*\})", candidate, re.DOTALL)
            return match.group(1).strip() if match else candidate

        client = genai.Client(api_key=self.api_key)

        last_text = ""
        for attempt in range(2):
            attempt_prompt = prompt
            extra_system = None
            if attempt == 1 and last_text:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "Your previous response was NOT valid JSON or was missing required keys. "
                    "Return ONLY valid JSON with keys reasoning and vote.\n\n"
                    f"Previous output:\n{last_text}"
                )
                extra_system = (
                    "Return a single JSON object and nothing else. "
                    "Do not include markdown formatting."
                )

            system_instruction = (
                judge_instruction
                if not extra_system
                else f"{judge_instruction}\n\n{extra_system}"
            )

            response = client.models.generate_content(
                model=self.model,
                contents=attempt_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    max_output_tokens=4096,
                ),
            )

            text = (response.text or "").strip()
            last_text = text

            try:
                parsed = json.loads(extract_json_object(text))
                if not isinstance(parsed, dict):
                    raise ValueError("Vote response was not a JSON object")
                if "vote" not in parsed:
                    raise ValueError("Vote response missing 'vote'")

                vote = str(parsed.get("vote", "")).strip()
                reasoning = str(parsed.get("reasoning", "")).strip()
                if not vote:
                    raise ValueError("Vote response had empty 'vote'")

                # Re-serialize so callers always get strict JSON.
                return json.dumps(
                    {"reasoning": reasoning, "vote": vote}, ensure_ascii=False
                )
            except Exception:
                continue

        # Give the caller *something* usable; game.py already has fallback parsing.
        return last_text
