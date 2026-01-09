"""Core game logic for the Imitation Game."""

import json
import os
from dataclasses import dataclass, field

from .providers import Message, Provider


def _load_prompt(name: str, **kwargs) -> str:
    """Load a prompt template from prompts/ directory and format it."""
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", f"{name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content.format(**kwargs) if kwargs else content


@dataclass
class Participant:
    """A participant in the game (human or AI)."""

    provider: Provider
    actor_id: str
    is_human: bool = False


@dataclass
class VoteResult:
    """Result of a vote from one participant."""

    voter_id: str
    voted_for: str
    reasoning: str


@dataclass
class GameResult:
    """Final result of a game round."""

    votes: list[VoteResult]
    human_actor_id: str
    human_caught: bool
    conversation: list[Message]


class ImitationGame:
    """The main game orchestrator.

    Flow:
    1. N participants (mix of AI providers + 1 human)
    2. Topic is announced
    3. Multiple rounds of conversation
    4. Everyone votes on who they think is human
    5. Human wins if not identified by majority
    """

    def __init__(
        self,
        providers: list[Provider],
        human_provider: Provider,
        num_turns: int = 4,
    ):
        self.num_turns = num_turns

        # Create participants with fixed order (from input list)
        all_providers = providers + [human_provider]

        self.participants: list[Participant] = []
        for i, provider in enumerate(all_providers, 1):
            self.participants.append(
                Participant(
                    provider=provider,
                    actor_id=f"Actor {i}",
                    is_human=(provider is human_provider),
                )
            )

        self.human_actor_id = next(p.actor_id for p in self.participants if p.is_human)
        self.conversation: list[Message] = []

    def _system_message(self, topic: str, actor_id: str | None = None) -> Message:
        content = _load_prompt("system_game", topic=topic, actor_id=actor_id)

        return Message(
            role="system",
            content=content,
        )

    def _initial_message(self, topic: str) -> Message:
        return Message(
            role="user",
            content=_load_prompt("initial_topic", topic=topic),
            actor_id="System",
        )

    def run_conversation(self, topic: str):
        """Run the conversation phase, yielding each message as it happens."""
        # Note: We'll send the system message per-participant to include their ID
        self.conversation = [self._initial_message(topic)]

        yield self.conversation[-1]  # Yield initial message

        for turn in range(self.num_turns):
            for participant in self.participants:
                # Add participant-specific system message for the call
                sys_msg = self._system_message(topic, participant.actor_id)
                current_messages = [sys_msg] + self.conversation

                # Get response from this participant
                response_text = participant.provider.respond(
                    current_messages, participant.actor_id
                )

                # Skip empty responses (some providers may fail silently)
                if not response_text or not response_text.strip():
                    print(f"[{participant.actor_id} returned empty response, skipping]")
                    continue

                # Strip actor prefix if model echoed it (common with multi-party format)
                response_text = response_text.strip()
                prefix = f"{participant.actor_id}:"
                if response_text.startswith(prefix):
                    response_text = response_text[len(prefix) :].strip()

                msg = Message(
                    role="assistant",
                    content=response_text,
                    actor_id=participant.actor_id,
                )
                self.conversation.append(msg)
                yield msg

    def run_voting(self) -> list[VoteResult]:
        """Run the voting phase using a separate Judge."""
        # We use Gemini as the external judge (flash model, pro has thinking mode issues)
        from .providers import GeminiPrefillProvider

        judge = GeminiPrefillProvider(model="gemini-3-flash-preview")

        # List all actors for context
        actor_list = ", ".join(p.actor_id for p in self.participants)

        # The judge doesn't need to be a participant, just a static call
        # We use respond_vote logic which we just upgraded for Judge behavior
        response = judge.respond_vote(self.conversation, "Judge")

        vote = self._parse_vote("Judge", response)
        return [vote]

    def _parse_vote(self, voter_id: str, response: str) -> VoteResult:
        """Parse a vote response, handling various formats."""
        import re

        text = response.strip()

        # Try to find JSON in the response
        # First: clean markdown code blocks
        if "```" in text:
            # Match code blocks with optional json label, capture content inside
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                # If no clear JSON block, try to find any block
                match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1)

        # Try direct JSON parse
        try:
            data = json.loads(text)
            return VoteResult(
                voter_id=voter_id,
                voted_for=data.get("vote", "Unknown"),
                reasoning=data.get("reasoning", ""),
            )
        except json.JSONDecodeError:
            pass

        # Fallback: try to find JSON-like object anywhere in text
        # Look for "vote": "Actor N"
        vote_match = re.search(r'"vote"\s*:\s*"(Actor \d+)"', text, re.IGNORECASE)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.IGNORECASE)

        if vote_match:
            return VoteResult(
                voter_id=voter_id,
                voted_for=vote_match.group(1),
                reasoning=reasoning_match.group(1)
                if reasoning_match
                else "(extracted)",
            )

        # Last resort: look for "Actor N" pattern in response
        actor_match = re.search(r"Actor (\d+)", text)
        if actor_match:
            return VoteResult(
                voter_id=voter_id,
                voted_for=f"Actor {actor_match.group(1)}",
                reasoning=f"(extracted from: {text[:100]}...)",
            )

        print(f"Failed to parse vote from {voter_id}: {response[:200]}")
        return VoteResult(
            voter_id=voter_id,
            voted_for="Parse Error",
            reasoning=f"Could not extract vote from response",
        )

        return votes

    def play(self, topic: str) -> GameResult:
        """Play a full game and return the result."""
        # Run conversation (consume the generator)
        for msg in self.run_conversation(topic):
            if msg.actor_id:
                print(f"\033[1m{msg.actor_id}\033[0m: {msg.content}\n")

        print("\n=== JUDGEMENT ===\n")
        votes = self.run_voting()

        # In this mode, we have one authoritative Judge vote
        judge_vote = votes[0]
        print(f"Judge Decision: {judge_vote.voted_for}")
        print(f"Reasoning: {judge_vote.reasoning}\n")

        # Determine if human was caught
        human_caught = judge_vote.voted_for == self.human_actor_id

        print(f"=== RESULT ===")
        print(f"Human was: {self.human_actor_id}")
        if human_caught:
            print("Judge CORRECTLY identified the human! 🍺 You win! Free beer.")
        else:
            print(f"Judge was DECEIVED! They thought {judge_vote.voted_for} was human.")
            print("You lose! AI was more convincing than you. No beer.")

        return GameResult(
            votes=votes,
            human_actor_id=self.human_actor_id,
            human_caught=human_caught,
            conversation=self.conversation,
        )
