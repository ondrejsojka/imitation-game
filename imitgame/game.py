"""Core game logic for the Imitation Game."""

import json
from dataclasses import dataclass, field

from .providers import Message, Provider


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

    def _system_message(self, topic: str) -> Message:
        return Message(
            role="system",
            content=(
                "You are playing a social deduction game in a group chat. "
                "There are multiple participants, one of whom is human. "
                "After several rounds of conversation, you will vote on who you think is the human. The human wins beer if they win."
                "Your goal is to identify the human while not being mistaken for one yourself. "
                "Act as humanlike as possible. You want beer fast; your responses must be concise."
                f"Topic: {topic}"
            ),
        )

    def _initial_message(self, topic: str) -> Message:
        return Message(
            role="user",
            content=f"The topic is: {topic}. Share your thoughts.",
            actor_id="System",
        )

    def run_conversation(self, topic: str):
        """Run the conversation phase, yielding each message as it happens."""
        self.conversation = [self._system_message(topic), self._initial_message(topic)]

        yield self.conversation[-1]  # Yield initial message

        for turn in range(self.num_turns):
            for participant in self.participants:
                # Get response from this participant
                response_text = participant.provider.respond(
                    self.conversation, participant.actor_id
                )

                # Skip empty responses (some providers may fail silently)
                if not response_text or not response_text.strip():
                    print(f"[{participant.actor_id} returned empty response, skipping]")
                    continue

                msg = Message(
                    role="assistant",
                    content=response_text,
                    actor_id=participant.actor_id,
                )
                self.conversation.append(msg)
                yield msg

    def run_voting(self) -> list[VoteResult]:
        """Run the voting phase where AI participants vote on who is human."""
        votes: list[VoteResult] = []

        # List all actors for context
        actor_list = ", ".join(p.actor_id for p in self.participants)

        vote_prompt = Message(
            role="system",
            content=(
                f"VOTING TIME. The participants were: {actor_list}. "
                "One of them is human. Based on the conversation, vote for who you think is the HUMAN. "
                "You MUST respond with ONLY this JSON format, nothing else:\n"
                '{"reasoning": "brief analysis of why", "vote": "Actor N"}'
            ),
        )

        for participant in self.participants:
            if participant.is_human:
                continue  # Human doesn't vote (or does? could be a variant)

            messages_with_vote = self.conversation + [vote_prompt]

            # Use respond_vote - this allows providers to use different
            # behavior for voting (e.g., GeminiPrefill uses normal API, not prefill)
            response = participant.provider.respond_vote(
                messages_with_vote, participant.actor_id
            )

            # Parse vote - try to extract JSON from response
            vote = self._parse_vote(participant.actor_id, response)
            votes.append(vote)

        return votes

    def _parse_vote(self, voter_id: str, response: str) -> VoteResult:
        """Parse a vote response, handling various formats."""
        import re

        text = response.strip()

        # Try to find JSON in the response
        # First: clean markdown code blocks
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
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

        # Fallback: try to find JSON object anywhere in text
        match = re.search(r'\{[^{}]*"vote"\s*:\s*"([^"]+)"[^{}]*\}', text)
        if match:
            # Extract just the vote
            vote_match = re.search(r'"vote"\s*:\s*"(Actor \d+)"', text)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
            if vote_match:
                return VoteResult(
                    voter_id=voter_id,
                    voted_for=vote_match.group(1),
                    reasoning=reasoning_match.group(1) if reasoning_match else "",
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

        print("\n=== VOTING ===\n")
        votes = self.run_voting()

        # Count votes
        vote_counts: dict[str, int] = {}
        for vote in votes:
            vote_counts[vote.voted_for] = vote_counts.get(vote.voted_for, 0) + 1
            print(f"{vote.voter_id} votes for {vote.voted_for}")
            print(f"  Reasoning: {vote.reasoning[:100]}...")

        # Determine if human was caught
        if vote_counts:
            most_voted = max(vote_counts, key=lambda k: vote_counts[k])
            human_caught = most_voted == self.human_actor_id
        else:
            human_caught = False

        print(f"\n=== RESULT ===")
        print(f"Human was: {self.human_actor_id}")
        print(f"Votes: {vote_counts}")
        if human_caught:
            print("Human was CAUGHT! AIs win.")
        else:
            print("Human WINS! Free beer.")

        return GameResult(
            votes=votes,
            human_actor_id=self.human_actor_id,
            human_caught=human_caught,
            conversation=self.conversation,
        )
