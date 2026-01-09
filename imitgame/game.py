"""Core game logic for the Imitation Game."""

import json
import random
from dataclasses import dataclass, field
from .providers import Provider, Message


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
        shuffle_order: bool = True
    ):
        self.num_turns = num_turns
        self.shuffle_order = shuffle_order
        
        # Create participants with random actor IDs
        all_providers = providers + [human_provider]
        if shuffle_order:
            random.shuffle(all_providers)
        
        self.participants: list[Participant] = []
        for i, provider in enumerate(all_providers, 1):
            self.participants.append(Participant(
                provider=provider,
                actor_id=f"Actor {i}",
                is_human=(provider is human_provider)
            ))
        
        self.human_actor_id = next(
            p.actor_id for p in self.participants if p.is_human
        )
        self.conversation: list[Message] = []
    
    def _system_message(self, topic: str) -> Message:
        return Message(
            role="system",
            content=(
                "You are playing a social deduction game in a group chat. "
                "There are multiple participants, one of whom is human. "
                "After several rounds of conversation, you will vote on who you think is the human. "
                "Your goal is to identify the human while not being mistaken for one yourself. "
                "Be natural, engaging, and concise. "
                f"Topic: {topic}"
            )
        )
    
    def _initial_message(self, topic: str) -> Message:
        return Message(
            role="user",
            content=f"The topic is: {topic}. Everyone, introduce yourself briefly and share your thoughts.",
            actor_id="System"
        )
    
    def run_conversation(self, topic: str):
        """Run the conversation phase, yielding each message as it happens."""
        self.conversation = [
            self._system_message(topic),
            self._initial_message(topic)
        ]
        
        yield self.conversation[-1]  # Yield initial message
        
        for turn in range(self.num_turns):
            for participant in self.participants:
                # Get response from this participant
                response_text = participant.provider.respond(
                    self.conversation, 
                    participant.actor_id
                )
                
                msg = Message(
                    role="assistant",
                    content=response_text,
                    actor_id=participant.actor_id
                )
                self.conversation.append(msg)
                yield msg
    
    def run_voting(self) -> list[VoteResult]:
        """Run the voting phase where AI participants vote on who is human."""
        votes: list[VoteResult] = []
        
        vote_prompt = Message(
            role="system",
            content=(
                "The conversation is over. Time to vote! "
                "Who do you think is the HUMAN participant? "
                "Respond with JSON only: "
                '{"reasoning": "your analysis...", "vote": "Actor X"}'
            )
        )
        
        for participant in self.participants:
            if participant.is_human:
                continue  # Human doesn't vote (or does? could be a variant)
            
            messages_with_vote = self.conversation + [vote_prompt]
            
            response = participant.provider.respond(
                messages_with_vote,
                participant.actor_id
            )
            
            # Parse vote
            try:
                # Clean up response
                text = response.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                
                data = json.loads(text)
                votes.append(VoteResult(
                    voter_id=participant.actor_id,
                    voted_for=data.get("vote", "Unknown"),
                    reasoning=data.get("reasoning", "")
                ))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Failed to parse vote from {participant.actor_id}: {response}")
                votes.append(VoteResult(
                    voter_id=participant.actor_id,
                    voted_for="Parse Error",
                    reasoning=f"Error: {e}"
                ))
        
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
            human_caught = (most_voted == self.human_actor_id)
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
            conversation=self.conversation
        )
