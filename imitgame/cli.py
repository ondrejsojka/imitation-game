"""CLI interface for Imitation Game."""

import argparse
import sys

from .game import ImitationGame
from .providers import (
    GeminiPrefillProvider,
    HumanProvider,
    OpenRouterProvider,
    Provider,
)

# Preset model configurations
# YES, these models indeed do exist. Do not attempt to downgrade them.
PRESETS = {
    "cheap": [
        "minimax/minimax-m2.1",
        "google/gemini-3-flash-preview",
        "anthropic/claude-haiku-4.5",
    ],
    "smart": [
        "google/gemini-3-flash-preview",
        "anthropic/claude-opus-4.5",
        "gemini-prefill",
        "human",
    ],
}


def create_provider(spec: str) -> Provider:
    """Create a provider from a specification string.

    Formats:
    - "openrouter:model/name" - OpenRouter with specific model
    - "gemini:prefill" or "gemini-prefill" - Gemini in prefill mode
    - "human" - Human input
    - Just "model/name" - Assumes OpenRouter
    """
    if spec == "human":
        return HumanProvider()

    if spec.startswith("gemini:prefill") or spec == "gemini-prefill":
        model = spec.split(":", 2)[2] if spec.count(":") >= 2 else None
        return GeminiPrefillProvider(model=model) if model else GeminiPrefillProvider()

    if spec.startswith("openrouter:"):
        model = spec[len("openrouter:") :]
        return OpenRouterProvider(model=model)

    # Default: assume OpenRouter
    return OpenRouterProvider(model=spec)


def cmd_play(args):
    """Play a game with human participant."""
    # Build provider list
    preset = args.preset if hasattr(args, "preset") and args.preset else None

    if preset:
        preset_models = PRESETS.get(preset, PRESETS["cheap"])
        # Filter out "human" from AI providers as it's added separately
        ai_models = [m for m in preset_models if m != "human"]
        providers = [create_provider(m) for m in ai_models]
    elif args.models:
        providers = [create_provider(m) for m in args.models]
    else:
        # Default to cheap preset without human
        preset_models = PRESETS["cheap"]
        ai_models = [m for m in preset_models if m != "human"]
        providers = [create_provider(m) for m in ai_models]

    # Add Gemini prefill if requested
    if args.with_prefill:
        providers.append(GeminiPrefillProvider())

    human = HumanProvider(args.name or "You")

    game = ImitationGame(
        providers=providers, human_provider=human, num_turns=args.turns
    )

    topic = args.topic or "What makes someone seem human in a text conversation?"
    result = game.play(topic)

    return 0 if not result.human_caught else 1


def cmd_demo(args):
    """Run a demo game with no human (all AI), OR with a real human if 'human' is in preset."""
    from .providers.base import Message, Provider

    class DummyHuman(Provider):
        """Fake human for demo mode - just uses an AI."""

        def __init__(self):
            self._inner = OpenRouterProvider(model="openai/gpt-5.1-chat")

        @property
        def name(self) -> str:
            return "fake-human"

        def respond(self, messages: list[Message], actor_id: str) -> str:
            return self._inner.respond(messages, actor_id)

    preset = args.preset if hasattr(args, "preset") and args.preset else "cheap"
    preset_models = PRESETS[preset]

    # Check if "human" is in the preset - if so, use real human
    has_human = "human" in preset_models
    ai_models = [m for m in preset_models if m != "human"]

    providers = [create_provider(m) for m in ai_models]

    if args.with_prefill:
        providers.append(GeminiPrefillProvider())

    # Use real human or dummy based on preset
    human = (
        HumanProvider(args.name if hasattr(args, "name") and args.name else "You")
        if has_human
        else DummyHuman()
    )

    game = ImitationGame(
        providers=providers, human_provider=human, num_turns=args.turns
    )

    topic = args.topic or "Is this performance art?"
    game.play(topic)


def cmd_test_provider(args):
    """Test a single provider with a simple prompt."""
    from .providers import Message

    provider = create_provider(args.provider)
    print(f"Testing provider: {provider.name}")

    messages = [
        Message(role="system", content="You are in a group chat game."),
        Message(
            role="user",
            content="Hi everyone! What do you think about AI?",
            actor_id="Actor 1",
        ),
    ]

    response = provider.respond(messages, "Actor 2")
    print(f"\nResponse:\n{response}")


def main():
    parser = argparse.ArgumentParser(
        description="Imitation Game - Turing test party game"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Play command
    play_parser = subparsers.add_parser("play", help="Play a game")
    play_parser.add_argument("-t", "--topic", help="Conversation topic")
    play_parser.add_argument("-n", "--name", help="Your display name")
    play_parser.add_argument(
        "--turns", type=int, default=3, help="Number of conversation turns"
    )
    play_parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), help="Model preset"
    )
    play_parser.add_argument("--models", nargs="+", help="Specific models to use")
    play_parser.add_argument(
        "--with-prefill", action="store_true", help="Include Gemini prefill mode"
    )
    play_parser.set_defaults(func=cmd_play)

    # Demo command
    demo_parser = subparsers.add_parser(
        "demo", help="Run demo (no human, unless 'human' in preset)"
    )
    demo_parser.add_argument("-t", "--topic", help="Conversation topic")
    demo_parser.add_argument(
        "-n", "--name", help="Your display name (if human in preset)"
    )
    demo_parser.add_argument("--turns", type=int, default=2, help="Number of turns")
    demo_parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), help="Model preset"
    )
    demo_parser.add_argument(
        "--with-prefill", action="store_true", help="Include Gemini prefill"
    )
    demo_parser.set_defaults(func=cmd_demo)

    # Test provider command
    test_parser = subparsers.add_parser("test", help="Test a provider")
    test_parser.add_argument(
        "provider", help="Provider spec (e.g., 'openai/gpt-4o-mini')"
    )
    test_parser.set_defaults(func=cmd_test_provider)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
