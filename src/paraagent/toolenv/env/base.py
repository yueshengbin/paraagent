from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class Observation:
    """Full next-round model input; set exactly one of text, messages, or token_ids."""

    text: str | None = None
    """Full prompt as a raw text string."""

    messages: list[dict] | None = None
    """Full chat messages list (OpenAI format)."""

    token_ids: list[int] | None = None
    """Fully tokenised prompt ids."""

@dataclass
class Action:
    """Action taken by the LLM."""

    text: str | None = None
    """Decoded LLM response text."""

    token_ids: list[int] | None = None
    """Raw LLM response token ids."""

class AgentEnv(ABC):
    """Stateful reset/step interface; register implementations with @AgentEnv.register(name)."""

    _registry: dict[str, type["AgentEnv"]] = {}

    use_trajectory_reward: bool = False
    """If True, the agent flow should defer reward computation to a single
    trajectory-level call after all steps complete, rather than scoring
    each step independently."""

    @classmethod
    def register(cls, name: str):
        """Decorator to register an AgentEnv subclass under *name*."""

        def decorator(subclass: type["AgentEnv"]) -> type["AgentEnv"]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_config(cls, env_type: str, **kwargs) -> "AgentEnv":
        """Instantiate a registered env by *env_type* with remaining kwargs.

        Raises:
            ValueError: If *env_type* is not registered.
        """
        if env_type not in cls._registry:
            raise ValueError(f"Unknown env type: {env_type!r}. Available: {list(cls._registry.keys())}")
        return cls._registry[env_type](**kwargs)

    def build_trajectory_solution_str(self, steps: list) -> str:
        """Join step response texts for reward computation; subclasses may override the format."""
        blocks = []
        for step in steps:
            response_text = step.extra_fields.get("response_text", "")
            blocks.append(response_text)
        return "\n".join(blocks)

    @abstractmethod
    async def reset(self, **kwargs) -> Observation:
        """Reset state and return the initial observation."""
        raise NotImplementedError

    @abstractmethod
    async def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute a model action and return (observation, reward, done, info)."""
        raise NotImplementedError

    def cleanup(self) -> None:  
        """Release per-trajectory resources after completion or failure.
        Stateless environments need no cleanup.
        """
