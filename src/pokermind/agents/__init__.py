"""Agents. Agents depend on the game's public interface only."""

from collections.abc import Callable

from pokermind.agents.base import Agent
from pokermind.agents.cfr_agent import CFRAgent, StrategyAgent
from pokermind.agents.random_agent import RandomAgent
from pokermind.agents.rule_based import AlwaysBetAgent, AlwaysPassAgent, ThresholdAgent

AGENT_REGISTRY: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "rule": ThresholdAgent,
    "always-pass": AlwaysPassAgent,
    "always-bet": AlwaysBetAgent,
}


def make_agent(name: str) -> Agent:
    """Construct a registered agent by name."""
    try:
        return AGENT_REGISTRY[name]()
    except KeyError:
        known = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(f"unknown agent {name!r}; known agents: {known}") from None


__all__ = [
    "AGENT_REGISTRY",
    "Agent",
    "AlwaysBetAgent",
    "AlwaysPassAgent",
    "CFRAgent",
    "RandomAgent",
    "StrategyAgent",
    "ThresholdAgent",
    "make_agent",
]
