"""Baseline agents."""

from __future__ import annotations

import numpy as np
import pytest

from pokermind.agents import (
    AGENT_REGISTRY,
    Agent,
    AlwaysBetAgent,
    AlwaysPassAgent,
    RandomAgent,
    ThresholdAgent,
    make_agent,
)
from pokermind.games.kuhn import Action, Card, InfoState

X, B, C, F = Action.CHECK, Action.BET, Action.CALL, Action.FOLD
OPEN = (X, B)
FACING = (F, C)


def info(card: Card, history: tuple[Action, ...]) -> InfoState:
    return InfoState(player=len(history) % 2, private_card=card, history=history)


@pytest.mark.parametrize("name", sorted(AGENT_REGISTRY))
def test_registry_agents_satisfy_protocol(name: str) -> None:
    agent = make_agent(name)
    assert isinstance(agent, Agent)
    assert agent.name


def test_unknown_agent() -> None:
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("nope")


def test_random_agent_legal_and_seeded() -> None:
    agent = RandomAgent()
    rng = np.random.default_rng(1)
    picks = [agent.act(info(Card.Q, ()), OPEN, rng) for _ in range(2_000)]
    assert set(picks) == set(OPEN)
    assert abs(picks.count(B) / len(picks) - 0.5) < 0.05
    again = RandomAgent()
    rng2 = np.random.default_rng(1)
    assert picks == [again.act(info(Card.Q, ()), OPEN, rng2) for _ in range(2_000)]
    with pytest.raises(ValueError):
        agent.act(info(Card.Q, ()), (), rng)


@pytest.mark.parametrize(
    ("card", "history", "legal", "expected"),
    [
        (Card.K, (), OPEN, B),
        (Card.Q, (), OPEN, X),
        (Card.J, (), OPEN, X),
        (Card.K, (X,), OPEN, B),
        (Card.Q, (X,), OPEN, X),
        (Card.K, (B,), FACING, C),
        (Card.Q, (B,), FACING, C),
        (Card.J, (B,), FACING, F),
        (Card.Q, (X, B), FACING, C),
        (Card.J, (X, B), FACING, F),
    ],
)
def test_threshold_agent_defaults(
    card: Card, history: tuple[Action, ...], legal: tuple[Action, ...], expected: Action
) -> None:
    rng = np.random.default_rng(0)
    assert ThresholdAgent().act(info(card, history), legal, rng) is expected


def test_threshold_agent_custom() -> None:
    agent = ThresholdAgent(bet_threshold=Card.J, call_threshold=Card.K)
    rng = np.random.default_rng(0)
    assert agent.act(info(Card.J, ()), OPEN, rng) is B
    assert agent.act(info(Card.Q, (B,)), FACING, rng) is F


def test_extreme_agents() -> None:
    rng = np.random.default_rng(0)
    for card in Card:
        assert AlwaysPassAgent().act(info(card, ()), OPEN, rng) is X
        assert AlwaysPassAgent().act(info(card, (B,)), FACING, rng) is F
        assert AlwaysBetAgent().act(info(card, ()), OPEN, rng) is B
        assert AlwaysBetAgent().act(info(card, (B,)), FACING, rng) is C
