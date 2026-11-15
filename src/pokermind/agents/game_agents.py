"""Agents for the generic game interface (M3): uniform random and tabular strategy.

They receive only an ``Observation``, which never contains the opponent's
private cards, and sample with the shared one-draw CDF sampler.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokermind.agents.cfr_agent import sample_action
from pokermind.games.base import Game, Observation
from pokermind.strategy import GameStrategy, load_game_strategy


class UniformGameAgent:
    name = "uniform"

    def act(
        self, observation: Observation, legal_actions: tuple[int, ...], rng: np.random.Generator
    ) -> int:
        n = len(legal_actions)
        return sample_action(legal_actions, [1.0 / n] * n, rng, observation.key)


class GameStrategyAgent:
    def __init__(self, strategy: GameStrategy, name: str = "cfr") -> None:
        self._strategy = {k: dict(v) for k, v in strategy.items()}
        self._name = name

    @classmethod
    def from_file(cls, path: Path, game: Game) -> GameStrategyAgent:
        return cls(load_game_strategy(path, game.name, game.action_codes), f"cfr({path.name})")

    @property
    def name(self) -> str:
        return self._name

    def act(
        self, observation: Observation, legal_actions: tuple[int, ...], rng: np.random.Generator
    ) -> int:
        key = observation.key
        dist = self._strategy.get(key)
        if dist is None:
            raise KeyError(f"strategy has no entry for information set {key!r}")
        if set(dist) != set(legal_actions):
            raise ValueError(f"strategy actions at {key!r} do not match legal {legal_actions}")
        return sample_action(legal_actions, [dist[a] for a in legal_actions], rng, key)
