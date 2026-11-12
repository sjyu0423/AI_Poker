"""An agent that plays a fixed tabular strategy, such as an exported CFR average.

Like every agent, it only receives an ``InfoState``. It looks up the
information-set key (own card + public history) and samples from the stored
distribution with the generator it is given, so seeded simulations stay
reproducible.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pokermind.games.kuhn import Action, InfoState
from pokermind.strategy import Strategy, load_strategy

_ATOL = 1e-9


class StrategyAgent:
    """Samples actions from a tabular strategy."""

    def __init__(self, strategy: Strategy, name: str = "cfr") -> None:
        self._strategy = {key: dict(dist) for key, dist in strategy.items()}
        self._name = name

    @classmethod
    def from_file(cls, path: Path, name: str | None = None) -> StrategyAgent:
        return cls(load_strategy(path), name or f"cfr({path.name})")

    @property
    def name(self) -> str:
        return self._name

    @property
    def strategy(self) -> dict[str, dict[Action, float]]:
        """A copy of the table this agent plays."""
        return {key: dict(dist) for key, dist in self._strategy.items()}

    def act(
        self,
        info_state: InfoState,
        legal_actions: tuple[Action, ...],
        rng: np.random.Generator,
    ) -> Action:
        key = info_state.key
        dist = self._strategy.get(key)
        if dist is None:
            raise KeyError(f"strategy has no entry for information set {key!r}")
        if set(dist) != set(legal_actions):
            raise ValueError(
                f"strategy actions at {key!r} {sorted(a.code for a in dist)} do not match "
                f"legal actions {sorted(a.code for a in legal_actions)}"
            )
        probs = [dist[a] for a in legal_actions]  # engine order: passive first
        if any(not math.isfinite(p) or p < -_ATOL for p in probs) or abs(sum(probs) - 1) > _ATOL:
            raise ValueError(f"invalid probabilities at {key!r}: {probs}")
        # One uniform draw per decision, inverted through the CDF.
        u = float(rng.random())
        cumulative = 0.0
        for action, p in zip(legal_actions, probs, strict=True):
            cumulative += p
            if u < cumulative:
                return action
        # Rounding can leave the CDF a hair below 1; fall back to the last
        # action that has positive probability.
        return next(
            a for a, p in zip(reversed(legal_actions), reversed(probs), strict=True) if p > 0
        )


CFRAgent = StrategyAgent
