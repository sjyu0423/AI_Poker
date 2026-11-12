"""Uniform random baseline."""

from __future__ import annotations

import numpy as np

from pokermind.games.kuhn import Action, InfoState


class RandomAgent:
    """Chooses uniformly among the legal actions."""

    name = "random"

    def act(
        self,
        info_state: InfoState,
        legal_actions: tuple[Action, ...],
        rng: np.random.Generator,
    ) -> Action:
        if not legal_actions:
            raise ValueError("no legal actions available")
        return legal_actions[int(rng.integers(len(legal_actions)))]
