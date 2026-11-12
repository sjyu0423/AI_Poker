"""Agent interface.

Agents receive only an :class:`~pokermind.games.kuhn.InfoState`, the tuple of
legal actions, and a dedicated random generator. They never receive the full
game state, so they cannot observe the opponent's private card.

Agents should draw all randomness from the provided ``rng`` so that seeded
simulations are reproducible.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from pokermind.games.kuhn import Action, InfoState


@runtime_checkable
class Agent(Protocol):
    """A policy that maps an information state to an action."""

    @property
    def name(self) -> str: ...

    def act(
        self,
        info_state: InfoState,
        legal_actions: tuple[Action, ...],
        rng: np.random.Generator,
    ) -> Action: ...
