"""Kuhn Poker behind the generic game interface (an adapter over M0's engine).

The M0 engine is not changed. This adapter adds:

* an explicit root chance node whose outcomes are the 6 ordered deals, in
  ``all_deals()`` order, each with probability ``1.0 / 6``. That is exactly the
  weight M1's CFR loop used, which keeps CFR results bit-identical;
* integer actions: the index in M0's ``Action`` enum (CHECK 0, BET 1, CALL 2,
  FOLD 3), listed in the engine's legal order (passive first);
* information-set keys and observations taken straight from ``InfoState``
  (``"K|xb"``), so M1 strategy files and digests stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from pokermind.games.base import CHANCE, TERMINAL
from pokermind.games.kuhn import Action, InfoState, KuhnConfig, KuhnState, all_deals, initial_state

KUHN_ACTIONS: Final[tuple[Action, ...]] = tuple(Action)
_ACTION_ID: Final = {a: i for i, a in enumerate(KUHN_ACTIONS)}
_DEALS: Final = all_deals()


@dataclass(frozen=True, slots=True)
class KuhnGameState:
    """Either the root chance node (``inner is None``) or an M0 ``KuhnState``."""

    config: KuhnConfig
    inner: KuhnState | None = None

    @property
    def is_chance_node(self) -> bool:
        return self.inner is None

    @property
    def is_terminal(self) -> bool:
        return self.inner is not None and self.inner.is_terminal

    @property
    def current_player(self) -> int:
        if self.inner is None:
            return CHANCE
        if self.inner.is_terminal:
            return TERMINAL
        return self.inner.current_player

    def chance_outcomes(self) -> tuple[tuple[int, float], ...]:
        if self.inner is not None:
            raise ValueError("not a chance node")
        p = 1.0 / len(_DEALS)
        return tuple((i, p) for i in range(len(_DEALS)))

    def legal_actions(self) -> tuple[int, ...]:
        if self.inner is None or self.inner.is_terminal:
            return ()
        return tuple(_ACTION_ID[a] for a in self.inner.legal_actions())

    def apply(self, action: int) -> KuhnGameState:
        if self.inner is None:
            if not 0 <= action < len(_DEALS):
                raise ValueError(f"invalid deal outcome {action!r}")
            return KuhnGameState(self.config, initial_state(_DEALS[action], self.config))
        return KuhnGameState(self.config, self.inner.apply(KUHN_ACTIONS[action]))

    def returns(self) -> tuple[float, float]:
        if self.inner is None or not self.inner.is_terminal:
            raise ValueError("returns are only defined at terminal states")
        p0, p1 = self.inner.payoffs()
        return float(p0), float(p1)

    def observation(self, player: int) -> InfoState:
        if self.inner is None:
            raise ValueError("no observation before the deal")
        return self.inner.information_state(player)

    def information_state_key(self, player: int) -> str:
        return self.observation(player).key


@dataclass(frozen=True, slots=True)
class KuhnGame:
    config: KuhnConfig = field(default_factory=KuhnConfig)

    @property
    def name(self) -> str:
        return "kuhn"

    @property
    def action_codes(self) -> tuple[str, ...]:
        return tuple(a.code for a in KUHN_ACTIONS)

    def initial_state(self) -> KuhnGameState:
        return KuhnGameState(self.config)


def action_id(action: Action) -> int:
    return _ACTION_ID[action]


def to_action_strategy(strategy: dict[str, dict[int, float]]) -> dict[str, dict[Action, float]]:
    """Convert an integer-action Kuhn strategy to M0/M1's ``Action``-keyed form."""
    return {k: {KUHN_ACTIONS[a]: p for a, p in d.items()} for k, d in strategy.items()}


def from_action_strategy(
    strategy: dict[str, dict[Action, float]],
) -> dict[str, dict[int, float]]:
    return {k: {_ACTION_ID[a]: p for a, p in d.items()} for k, d in strategy.items()}
