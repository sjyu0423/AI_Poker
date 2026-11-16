"""The Restricted Nash Response game, written out as an explicit ``Game`` (M7A).

The vectorized solver (:mod:`pokermind.robust.solver`) never builds this
game; it reasons about weighted copies instead. This class states the
modified game literally, so its rules and chance probabilities can be tested,
and the generic M3 CFR solver can be run on it to cross-check the
vectorized one.

Players are **roles**: 0 = hero, 1 = opponent. The first chance node draws
``(hero seat, restricted?)``. The four outcomes are encoded as
``2·seat + restricted``, with probabilities ``½(1 − p)`` (free) and ``½p``
(restricted). Then Leduc is played:

* hero acts at its own Leduc information sets, keyed ``"H|<Leduc key>"``. The
  key has neither the coin nor the seat (the seat is implied by the public
  history), so hero can't tell restricted from free;
* a **free** opponent acts at ``"O|<Leduc key>"``;
* a **restricted** opponent's decisions are chance nodes with the model's
  probabilities (zero-probability actions are omitted). It has no
  information sets, since it chooses nothing.

Returns are ``(hero, opponent)``, from the Leduc returns of the matching seats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pokermind.games.base import CHANCE, TERMINAL
from pokermind.games.leduc import LeducGame, LeducObservation, LeducState
from pokermind.strategy import GameStrategy


class RNRState:
    __slots__ = ("game", "inner", "restricted", "seat")

    def __init__(
        self, game: RNRGame, inner: LeducState | None, seat: int, restricted: bool
    ) -> None:
        self.game = game
        self.inner = inner
        self.seat = seat
        self.restricted = restricted

    def _role(self, leduc_seat: int) -> int:
        return 0 if leduc_seat == self.seat else 1

    @property
    def is_terminal(self) -> bool:
        return self.inner is not None and self.inner.is_terminal

    @property
    def is_chance_node(self) -> bool:
        if self.inner is None:
            return True
        if self.inner.is_chance_node:
            return True
        return self.restricted and self._role(self.inner.current_player) == 1

    @property
    def current_player(self) -> int:
        if self.is_terminal:
            return TERMINAL
        if self.is_chance_node:
            return CHANCE
        assert self.inner is not None
        return self._role(self.inner.current_player)

    def chance_outcomes(self) -> tuple[tuple[int, float], ...]:
        p = self.game.p
        if self.inner is None:
            return tuple(
                (2 * seat + r, 0.5 * (p if r else 1.0 - p))
                for seat in (0, 1)
                for r in (0, 1)
                if (p if r else 1.0 - p) > 0
            )
        if self.inner.is_chance_node:
            return self.inner.chance_outcomes()
        key = self.inner.information_state_key(self.inner.current_player)
        dist = self.game.model[key]
        return tuple((a, dist[a]) for a in self.inner.legal_actions() if dist[a] > 0)

    def legal_actions(self) -> tuple[int, ...]:
        if self.is_terminal or self.is_chance_node:
            return ()
        assert self.inner is not None
        return self.inner.legal_actions()

    def apply(self, move: int) -> RNRState:
        if self.inner is None:
            return RNRState(self.game, self.game.base.initial_state(), move // 2, bool(move % 2))
        return RNRState(self.game, self.inner.apply(move), self.seat, self.restricted)

    def returns(self) -> tuple[float, float]:
        assert self.inner is not None
        r = self.inner.returns()
        return (r[self.seat], r[1 - self.seat])

    def information_state_key(self, player: int) -> str:
        assert self.inner is not None and not self.is_chance_node
        leduc_seat = self.seat if player == 0 else 1 - self.seat
        prefix = "H|" if player == 0 else "O|"
        return prefix + self.inner.information_state_key(leduc_seat)

    def observation(self, player: int) -> LeducObservation:
        assert self.inner is not None
        return self.inner.observation(self.seat if player == 0 else 1 - self.seat)


@dataclass(frozen=True, slots=True)
class RNRGame:
    model: GameStrategy
    p: float
    base: LeducGame = field(default_factory=LeducGame)

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")

    @property
    def name(self) -> str:
        return f"leduc-rnr(p={self.p:g})"

    @property
    def action_codes(self) -> tuple[str, ...]:
        return self.base.action_codes

    def initial_state(self) -> RNRState:
        return RNRState(self, None, 0, False)


def hero_table(average: GameStrategy) -> dict[str, dict[int, float]]:
    """Hero's part of a strategy over the modified game, keyed by Leduc keys."""
    return {k[2:]: dict(v) for k, v in average.items() if k.startswith("H|")}
