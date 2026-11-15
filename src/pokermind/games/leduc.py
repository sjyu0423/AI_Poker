"""Leduc Hold'em (Southey et al., 2005), matching OpenSpiel's ``leduc_poker``.

Rules and conventions (see ``docs/games/leduc.md``):

* **Deck:** six cards, ids 0-5. ``rank = id // 2`` (0 = J, 1 = Q, 2 = K) and
  ``suit = id % 2``. Two cards of each rank.
* **Antes:** each player puts in ``ante`` (default 1) before the deal.
* **Deal:** chance gives player 0 a private card (6 outcomes, 1/6 each), then
  player 1 (5 remaining, 1/5 each). The public card comes after round 1
  (4 remaining, 1/4 each). Outcomes are listed in increasing card id.
* **Rounds:** two betting rounds. **Player 0 acts first in both.** Players
  alternate within a round.
* **Actions:** ``FOLD = 0``, ``CALL = 1``, ``RAISE = 2``, as in OpenSpiel.
  * ``CALL`` matches the opponent's contribution, which is a *check* when the
    two contributions are already equal.
  * ``RAISE`` matches and adds the round's raise size (2 in round 1, 4 in
    round 2). It is a *bet* when nothing is owed.
  * ``FOLD`` is legal only when facing a bet (contributions unequal).
  * ``RAISE`` is legal only while fewer than ``max_raises`` (2) raises have been
    made this round.
* **End of a round:** any ``CALL`` that isn't the round's first action ends it
  (check-check, or a call of a raise). ``FOLD`` ends the hand.
* **Showdown:** a player whose card pairs the public card wins. Otherwise the
  higher rank wins. Equal ranks split the pot: both returns are 0, because
  contributions are equal at showdown.
* **Utilities:** net chips. The winner gains the loser's contribution. Range ±13
  with the defaults.

Information-set keys contain only the acting player's own card, the public
card once dealt, and the public action history of each round:
``"K1|cr"`` in round 1, ``"K1|J0|rrc|c"`` in round 2 (card = rank letter +
suit digit; actions: f, c, r).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from pokermind.games.base import CHANCE, TERMINAL

FOLD: Final = 0
CALL: Final = 1
RAISE: Final = 2
ACTION_CODES: Final = ("f", "c", "r")
ACTION_NAMES: Final = ("fold", "call", "raise")
NUM_CARDS: Final = 6
RANK_NAMES: Final = ("J", "Q", "K")


def rank(card: int) -> int:
    return card // 2


def card_name(card: int) -> str:
    return f"{RANK_NAMES[rank(card)]}{card % 2}"


@dataclass(frozen=True, slots=True)
class LeducConfig:
    ante: int = 1
    raise_sizes: tuple[int, int] = (2, 4)
    max_raises: int = 2

    def __post_init__(self) -> None:
        if self.ante <= 0 or any(r <= 0 for r in self.raise_sizes) or self.max_raises <= 0:
            raise ValueError(f"invalid Leduc config {self!r}")
        if len(self.raise_sizes) != 2:
            raise ValueError("exactly two raise sizes (one per round) are required")


@dataclass(frozen=True, slots=True)
class LeducObservation:
    """What ``player`` may know: own card, public card, public betting."""

    player: int
    private_card: int
    public_card: int | None
    rounds: tuple[tuple[int, ...], ...]

    @property
    def key(self) -> str:
        own = card_name(self.private_card)
        r1 = "".join(ACTION_CODES[a] for a in self.rounds[0])
        if self.public_card is None:
            return f"{own}|{r1}"
        r2 = "".join(ACTION_CODES[a] for a in self.rounds[1]) if len(self.rounds) > 1 else ""
        return f"{own}|{card_name(self.public_card)}|{r1}|{r2}"


class LeducState:
    """Immutable Leduc state. Build with :meth:`LeducGame.initial_state` and :meth:`apply`.

    Derived quantities (whose turn it is, contributions, raises this round)
    are computed once per state, from its parent, in :meth:`apply`. Only
    reachable states can be built.
    """

    __slots__ = (
        "_contrib",
        "_current",
        "_folder",
        "_raises",
        "config",
        "private",
        "public",
        "rounds",
    )

    config: LeducConfig
    private: tuple[int, ...]
    public: int | None
    rounds: tuple[tuple[int, ...], ...]
    _contrib: tuple[int, int]
    _raises: int
    _current: int
    _folder: int | None

    def __init__(
        self,
        config: LeducConfig,
        private: tuple[int, ...],
        public: int | None,
        rounds: tuple[tuple[int, ...], ...],
        contrib: tuple[int, int],
        raises: int,
        current: int,
        folder: int | None,
    ) -> None:
        # Written once here; __setattr__ below rejects every later assignment.
        setattr_ = object.__setattr__
        setattr_(self, "config", config)
        setattr_(self, "private", private)
        setattr_(self, "public", public)
        setattr_(self, "rounds", rounds)
        setattr_(self, "_contrib", contrib)
        setattr_(self, "_raises", raises)
        setattr_(self, "_current", current)
        setattr_(self, "_folder", folder)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("LeducState is immutable")

    def __repr__(self) -> str:
        names = tuple(card_name(c) for c in self.private)
        pub = None if self.public is None else card_name(self.public)
        return f"LeducState(private={names}, public={pub}, rounds={self.rounds})"

    # -- structure -------------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        return self._current

    @property
    def is_terminal(self) -> bool:
        return self._current == TERMINAL

    @property
    def is_chance_node(self) -> bool:
        return self._current == CHANCE

    @property
    def round_index(self) -> int:
        return len(self.rounds) - 1

    @property
    def contributions(self) -> tuple[int, int]:
        return self._contrib

    def chance_outcomes(self) -> tuple[tuple[int, float], ...]:
        if self._current != CHANCE:
            raise ValueError("not a chance node")
        used = {*self.private, *(() if self.public is None else (self.public,))}
        remaining = tuple(c for c in range(NUM_CARDS) if c not in used)
        p = 1.0 / len(remaining)
        return tuple((c, p) for c in remaining)

    def legal_actions(self) -> tuple[int, ...]:
        if self._current < 0:
            return ()
        facing = self._contrib[0] != self._contrib[1]
        actions = [FOLD] if facing else []
        actions.append(CALL)
        if self._raises < self.config.max_raises:
            actions.append(RAISE)
        return tuple(actions)

    # -- transitions -------------------------------------------------------------------------

    def apply(self, action: int) -> LeducState:
        if self._current == TERMINAL:
            raise ValueError("cannot act in a terminal state")
        if self._current == CHANCE:
            return self._deal(action)
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action!r} in {self!r}")
        player = self._current
        contrib = list(self._contrib)
        round_actions = (*self.rounds[-1], action)
        rounds = (*self.rounds[:-1], round_actions)
        if action == FOLD:
            return LeducState(
                self.config,
                self.private,
                self.public,
                rounds,
                self._contrib,
                self._raises,
                TERMINAL,
                player,
            )
        if action == RAISE:
            contrib[player] = max(contrib) + self.config.raise_sizes[self.round_index]
            return LeducState(
                self.config,
                self.private,
                self.public,
                rounds,
                (contrib[0], contrib[1]),
                self._raises + 1,
                1 - player,
                None,
            )
        # CALL: match the opponent; ends the round unless it is the round's first action.
        contrib[player] = max(contrib)
        new = (contrib[0], contrib[1])
        if len(round_actions) == 1:
            return LeducState(
                self.config, self.private, self.public, rounds, new, self._raises, 1 - player, None
            )
        if self.round_index == 0:  # deal the public card next
            return LeducState(self.config, self.private, None, rounds, new, 0, CHANCE, None)
        return LeducState(self.config, self.private, self.public, rounds, new, 0, TERMINAL, None)

    def _deal(self, card: int) -> LeducState:
        if card not in dict(self.chance_outcomes()):
            raise ValueError(f"card {card!r} is not available")
        if len(self.private) < 2:
            private = (*self.private, card)
            current = 0 if len(private) == 2 else CHANCE
            return LeducState(
                self.config, private, None, self.rounds, self._contrib, 0, current, None
            )
        return LeducState(
            self.config, self.private, card, (*self.rounds, ()), self._contrib, 0, 0, None
        )

    # -- outcomes ------------------------------------------------------------------------------

    def winner(self) -> int | None:
        """Seat that wins the pot, or ``None`` for a split. Terminal states only."""
        if self._current != TERMINAL:
            raise ValueError("winner is only defined at terminal states")
        if self._folder is not None:
            return 1 - self._folder
        assert self.public is not None
        strength = [self._hand_strength(p) for p in (0, 1)]
        if strength[0] == strength[1]:
            return None
        return 0 if strength[0] > strength[1] else 1

    def _hand_strength(self, player: int) -> tuple[int, int]:
        assert self.public is not None
        own = rank(self.private[player])
        return (int(own == rank(self.public)), own)

    def returns(self) -> tuple[float, float]:
        winner = self.winner()
        if winner is None:
            return 0.0, 0.0
        amount = float(self._contrib[1 - winner])
        return (amount, -amount) if winner == 0 else (-amount, amount)

    # -- observation ---------------------------------------------------------------------------

    def observation(self, player: int) -> LeducObservation:
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player!r}")
        if len(self.private) <= player:
            raise ValueError("player has no card yet")
        return LeducObservation(player, self.private[player], self.public, self.rounds)

    def information_state_key(self, player: int) -> str:
        return self.observation(player).key


@dataclass(frozen=True, slots=True)
class LeducGame:
    config: LeducConfig = field(default_factory=LeducConfig)

    @property
    def name(self) -> str:
        return "leduc"

    @property
    def action_codes(self) -> tuple[str, ...]:
        return ACTION_CODES

    def initial_state(self) -> LeducState:
        a = self.config.ante
        return LeducState(self.config, (), None, ((),), (a, a), 0, CHANCE, None)

    def state_from_moves(self, moves: tuple[int, ...]) -> LeducState:
        """Replay chance outcomes and actions from the root (tests, debugging)."""
        state = self.initial_state()
        for move in moves:
            state = state.apply(move)
        return state
