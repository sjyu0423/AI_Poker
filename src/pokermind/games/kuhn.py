"""Kuhn Poker: a minimal, exact two-player poker game.

Rules implemented (see ``docs/kuhn_poker.md`` for a full explanation):

* Two players, a three-card deck ``J < Q < K``.
* Each player antes ``config.ante`` chips and receives one private card.
  The remaining card is never revealed.
* A single betting round with a fixed bet size of ``config.bet_size`` and at
  most one bet (no raises). Player 0 acts first.
* When not facing a bet a player may CHECK or BET. When facing a bet a player
  may FOLD or CALL.
* If a player folds, the other player wins the pot. Otherwise the higher card
  wins the pot at showdown.

The module is pure: states are immutable and every transition returns a new
state. Agents must only ever see :class:`InfoState`, which by construction has
no field for the opponent's private card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from itertools import permutations
from typing import Final

import numpy as np

NUM_PLAYERS: Final = 2


class Card(IntEnum):
    """Card ranks. Integer order is showdown strength order."""

    J = 0
    Q = 1
    K = 2

    def __str__(self) -> str:
        return self.name


DECK: Final[tuple[Card, ...]] = (Card.J, Card.Q, Card.K)


class Action(Enum):
    """Betting actions.

    CHECK and BET are available when no bet is outstanding; FOLD and CALL are
    available when facing a bet. Each action has a one-letter code used in
    compact history strings: ``x`` check, ``b`` bet, ``c`` call, ``f`` fold.
    """

    CHECK = "x"
    BET = "b"
    CALL = "c"
    FOLD = "f"

    @property
    def code(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.name.lower()


History = tuple[Action, ...]

_X, _B, _C, _F = Action.CHECK, Action.BET, Action.CALL, Action.FOLD

# The complete game tree, written out explicitly. Keys are non-terminal
# histories; values are the legal actions in a fixed order (passive first).
_LEGAL_ACTIONS: Final[dict[History, tuple[Action, ...]]] = {
    (): (_X, _B),
    (_X,): (_X, _B),
    (_B,): (_F, _C),
    (_X, _B): (_F, _C),
}

TERMINAL_HISTORIES: Final[frozenset[History]] = frozenset(
    {
        (_X, _X),  # check, check -> showdown for the antes
        (_B, _F),  # bet, fold -> player 0 wins the antes
        (_B, _C),  # bet, call -> showdown for antes + bets
        (_X, _B, _F),  # check, bet, fold -> player 1 wins the antes
        (_X, _B, _C),  # check, bet, call -> showdown for antes + bets
    }
)


class IllegalActionError(ValueError):
    """Raised when an action is not legal in the current state."""


def history_key(history: History) -> str:
    """Compact string form of a history, e.g. ``"xb"`` for check-bet."""
    return "".join(a.code for a in history)


def _is_valid_history(history: History) -> bool:
    return history in _LEGAL_ACTIONS or history in TERMINAL_HISTORIES


@dataclass(frozen=True, slots=True)
class KuhnConfig:
    """Immutable game parameters."""

    ante: int = 1
    bet_size: int = 1

    def __post_init__(self) -> None:
        for name in ("ante", "bet_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class InfoState:
    """Everything one player is allowed to know at a decision point.

    Contains the observing player's seat, their own private card, and the
    public action history. It deliberately has no field that could hold the
    opponent's card.
    """

    player: int
    private_card: Card
    history: History

    @property
    def key(self) -> str:
        """Stable string key, e.g. ``"K|xb"``; useful for tabular methods.

        The key omits ``player``. That is safe in Kuhn Poker because players
        alternate strictly, so the seat to act is ``len(history) % 2`` and each
        key names exactly one information set. Games where the seat cannot be
        derived from the public history must include it in their keys.
        """
        return f"{self.private_card.name}|{history_key(self.history)}"

    @property
    def facing_bet(self) -> bool:
        return bool(self.history) and self.history[-1] is Action.BET


@dataclass(frozen=True, slots=True)
class KuhnState:
    """Full (omniscient) game state. Never pass this to an agent.

    ``cards[p]`` is player ``p``'s private card. ``history`` is the sequence of
    betting actions taken so far.
    """

    cards: tuple[Card, Card]
    history: History = ()
    config: KuhnConfig = field(default_factory=KuhnConfig)

    def __post_init__(self) -> None:
        if len(self.cards) != NUM_PLAYERS:
            raise ValueError(f"expected {NUM_PLAYERS} cards, got {len(self.cards)}")
        if not all(isinstance(c, Card) for c in self.cards):
            raise ValueError(f"cards must be Card values, got {self.cards!r}")
        if self.cards[0] == self.cards[1]:
            raise ValueError(f"players cannot hold the same card: {self.cards!r}")
        if not _is_valid_history(self.history):
            raise ValueError(f"not a reachable history: {self.history!r}")

    # -- structure ---------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.history in TERMINAL_HISTORIES

    @property
    def current_player(self) -> int:
        """Seat to act. Players alternate, starting with player 0."""
        if self.is_terminal:
            raise ValueError("terminal state has no player to act")
        return len(self.history) % NUM_PLAYERS

    def legal_actions(self) -> tuple[Action, ...]:
        """Legal actions, passive action first. Empty at terminal states."""
        return _LEGAL_ACTIONS.get(self.history, ())

    def is_legal(self, action: object) -> bool:
        return action in self.legal_actions()

    def apply(self, action: Action) -> KuhnState:
        """Return the successor state. Raises :class:`IllegalActionError`."""
        if self.is_terminal:
            raise IllegalActionError(f"cannot act in terminal state {history_key(self.history)!r}")
        if not self.is_legal(action):
            raise IllegalActionError(
                f"{action!r} is not legal after {history_key(self.history)!r}; "
                f"legal: {[str(a) for a in self.legal_actions()]}"
            )
        return KuhnState(self.cards, (*self.history, action), self.config)

    # -- chips -------------------------------------------------------------

    def contributions(self) -> tuple[int, int]:
        """Chips each player has put into the pot so far."""
        contrib = [self.config.ante, self.config.ante]
        for i, action in enumerate(self.history):
            if action in (Action.BET, Action.CALL):
                contrib[i % NUM_PLAYERS] += self.config.bet_size
        return contrib[0], contrib[1]

    @property
    def pot(self) -> int:
        return sum(self.contributions())

    def winner(self) -> int:
        """Seat that wins the pot. Only defined at terminal states."""
        if not self.is_terminal:
            raise ValueError("winner is only defined at terminal states")
        if self.history[-1] is Action.FOLD:
            folder = (len(self.history) - 1) % NUM_PLAYERS
            return 1 - folder
        return 0 if self.cards[0] > self.cards[1] else 1

    def payoffs(self) -> tuple[int, int]:
        """Net chip result for each player. Zero-sum by construction."""
        winner = self.winner()
        loser = 1 - winner
        amount = self.contributions()[loser]
        result = [0, 0]
        result[winner] = amount
        result[loser] = -amount
        return result[0], result[1]

    # -- observation -------------------------------------------------------

    def information_state(self, player: int) -> InfoState:
        """The view of ``player``: own card and public history only."""
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player!r}")
        return InfoState(player=player, private_card=self.cards[player], history=self.history)


def all_deals() -> tuple[tuple[Card, Card], ...]:
    """All six ordered deals ``(card_p0, card_p1)``."""
    return tuple((a, b) for a, b in permutations(DECK, NUM_PLAYERS))


def initial_state(cards: tuple[Card, Card], config: KuhnConfig | None = None) -> KuhnState:
    """Start-of-hand state for a specific deal."""
    return KuhnState(cards=cards, history=(), config=config or KuhnConfig())


def deal(rng: np.random.Generator, config: KuhnConfig | None = None) -> KuhnState:
    """Deal a uniformly random hand using ``rng``."""
    order = rng.permutation(len(DECK))
    return initial_state((DECK[int(order[0])], DECK[int(order[1])]), config)
