"""The minimal game interface shared by Kuhn and Leduc Poker (M3).

Design notes are in ``docs/design/leduc_interface.md``. The interface is
deliberately small:

* every state is a chance node, a decision node, or terminal;
* chance nodes list their outcomes with probabilities, so CFR and the exact
  evaluator weight them explicitly and the simulator samples them;
* actions are small integers defined by each game, listed in a fixed order;
* ``information_state_key(player)`` and ``observation(player)`` are derived only
  from what that player may observe. They never contain the opponent's private
  cards.

Two-player, zero-sum games only: the evaluator's ``NashConv = br_0 + br_1``
identity relies on it.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

CHANCE: Final = -1
"""``current_player`` at chance nodes."""
TERMINAL: Final = -4
"""``current_player`` at terminal states (OpenSpiel's convention)."""
NUM_PLAYERS: Final = 2


@runtime_checkable
class Observation(Protocol):
    """What one player may know at a decision point."""

    @property
    def player(self) -> int: ...

    @property
    def key(self) -> str: ...


class GameState(Protocol):
    @property
    def current_player(self) -> int:
        """0 or 1 at decision nodes, :data:`CHANCE` or :data:`TERMINAL` otherwise."""
        ...

    @property
    def is_terminal(self) -> bool: ...

    @property
    def is_chance_node(self) -> bool: ...

    def chance_outcomes(self) -> tuple[tuple[int, float], ...]:
        """``(outcome, probability)`` pairs at a chance node, in a fixed order."""
        ...

    def legal_actions(self) -> tuple[int, ...]:
        """Legal actions at a decision node, in a fixed order."""
        ...

    def apply(self, action: int) -> GameState:
        """Successor after a chance outcome or a player action. Never mutates."""
        ...

    def returns(self) -> tuple[float, float]:
        """Net utility for each player at a terminal state (zero-sum)."""
        ...

    def information_state_key(self, player: int) -> str:
        """Stable key of ``player``'s information set; opponent-free."""
        ...

    def observation(self, player: int) -> Observation: ...


class Game(Protocol):
    @property
    def name(self) -> str:
        """Short name recorded in strategy files, e.g. ``"kuhn"``."""
        ...

    @property
    def action_codes(self) -> tuple[str, ...]:
        """One-letter code for each action id, used in strategy files."""
        ...

    def initial_state(self) -> GameState:
        """The root of the game tree (a chance node: the deal)."""
        ...
