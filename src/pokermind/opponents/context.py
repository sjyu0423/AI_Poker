"""Public betting contexts and public hand records (M5).

A *public context* identifies an opponent decision using only public
information: the round, the public card's rank (round 2), and the betting
history of each round. See ``docs/design/opponent_modeling.md`` §2.

``public_record`` is the dealer-side boundary: it sees the whole hand but
emits only what the observing player legitimately learns. The opponent's card
is included only when the hand reached a showdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pokermind.games.leduc import (
    ACTION_CODES,
    CALL,
    FOLD,
    RAISE,
    RANK_NAMES,
    LeducGame,
    LeducState,
    rank,
)

ACTION_SETS: Final[dict[tuple[int, ...], str]] = {
    (CALL, RAISE): "cr",
    (FOLD, CALL, RAISE): "fcr",
    (FOLD, CALL): "fc",
}
NUM_LEVELS: Final = 4


def action_set_name(legal_actions: tuple[int, ...]) -> str:
    try:
        return ACTION_SETS[legal_actions]
    except KeyError:
        raise ValueError(f"not a Leduc decision action set: {legal_actions!r}") from None


def _codes(actions: tuple[int, ...]) -> str:
    return "".join(ACTION_CODES[a] for a in actions)


def public_context(state: LeducState) -> str:
    """``"R1:<round 1>"`` or ``"R2:<public rank>:<round 1>:<round 2>"`` at a decision node."""
    if state.is_terminal or state.is_chance_node:
        raise ValueError("public contexts exist only at decision nodes")
    if state.public is None:
        return f"R1:{_codes(state.rounds[0])}"
    return (
        f"R2:{RANK_NAMES[rank(state.public)]}:{_codes(state.rounds[0])}:{_codes(state.rounds[1])}"
    )


def context_levels(context: str, legal_actions: tuple[int, ...]) -> tuple[str, str, str, str]:
    """Keys from finest to coarsest: full context, round + current-round history,
    round + action set, action set. Every level shares ``legal_actions``."""
    parts = context.split(":")
    if parts[0] not in ("R1", "R2") or len(parts) != (2 if parts[0] == "R1" else 4):
        raise ValueError(f"malformed context {context!r}")
    action_set = action_set_name(legal_actions)
    return (
        context,
        f"{parts[0]}:{parts[-1]}",
        f"{parts[0]}:{action_set}",
        action_set,
    )


def context_class(context: str, legal_actions: tuple[int, ...]) -> str:
    """Coarse class for reporting, e.g. ``"R2:fcr"`` (round 2, facing a bet, may raise)."""
    return context_levels(context, legal_actions)[2]


@dataclass(frozen=True, slots=True)
class PublicDecision:
    context: str
    legal_actions: tuple[int, ...]
    action: int


@dataclass(frozen=True, slots=True)
class PublicHandRecord:
    """What the observer learns about the opponent in one hand."""

    hand_index: int
    opponent_seat: int
    decisions: tuple[PublicDecision, ...]
    """Every opponent decision, in order, with its public context."""
    showdown_card: int | None
    """The opponent's card if the hand reached a showdown, else ``None``."""
    ended_by_fold: bool


def public_record(
    game: LeducGame, moves: tuple[int, ...], opponent_seat: int, hand_index: int = 0
) -> PublicHandRecord:
    """Replay a complete hand and keep only the observer-visible facts."""
    if opponent_seat not in (0, 1):
        raise ValueError(f"opponent_seat must be 0 or 1, got {opponent_seat!r}")
    state = game.initial_state()
    decisions = []
    for move in moves:
        if not state.is_chance_node and state.current_player == opponent_seat:
            decisions.append(PublicDecision(public_context(state), state.legal_actions(), move))
        state = state.apply(move)
    if not state.is_terminal:
        raise ValueError("moves do not describe a complete hand")
    ended_by_fold = any(r and r[-1] == FOLD for r in state.rounds)
    showdown = None if ended_by_fold else state.private[opponent_seat]
    return PublicHandRecord(hand_index, opponent_seat, tuple(decisions), showdown, ended_by_fold)
