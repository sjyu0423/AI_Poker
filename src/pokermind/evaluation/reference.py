"""Known Kuhn Poker equilibrium (Kuhn, 1950) in M0's action names and payoffs.

Valid for the default :class:`~pokermind.games.kuhn.KuhnConfig` (ante 1,
bet 1) only. Derived from the literature, not from CFR. See
``docs/concepts/cfr.md`` §3 for the table.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Final

from pokermind.games.kuhn import Action
from pokermind.strategy import Strategy

GAME_VALUE: Final = Fraction(-1, 18)
"""Seat 0's expected payoff under any equilibrium (chips per hand)."""

ALPHA_MAX: Final = Fraction(1, 3)

_X, _B, _C, _F = Action.CHECK, Action.BET, Action.CALL, Action.FOLD

PLAYER0_KEYS: Final = ("J|", "Q|", "K|", "J|xb", "Q|xb", "K|xb")
PLAYER1_KEYS: Final = ("J|b", "Q|b", "K|b", "J|x", "Q|x", "K|x")


def kuhn_equilibrium(alpha: Fraction | float = 0) -> dict[str, dict[Action, Fraction]]:
    """The equilibrium with parameter ``alpha`` in ``[0, 1/3]``, as exact fractions."""
    a = Fraction(alpha)
    if not 0 <= a <= ALPHA_MAX:
        raise ValueError(f"alpha must be in [0, 1/3], got {alpha!r}")
    one, zero, third = Fraction(1), Fraction(0), Fraction(1, 3)

    def open_(bet: Fraction) -> dict[Action, Fraction]:
        return {_X: one - bet, _B: bet}

    def facing(call: Fraction) -> dict[Action, Fraction]:
        return {_F: one - call, _C: call}

    return {
        "J|": open_(a),
        "Q|": open_(zero),
        "K|": open_(3 * a),
        "J|xb": facing(zero),
        "Q|xb": facing(a + third),
        "K|xb": facing(one),
        "J|b": facing(zero),
        "Q|b": facing(third),
        "K|b": facing(one),
        "J|x": open_(third),
        "Q|x": open_(zero),
        "K|x": open_(one),
    }


def as_float_strategy(
    strategy: dict[str, dict[Action, Fraction]],
) -> dict[str, dict[Action, float]]:
    return {k: {a: float(p) for a, p in d.items()} for k, d in strategy.items()}


def estimated_alpha(strategy: Strategy) -> float:
    """Player 0's bluff frequency with J, the family parameter, clipped to [0, 1/3]."""
    return min(max(strategy["J|"][_B], 0.0), float(ALPHA_MAX))


def max_deviation_from_equilibrium(strategy: Strategy) -> float:
    """Largest absolute probability gap to the equilibrium with the same ``alpha``.

    Player 1's equilibrium is unique; player 0's is matched to the family member
    whose ``alpha`` equals the strategy's J-bluff frequency. Measures distance
    in strategy space; exploitability measures it in chips.
    """
    target = kuhn_equilibrium(Fraction(estimated_alpha(strategy)))
    return max(
        abs(strategy[key][action] - float(prob))
        for key, dist in target.items()
        for action, prob in dist.items()
    )
