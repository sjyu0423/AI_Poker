"""Best responses, policy mixtures and exploitability budgets for hero (M6).

Hero and opponent alternate seats, and strategy tables are seat-free (each
information set belongs to one seat), so hero's best response to an
opponent table is the union of the M3 evaluator's best responses for seat 0
and seat 1.

**Mixtures.** ``realization_mixture`` combines two hero strategies the way a
*mixed* strategy would: with probability ``1 − α`` play the base strategy for
the whole hand, otherwise the response. By Kuhn's theorem this equals the
behavioural strategy

    π_α(a | I) = ((1 − α) x_b(I) π_b(a | I) + α x_r(I) π_r(a | I)) / ((1 − α) x_b(I) + α x_r(I))

where ``x(I)`` is a strategy's own probability of reaching ``I``. Two
consequences, both tested:

* the expected payoff against any fixed opponent is **linear** in ``α``;
* the exploitability (NashConv / 2) is **convex** in ``α`` (a maximum of
  linear functions), so ``{α : exploitability ≤ budget}`` is an interval
  ``[0, α*]`` and can be found by search.

Mixing per information set with fixed weights (``(1 − α) π_b + α π_r``) is a
different policy with neither property. It is not used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from pokermind.evaluation.game_exact import (
    GameInfoSet,
    best_response,
    expected_value,
    exploitability_report,
)
from pokermind.games.leduc import LeducGame, LeducState
from pokermind.strategy import GameStrategy

Policy = dict[str, dict[int, float]]
ALPHA_GRID: Final = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def hero_best_response(
    game: LeducGame, opponent: GameStrategy, specs: dict[str, GameInfoSet]
) -> Policy:
    """Pure best response of hero (both seats) to ``opponent``."""
    out: Policy = {}
    for seat in (0, 1):
        br = best_response(game, opponent, seat, infosets=specs)
        for key, chosen in br.policy.items():
            out[key] = {a: float(a == chosen) for a in specs[key].actions}
    return out


def hero_value(
    game: LeducGame, hero: GameStrategy, opponent: GameStrategy, specs: dict[str, GameInfoSet]
) -> float:
    """Hero's exact expected payoff per hand, seats alternating."""
    as0 = expected_value(game, (hero, opponent), infosets=specs)
    as1 = -expected_value(game, (opponent, hero), infosets=specs)
    return (as0 + as1) / 2


def exploitability(game: LeducGame, policy: GameStrategy, specs: dict[str, GameInfoSet]) -> float:
    return exploitability_report(game, policy, infosets=specs).exploitability


def own_reach(game: LeducGame, policy: GameStrategy) -> dict[str, float]:
    """Each information set's own reach probability under ``policy`` (perfect recall:
    the same for every history in the set)."""
    reach: dict[str, float] = {}

    def walk(state: LeducState, x: tuple[float, float]) -> None:
        if state.is_terminal:
            return
        if state.is_chance_node:
            for card, _ in state.chance_outcomes():
                walk(state.apply(card), x)
            return
        player = state.current_player
        key = state.information_state_key(player)
        reach.setdefault(key, x[player])
        for a in state.legal_actions():
            p = policy[key][a]
            walk(state.apply(a), (x[0] * p, x[1]) if player == 0 else (x[0], x[1] * p))

    walk(game.initial_state(), (1.0, 1.0))
    return reach


def realization_mixture(
    base: GameStrategy,
    response: GameStrategy,
    alpha: float,
    base_reach: Mapping[str, float],
    response_reach: Mapping[str, float],
) -> Policy:
    """Behavioural form of the mixed strategy ``(1 − α)·base + α·response``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    out: Policy = {}
    for key, dist in base.items():
        wb = (1.0 - alpha) * base_reach[key]
        wr = alpha * response_reach[key]
        total = wb + wr
        if total <= 0.0:
            out[key] = dict(dist) if alpha < 1.0 else dict(response[key])
            continue
        out[key] = {a: (wb * dist[a] + wr * response[key][a]) / total for a in dist}
    return out


@dataclass(slots=True)
class BudgetSearch:
    alpha: float
    """Largest mixture weight found within the budget (0 = base strategy)."""
    exploitability: float
    """Exact exploitability of the returned mixture."""
    evaluated: dict[float, float] = field(default_factory=dict)
    """Every α whose mixture was evaluated exactly, with its exploitability."""


class Mixer:
    """Mixtures of a fixed base strategy with one response, evaluated exactly."""

    def __init__(
        self,
        game: LeducGame,
        specs: dict[str, GameInfoSet],
        base: GameStrategy,
        response: GameStrategy,
        base_exploitability: float | None = None,
    ) -> None:
        self.game, self.specs = game, specs
        self.base, self.response = base, response
        self.base_reach = own_reach(game, base)
        self.response_reach = own_reach(game, response)
        self._cache: dict[float, float] = {}
        if base_exploitability is not None:
            self._cache[0.0] = base_exploitability

    def policy(self, alpha: float) -> Policy:
        return realization_mixture(
            self.base, self.response, alpha, self.base_reach, self.response_reach
        )

    def exploitability(self, alpha: float) -> float:
        if alpha not in self._cache:
            self._cache[alpha] = exploitability(self.game, self.policy(alpha), self.specs)
        return self._cache[alpha]

    def largest_alpha(
        self, budget: float, cap: float = 1.0, grid: Sequence[float] = ALPHA_GRID, refine: int = 6
    ) -> BudgetSearch:
        """Largest ``α ≤ cap`` whose exact exploitability is ``≤ budget``.

        Ascends ``grid`` until the first violation, then bisects ``refine``
        times between the last feasible and first infeasible weight. Relies on
        convexity (feasible set = an interval containing 0) and returns 0 if
        even the base strategy exceeds the budget.
        """
        if self.exploitability(0.0) > budget:
            return BudgetSearch(0.0, self.exploitability(0.0), dict(self._cache))
        points = sorted({a for a in grid if a < cap} | {cap})
        lo, hi = 0.0, None
        for a in points:
            if a == 0.0:
                continue
            if self.exploitability(a) <= budget:
                lo = a
            else:
                hi = a
                break
        if hi is not None:
            for _ in range(refine):
                mid = 0.5 * (lo + hi)
                if self.exploitability(mid) <= budget:
                    lo = mid
                else:
                    hi = mid
        return BudgetSearch(lo, self.exploitability(lo), dict(self._cache))
