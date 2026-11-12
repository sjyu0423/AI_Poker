"""Tabular vanilla Counterfactual Regret Minimization for Kuhn Poker.

This is the original algorithm of Zinkevich et al. (2007), with simultaneous
updates and full tree traversal. (OpenSpiel's public ``CFRSolver`` uses
alternating updates; its base ``_CFRSolver`` with ``alternating_updates=False``
matches this implementation to rounding error, see
``tests/test_openspiel_crosscheck.py``.) The
mathematics is explained in ``docs/concepts/cfr.md``. In short, each
iteration:

1. freezes the current strategy ``σ^t`` from cumulative regrets (regret
   matching, uniform when no regret is positive);
2. walks the whole game tree: chance first (each deal with probability
   ``1/6``), carrying the reach probabilities ``(π_0, π_1, π_c)``;
3. at every decision node of player ``i`` in information set ``I``, adds
   ``π_{-i}(h) · (u_i(h·a) − u_i(h))`` to ``R(I, a)`` and
   ``π_i(h) · σ^t(I, a)`` to the average-strategy accumulator ``S(I, a)``.

Tables are keyed by :attr:`pokermind.games.kuhn.InfoState.key`, built from
:meth:`KuhnState.information_state`, so they cannot contain the opponent's
card. Training is deterministic: it uses no randomness, and the deals and
actions are visited in a fixed order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pokermind.games.kuhn import (
    NUM_PLAYERS,
    Action,
    KuhnConfig,
    KuhnState,
    all_deals,
    initial_state,
)

FloatArray = npt.NDArray[np.float64]


def regret_matching(regrets: FloatArray) -> FloatArray:
    """Strategy proportional to positive regret; uniform if none is positive.

    ``regrets`` is indexed like the information set's legal actions, so
    illegal actions can never receive probability.
    """
    if regrets.ndim != 1 or regrets.size == 0:
        raise ValueError(f"expected a non-empty 1-D regret vector, got shape {regrets.shape}")
    if not np.all(np.isfinite(regrets)):
        raise ValueError(f"regrets must be finite, got {regrets}")
    positive = np.maximum(regrets, 0.0)
    total = float(positive.sum())
    if total > 0.0:
        return positive / total
    return np.full(regrets.size, 1.0 / regrets.size)


def normalize_or_uniform(weights: FloatArray) -> FloatArray:
    """Normalize non-negative weights; uniform if they sum to zero (never reached)."""
    total = float(weights.sum())
    if total > 0.0:
        return weights / total
    return np.full(weights.size, 1.0 / weights.size)


@dataclass(slots=True)
class InfoSetTable:
    """Per-information-set CFR data. Arrays are indexed like ``actions``."""

    player: int
    actions: tuple[Action, ...]
    regret_sum: FloatArray
    strategy_sum: FloatArray


class CFRSolver:
    """Vanilla CFR. Call :meth:`iterate`, then read :meth:`average_strategy`."""

    def __init__(self, config: KuhnConfig | None = None) -> None:
        self.config = config or KuhnConfig()
        self.iterations = 0
        self._deals = all_deals()
        self._chance_prob = 1.0 / len(self._deals)
        self.tables: dict[str, InfoSetTable] = {}
        for cards in self._deals:
            self._register(initial_state(cards, self.config))
        # σ^t, frozen for the duration of each iteration.
        self._current: dict[str, FloatArray] = self._compute_current()

    def _register(self, state: KuhnState) -> None:
        """Create a table entry for every information set (fixed visiting order)."""
        if state.is_terminal:
            return
        player = state.current_player
        key = state.information_state(player).key
        actions = state.legal_actions()
        if key not in self.tables:
            n = len(actions)
            self.tables[key] = InfoSetTable(player, actions, np.zeros(n), np.zeros(n))
        elif self.tables[key].actions != actions:  # pragma: no cover - engine invariant
            raise AssertionError(f"inconsistent legal actions at {key!r}")
        for action in actions:
            self._register(state.apply(action))

    def _compute_current(self) -> dict[str, FloatArray]:
        return {key: regret_matching(t.regret_sum) for key, t in self.tables.items()}

    # -- training ----------------------------------------------------------

    def iterate(self, num_iterations: int = 1) -> None:
        """Run ``num_iterations`` full CFR iterations."""
        if num_iterations < 0:
            raise ValueError(f"num_iterations must be non-negative, got {num_iterations}")
        for _ in range(num_iterations):
            for cards in self._deals:
                state = initial_state(cards, self.config)
                self._traverse(state, (1.0, 1.0), self._chance_prob)
            self.iterations += 1
            self._current = self._compute_current()

    def _traverse(
        self, state: KuhnState, reach: tuple[float, float], chance_reach: float
    ) -> tuple[float, float]:
        """Return each player's expected utility at ``state``, given it is reached.

        ``reach[p]`` is ``π_p(h)``, player ``p``'s own contribution, and
        ``chance_reach`` is ``π_c(h)``.
        """
        if state.is_terminal:
            p0, p1 = state.payoffs()
            return float(p0), float(p1)

        player = state.current_player
        key = state.information_state(player).key
        table = self.tables[key]
        sigma = self._current[key]

        child_values = np.empty((len(table.actions), NUM_PLAYERS))
        for idx, action in enumerate(table.actions):
            child_reach = (
                (reach[0] * sigma[idx], reach[1])
                if player == 0
                else (reach[0], reach[1] * sigma[idx])
            )
            child_values[idx] = self._traverse(state.apply(action), child_reach, chance_reach)

        node_values = sigma @ child_values  # shape (NUM_PLAYERS,)

        counterfactual_reach = reach[1 - player] * chance_reach  # π_{-i}(h)
        table.regret_sum += counterfactual_reach * (child_values[:, player] - node_values[player])
        table.strategy_sum += reach[player] * sigma
        return float(node_values[0]), float(node_values[1])

    # -- results -----------------------------------------------------------

    def current_strategy(self) -> dict[str, dict[Action, float]]:
        """``σ^{t+1}``: what regret matching plays next. Does not converge in general."""
        return {
            key: dict(zip(t.actions, map(float, self._current[key]), strict=True))
            for key, t in self.tables.items()
        }

    def average_strategy(self) -> dict[str, dict[Action, float]]:
        """Reach-weighted average strategy ``σ̄^T``; converges to a Nash equilibrium."""
        return {
            key: dict(zip(t.actions, map(float, normalize_or_uniform(t.strategy_sum)), strict=True))
            for key, t in self.tables.items()
        }

    def cumulative_regrets(self) -> dict[str, dict[Action, float]]:
        return {
            key: dict(zip(t.actions, map(float, t.regret_sum), strict=True))
            for key, t in self.tables.items()
        }


def train_cfr(iterations: int, config: KuhnConfig | None = None) -> CFRSolver:
    """Convenience: a solver after ``iterations`` iterations."""
    solver = CFRSolver(config)
    solver.iterate(iterations)
    return solver


__all__ = [
    "CFRSolver",
    "InfoSetTable",
    "normalize_or_uniform",
    "regret_matching",
    "train_cfr",
]
