"""Vanilla CFR for any two-player zero-sum ``Game`` (M3).

The same algorithm as :mod:`pokermind.learning.cfr` (Zinkevich et al. 2007:
simultaneous updates, full traversal, frozen ``σ^t`` per iteration, regret
matching with uniform fallback, reach-weighted average strategy), but with
chance handled as explicit tree nodes instead of a Kuhn-specific loop over
deals:

* at a chance node, every outcome is visited with ``π_c`` multiplied by its
  probability, so conditional probabilities (Leduc's 1/6, then 1/5, then 1/4)
  enter the counterfactual reach exactly;
* at a decision node of player ``i`` in information set ``I``:
  ``R(I, a) += π_{-i}(h) · (u_i(h·a) − u_i(h))`` with ``π_{-i} = π_opp · π_c``, and
  ``S(I, a) += π_i(h) · σ^t(I, a)``.

The M1 implementation is kept unchanged as the regression reference. On
``KuhnGame`` this solver performs the same floating-point operations in the
same order (the root chance weight ``1.0 · (1/6)`` equals M1's ``1/6``), so it
reproduces M1's strategies bit for bit. Tests check this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokermind.games.base import NUM_PLAYERS, Game, GameState
from pokermind.learning.cfr import FloatArray, normalize_or_uniform, regret_matching


@dataclass(slots=True)
class GameInfoSetTable:
    player: int
    actions: tuple[int, ...]
    regret_sum: FloatArray
    strategy_sum: FloatArray


class GameCFRSolver:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.iterations = 0
        self.decision_visits = 0
        """Decision-node visits accumulated over all iterations."""
        self.tables: dict[str, GameInfoSetTable] = {}
        self._register(game.initial_state())
        self._current: dict[str, FloatArray] = self._compute_current()

    def _register(self, state: GameState) -> None:
        if state.is_terminal:
            return
        if state.is_chance_node:
            for outcome, _ in state.chance_outcomes():
                self._register(state.apply(outcome))
            return
        player = state.current_player
        key = state.information_state_key(player)
        actions = state.legal_actions()
        table = self.tables.get(key)
        if table is None:
            n = len(actions)
            self.tables[key] = GameInfoSetTable(player, actions, np.zeros(n), np.zeros(n))
        elif table.actions != actions or table.player != player:
            raise ValueError(f"inconsistent information set {key!r}: imperfect key")
        for action in actions:
            self._register(state.apply(action))

    def _compute_current(self) -> dict[str, FloatArray]:
        return {key: regret_matching(t.regret_sum) for key, t in self.tables.items()}

    def iterate(self, num_iterations: int = 1) -> None:
        if num_iterations < 0:
            raise ValueError(f"num_iterations must be non-negative, got {num_iterations}")
        for _ in range(num_iterations):
            self._traverse(self.game.initial_state(), (1.0, 1.0), 1.0)
            self.iterations += 1
            self._current = self._compute_current()

    def _traverse(
        self, state: GameState, reach: tuple[float, float], chance_reach: float
    ) -> tuple[float, float]:
        """Each player's expected utility at ``state``, given that it is reached."""
        if state.is_terminal:
            return state.returns()

        if state.is_chance_node:
            v0 = v1 = 0.0
            for outcome, prob in state.chance_outcomes():
                c0, c1 = self._traverse(state.apply(outcome), reach, chance_reach * prob)
                v0 += prob * c0
                v1 += prob * c1
            return v0, v1

        self.decision_visits += 1
        player = state.current_player
        key = state.information_state_key(player)
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

        node_values = sigma @ child_values

        counterfactual_reach = reach[1 - player] * chance_reach
        table.regret_sum += counterfactual_reach * (child_values[:, player] - node_values[player])
        table.strategy_sum += reach[player] * sigma
        return float(node_values[0]), float(node_values[1])

    def current_strategy(self) -> dict[str, dict[int, float]]:
        return {
            key: dict(zip(t.actions, map(float, self._current[key]), strict=True))
            for key, t in self.tables.items()
        }

    def average_strategy(self) -> dict[str, dict[int, float]]:
        return {
            key: dict(zip(t.actions, map(float, normalize_or_uniform(t.strategy_sum)), strict=True))
            for key, t in self.tables.items()
        }

    def cumulative_regrets(self) -> dict[str, dict[int, float]]:
        return {
            key: dict(zip(t.actions, map(float, t.regret_sum), strict=True))
            for key, t in self.tables.items()
        }

    def table_bytes(self) -> int:
        """Bytes held by the regret and strategy arrays (excluding dict overhead)."""
        return sum(t.regret_sum.nbytes + t.strategy_sum.nbytes for t in self.tables.values())


def train_game_cfr(game: Game, iterations: int) -> GameCFRSolver:
    solver = GameCFRSolver(game)
    solver.iterate(iterations)
    return solver
