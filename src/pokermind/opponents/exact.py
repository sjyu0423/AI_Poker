"""Exact public-context marginals of an opponent against a fixed hero (M5).

For hero strategy ``h`` and opponent policy ``π`` (seats alternating, each with
probability ½ per hand), walk the whole Leduc tree and accumulate, at every
opponent decision node with public context ``c``:

* ``weight[c]  += ½ · π_chance · π_hero · π_opp``: the expected number of
  times per hand the opponent acts in context ``c``;
* ``joint[c][a] += ½ · π_chance · π_hero · π_opp · π(a | I)``.

``marginal[c] = joint[c] / weight[c]`` is the oracle public model: the best
prediction of the opponent's action from public information.
``card_aware_entropy`` is the weight-averaged entropy of ``π`` itself, the
expected log loss of a predictor that also knew the opponent's card.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pokermind.games.leduc import LeducGame, LeducState
from pokermind.opponents.context import action_set_name, public_context
from pokermind.strategy import GameStrategy

Prediction = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ContextStats:
    weight: dict[str, float]
    """Expected opponent decisions per hand in each context (only reached contexts)."""
    marginal: dict[str, Prediction]
    """Oracle public-context action distribution, in legal-action order."""
    legal: dict[str, tuple[int, ...]]
    card_aware_entropy: float
    """Expected per-decision log loss of the card-aware oracle."""

    @property
    def decisions_per_hand(self) -> float:
        return sum(self.weight.values())


def context_stats(game: LeducGame, hero: GameStrategy, opponent: GameStrategy) -> ContextStats:
    weight: dict[str, float] = {}
    joint: dict[str, Prediction] = {}
    legal: dict[str, tuple[int, ...]] = {}
    entropy = 0.0

    def walk(state: LeducState, reach: float, opp_seat: int) -> None:
        nonlocal entropy
        if state.is_terminal or reach == 0.0:
            return
        if state.is_chance_node:
            for card, p in state.chance_outcomes():
                walk(state.apply(card), reach * p, opp_seat)
            return
        player = state.current_player
        key = state.information_state_key(player)
        actions = state.legal_actions()
        if player == opp_seat:
            dist = opponent[key]
            ctx = public_context(state)
            probs = np.array([dist[a] for a in actions])
            weight[ctx] = weight.get(ctx, 0.0) + reach
            joint[ctx] = joint.get(ctx, np.zeros(len(actions))) + reach * probs
            legal[ctx] = actions
            entropy -= reach * sum(p * math.log(p) for p in probs if p > 0)
            for a, p in zip(actions, probs, strict=True):
                walk(state.apply(a), reach * p, opp_seat)
        else:
            dist = hero[key]
            for a in actions:
                walk(state.apply(a), reach * dist[a], opp_seat)

    for seat in (0, 1):
        walk(game.initial_state(), 0.5, seat)
    total = sum(weight.values())
    marginal = {c: joint[c] / weight[c] for c in weight}
    return ContextStats(weight, marginal, legal, entropy / total)


@dataclass(frozen=True, slots=True)
class ExpectedScores:
    log_loss: float
    brier: float
    accuracy: float
    oracle_log_loss: float
    card_aware_log_loss: float

    @property
    def excess_log_loss(self) -> float:
        """Weighted KL(oracle ‖ prediction): distance from the best public model."""
        return self.log_loss - self.oracle_log_loss


def expected_scores(
    stats: ContextStats,
    predict: Callable[[str, tuple[int, ...]], Prediction],
    floor: float = 1e-12,
) -> ExpectedScores:
    """Per-decision scores in expectation over future hands, computed exactly."""
    total = stats.decisions_per_hand
    log_loss = brier = accuracy = oracle = 0.0
    for ctx, w in stats.weight.items():
        p = stats.marginal[ctx]
        q = predict(ctx, stats.legal[ctx])
        log_loss += w * float(-(p * np.log(np.maximum(q, floor))).sum())
        brier += w * float((q * q).sum() - 2 * (p * q).sum() + 1.0)
        accuracy += w * float(p[int(np.argmax(q))])
        oracle += w * float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return ExpectedScores(
        log_loss / total, brier / total, accuracy / total, oracle / total, stats.card_aware_entropy
    )


TRAITS: dict[str, tuple[int, tuple[str, ...]]] = {
    "fold_to_bet": (0, ("fcr", "fc")),
    "raise_facing_bet": (2, ("fcr",)),
    "bet_unopened": (2, ("cr",)),
}
"""Summary tendencies: (action, action sets) -> reach-weighted action probability."""


def weighted_trait(
    stats: ContextStats,
    predict: Callable[[str, tuple[int, ...]], Prediction],
    trait: str,
) -> float:
    """Reach-weighted probability of the trait's action over the contexts whose legal
    set is one of the trait's action sets (weights from the true opponent)."""
    action, sets = TRAITS[trait]
    num = den = 0.0
    for ctx, w in stats.weight.items():
        legal = stats.legal[ctx]
        if action_set_name(legal) not in sets:
            continue
        num += w * float(predict(ctx, legal)[legal.index(action)])
        den += w
    return num / den if den else float("nan")
