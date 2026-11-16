"""Choosing hero's RNR strategy under an exact exploitability budget (M7B).

This is the acting procedure that M7A evaluated, factored out of the
experiment and made faster. It also has an optional uncertainty-aware rule.

1. **Candidates.** RNR strategies against the opponent estimate for a grid of
   ``p``, all solved in one :class:`~pokermind.robust.batch.BatchedRNR`
   batch. Each candidate gets:
   * its exact exploitability and exact value against the estimate, from
     the batched :class:`~pokermind.robust.batch.TreeEvaluator`;
   * M7A's stopping rule: the grid is cut after two consecutive candidates
     above the largest budget. This reproduces M7A's candidate sets exactly.
2. **Rule.**
   * ``point`` (M7A): the feasible candidate with the highest value against
     the point estimate.
   * ``lcb``: each candidate's gain over CFR is evaluated against posterior
     samples of the opponent. The score is
     ``mean − z·κ·sd``, where κ is a calibration inflation. Hero plays the
     feasible candidate with the highest score, or the CFR baseline if no
     score is positive.
3. **Certificate.** The chosen strategy's exploitability is computed again
   by the M3 recursive evaluator, independently of the batched one. A
   candidate that fails is discarded and selection repeats.

Only the estimate, posterior samples and public baselines are inputs. The true
opponent strategy is never passed to this module.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from pokermind.adaptive.response import exploitability
from pokermind.evaluation.game_exact import GameInfoSet
from pokermind.games.leduc import LeducGame
from pokermind.robust.batch import BatchedRNR, TreeEvaluator
from pokermind.robust.tree import CompiledTree, FloatArray
from pokermind.strategy import GameStrategy

P_GRID: Final = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
"""M7A's restriction grid."""
TOLERANCE: Final = 1e-12
RULES: Final = ("point", "lcb")


@dataclass(frozen=True, slots=True)
class SelectionRule:
    rule: str = "point"
    z: float = 0.0
    """One-sided normal quantile of the lower bound (``lcb`` only)."""
    inflation: float = 1.0
    """Calibration factor κ applied to the posterior standard deviation."""

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        if self.z < 0 or self.inflation <= 0:
            raise ValueError("need z >= 0 and inflation > 0")


@dataclass(slots=True)
class CandidateSet:
    ps: tuple[float, ...]
    hero: FloatArray
    """(n, slots) hero strategies."""
    exploitability: FloatArray
    model_value: FloatArray
    sample_gain: FloatArray | None = None
    """(n, S) gain over CFR against each posterior sample, if computed."""
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Selection:
    index: int | None
    """Candidate played, or None for the CFR baseline."""
    p: float | None
    exploitability: float
    """Certified by the M3 evaluator (CFR's own value on fallback)."""
    score: float | None
    certificates: int
    """Number of M3 exploitability evaluations made."""


def stop_rule(expl: FloatArray, stop_above: float) -> int:
    """Number of grid points kept by M7A's rule: ascend, stop after two consecutive
    candidates above ``stop_above``."""
    over = 0
    for i, e in enumerate(expl):
        over = over + 1 if e > stop_above else 0
        if over >= 2:
            return i + 1
    return len(expl)


class Selector:
    def __init__(
        self,
        game: LeducGame,
        specs: dict[str, GameInfoSet],
        tree: CompiledTree,
        cfr: GameStrategy,
        cfr_exploitability: float,
    ) -> None:
        self.game, self.specs, self.tree = game, specs, tree
        self.evaluator = TreeEvaluator(tree)
        self.cfr = cfr
        self.cfr_vec = tree.to_vector(cfr)
        self.cfr_exploitability = cfr_exploitability

    def candidates(
        self,
        estimate: GameStrategy,
        stop_above: float,
        ps: Sequence[float] = P_GRID,
        iterations: int = 500,
        start: BatchedRNR | None = None,
    ) -> tuple[CandidateSet, BatchedRNR]:
        t0 = time.perf_counter()
        model = self.tree.to_vector(estimate)
        solver = BatchedRNR(self.tree, model, ps, start=start)
        solver.iterate(iterations)
        hero = solver.result().hero
        t1 = time.perf_counter()
        expl = self.evaluator.exploitability(hero)
        keep = stop_rule(expl, stop_above)
        values = self.evaluator.hero_values(hero[:keep], model[None, :])[:, 0]
        t2 = time.perf_counter()
        cands = CandidateSet(
            tuple(float(p) for p in ps[:keep]),
            hero[:keep],
            expl[:keep],
            values,
            timings={"solve": t1 - t0, "evaluate": t2 - t1},
        )
        return cands, solver

    def add_samples(self, cands: CandidateSet, samples: Sequence[GameStrategy]) -> None:
        """Gain over CFR of every candidate against every posterior sample."""
        t0 = time.perf_counter()
        opp = np.array([self.tree.to_vector(s) for s in samples])
        rows = np.vstack([cands.hero, self.cfr_vec[None, :]])
        v = self.evaluator.hero_values(rows, opp)
        cands.sample_gain = v[:-1] - v[-1][None, :]
        cands.timings["samples"] = time.perf_counter() - t0

    def scores(self, cands: CandidateSet, rule: SelectionRule) -> FloatArray:
        if rule.rule == "point":
            return cands.model_value.copy()
        if cands.sample_gain is None:
            raise ValueError("the lcb rule needs posterior samples")
        g = cands.sample_gain
        sd = g.std(axis=1, ddof=1)
        out: FloatArray = g.mean(axis=1) - rule.z * rule.inflation * sd
        return out

    def select(self, cands: CandidateSet, budget: float, rule: SelectionRule) -> Selection:
        """Best feasible candidate by ``rule``; every pick is certified by the M3 evaluator."""
        score = self.scores(cands, rule)
        feasible = [
            i for i in range(len(cands.ps)) if cands.exploitability[i] <= budget + TOLERANCE
        ]
        # M7A tie-break: higher score, then smaller p
        order = sorted(feasible, key=lambda i: (score[i], -cands.ps[i]), reverse=True)
        certificates = 0
        for i in order:
            if rule.rule == "lcb" and score[i] <= 0.0:
                break  # no candidate is confidently better than CFR
            certificates += 1
            policy = self.tree.to_table(cands.hero[i])
            exact = exploitability(self.game, policy, self.specs)
            if exact <= budget + TOLERANCE:
                return Selection(i, cands.ps[i], exact, float(score[i]), certificates)
        return Selection(None, None, self.cfr_exploitability, None, certificates)

    def policy(self, cands: CandidateSet, selection: Selection) -> GameStrategy:
        if selection.index is None:
            return self.cfr
        return self.tree.to_table(cands.hero[selection.index])


__all__ = [
    "P_GRID",
    "RULES",
    "CandidateSet",
    "Selection",
    "SelectionRule",
    "Selector",
    "stop_rule",
]
