"""Restricted Nash Response and Data-Biased Response for hero (M7A).

Hero plays both seats. Each seat's strategy is computed by an independent
:class:`~pokermind.robust.solver.RestrictedSolver` in which hero is the
responder. Hero's seat-0 and seat-1 information sets never meet in the same
game, so the two solves are independent. The results are joined into one
seat-free table.

* :func:`rnr_strategy`: copies ``(p, c ≡ 1)`` and ``(1 − p, c ≡ 0)``. The
  model is the full (smoothed) reconstruction from M6.
* :func:`dbr_strategy`: one copy with ``c(I) = p_max · f(n_I)``. The model is
  the *frequency* estimate ``n_I(a) / n_I`` from the reconstruction's
  expected counts; ``f(0) = 0``, so unobserved information sets are
  unrestricted (the opponent is free to best-respond there).

Neither guarantees a requested exploitability. Callers measure every output
with the exact evaluator (:mod:`pokermind.robust.frontier`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from pokermind.adaptive.reconstruction import Theta, canonical_infoset
from pokermind.robust.solver import Copy, RestrictedSolver
from pokermind.robust.tree import CompiledTree, FloatArray
from pokermind.strategy import GameStrategy

DEFAULT_ITERATIONS: Final = 1000
Policy = dict[str, dict[int, float]]


def _solve(
    tree: CompiledTree, copies: list[Copy], model: FloatArray, iterations: int
) -> FloatArray:
    out = np.zeros(tree.num_slots)
    for responder in (0, 1):
        solver = RestrictedSolver(tree, responder, copies, model)
        solver.iterate(iterations)
        avg = solver.result().average
        mask = tree.slot_seat == responder
        out[mask] = avg[mask]
    return out


def rnr_strategy(
    tree: CompiledTree, model: GameStrategy, p: float, iterations: int = DEFAULT_ITERATIONS
) -> Policy:
    """Hero's Restricted Nash Response to ``model`` with restriction probability ``p``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    ones, zeros = np.ones(tree.num_isets), np.zeros(tree.num_isets)
    copies = [Copy(w, c) for w, c in ((p, ones), (1.0 - p, zeros)) if w > 0]
    return tree.to_table(_solve(tree, copies, tree.to_vector(model), iterations))


@dataclass(frozen=True, slots=True)
class DBRModel:
    """Frequency model and observation counts per information set (seat-free keys)."""

    frequencies: Policy
    counts: dict[str, float]


def dbr_model(expected_counts: Theta, actions: Mapping[str, tuple[int, ...]]) -> DBRModel:
    """Frequencies ``n(a) / n`` and totals ``n`` from canonical expected counts.

    ``actions`` maps every full information-set key to its legal actions;
    suits share their canonical counts. Unobserved sets get a uniform
    placeholder, which DBR never uses because their confidence is 0.
    """
    freq: Policy = {}
    counts: dict[str, float] = {}
    for key, legal in actions.items():
        vec = expected_counts[canonical_infoset(key)]
        n = float(vec.sum())
        counts[key] = n
        if n > 0:
            freq[key] = {a: float(v / n) for a, v in zip(legal, vec, strict=True)}
        else:
            freq[key] = {a: 1.0 / len(legal) for a in legal}
    return DBRModel(freq, counts)


def confidence(n: float, p_max: float, threshold: float) -> float:
    """``p_max · min(1, n / threshold)``: the paper's "0-threshold linear" function."""
    if not 0.0 <= p_max <= 1.0 or threshold <= 0:
        raise ValueError("need 0 <= p_max <= 1 and threshold > 0")
    return p_max * min(1.0, n / threshold)


def dbr_strategy(
    tree: CompiledTree,
    model: DBRModel,
    p_max: float,
    threshold: float,
    iterations: int = DEFAULT_ITERATIONS,
) -> Policy:
    """Hero's Data-Biased Response: per-information-set confidence from counts."""
    c = tree.iset_values({k: confidence(n, p_max, threshold) for k, n in model.counts.items()})
    return tree.to_table(
        _solve(tree, [Copy(1.0, c)], tree.to_vector(model.frequencies), iterations)
    )
