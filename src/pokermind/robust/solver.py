"""Vectorized CFR / CFR+ for restricted opponents: RNR and DBR (M7A).

One *responder* seat ``r`` (hero) plays against the other seat (the
opponent). The opponent's behaviour is described by one or more weighted
**copies** of the game. In copy ``k`` (weight ``w_k``) the opponent plays, at
each of its information sets ``I``,

    μ_k(I) = c_k(I) · model(I) + (1 − c_k(I)) · y(I)

where ``y`` is the opponent's free strategy, which the solver learns, and
``c_k(I) ∈ [0, 1]`` is a restriction weight. Hero's strategy ``x`` is shared
by all copies, because hero can't tell them apart. Special cases:

* **Plain CFR:** one copy, ``c ≡ 0``.
* **Restricted Nash Response** (Johanson et al. 2008): copies
  ``(p, c ≡ 1)`` and ``(1 − p, c ≡ 0)``. Before the hand a hidden coin forces
  the opponent to play the model with probability ``p``. The opponent knows
  the coin; hero doesn't. This is an ordinary perfect-recall extensive-form
  game.
* **Data-Biased Response** (Johanson & Bowling 2009): one copy with an
  information-set-dependent confidence ``c(I)``. The opponent's behaviour at
  ``I`` is forced to the model with probability ``c(I)``, independently at
  each information set. As a game this is *not* bilinear in the opponent's
  sequence-form strategy, so the usual CFR convergence guarantee doesn't
  formally apply. The paper computes it the same way, and M7A verifies every
  output exactly.

Updates for hero: counterfactual regret ``Σ_k w_k π^k_{-r}(h) (v^k(h·a) − v^k(h))``.
For the opponent: ``Σ_k w_k (1 − c_k(I)) π^k_{-o}(h) (v^k_o(h·a) − Σ_b y(b) v^k_o(h·b))``,
the regret of its *free* choice.

Modes: ``plus=False, alternating=False, linear=False`` is vanilla CFR with
simultaneous updates (M1/M3). The defaults ``plus=True, alternating=True,
linear=True`` are CFR+ as in OpenSpiel's ``CFRPlusSolver``: seat 0 is updated
first, regrets are floored at 0, and the average is weighted by the iteration
number. Tests match both against the existing solvers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pokermind.robust.tree import BoolArray, CompiledTree, FloatArray


@dataclass(frozen=True, slots=True)
class Copy:
    weight: float
    restriction: FloatArray
    """``c(I)`` per information set (only opponent information sets matter)."""


@dataclass(slots=True)
class SolveResult:
    average: FloatArray
    """Average strategy (slot vector, normalized per information set), both seats."""
    current: FloatArray
    iterations: int


class RestrictedSolver:
    def __init__(
        self,
        tree: CompiledTree,
        responder: int,
        copies: Sequence[Copy],
        model: FloatArray,
        *,
        plus: bool = True,
        alternating: bool = True,
        linear: bool = True,
    ) -> None:
        if responder not in (0, 1):
            raise ValueError("responder must be seat 0 or 1")
        if not copies or abs(sum(c.weight for c in copies) - 1.0) > 1e-12:
            raise ValueError("copy weights must be positive and sum to 1")
        for c in copies:
            if c.weight < 0 or (c.restriction < 0).any() or (c.restriction > 1).any():
                raise ValueError("weights and restrictions must lie in [0, 1]")
        self.t = tree
        self.r = responder
        self.copies = list(copies)
        self.model = model
        self.plus, self.alternating, self.linear = plus, alternating, linear
        self.hero_slot: BoolArray = tree.slot_seat == responder
        self.regret = np.zeros(tree.num_slots)
        self.avg = np.zeros(tree.num_slots)
        self.current = tree.normalize(np.zeros(tree.num_slots))
        self.iterations = 0
        # per-copy restriction on slots of opponent information sets (0 on hero slots)
        self._c_slot = [
            np.where(self.hero_slot, 0.0, c.restriction[tree.slot_iset]) for c in copies
        ]

    # -- one pass ------------------------------------------------------------------------

    def _effective(self, k: int) -> FloatArray:
        c = self._c_slot[k]
        out: FloatArray = c * self.model + (1.0 - c) * self.current
        return out

    def _pass(
        self, k: int, sigma: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Reach (seat 0, seat 1, chance) and seat-0 values at every node."""
        t = self.t
        n = t.num_nodes
        reach = np.ones((3, n))
        for lv in t.levels:
            p = lv.parent
            factor = np.where(lv.slot >= 0, sigma[np.maximum(lv.slot, 0)], 1.0)
            reach[0, lv.child] = reach[0, p] * np.where(lv.parent_seat == 0, factor, 1.0)
            reach[1, lv.child] = reach[1, p] * np.where(lv.parent_seat == 1, factor, 1.0)
            reach[2, lv.child] = reach[2, p] * lv.chance_prob
        value = t.payoff0.copy()
        for lv in reversed(t.levels):
            w = np.where(lv.slot >= 0, sigma[np.maximum(lv.slot, 0)], lv.chance_prob)
            np.add.at(value, lv.parent, w * value[lv.child])
        return reach[0], reach[1], reach[2], value

    def _update_seat(self, seat: int, weight_t: float) -> None:
        t = self.t
        regret = np.zeros(t.num_slots)
        avg = np.zeros(t.num_slots)
        is_hero = seat == self.r
        sign = 1.0 if seat == 0 else -1.0
        first = True
        for k, copy in enumerate(self.copies):
            if copy.weight == 0.0:
                continue
            sigma = self._effective(k)
            r0, r1, rc, v0 = self._pass(k, sigma)
            own = r0 if seat == 0 else r1
            other = r1 if seat == 0 else r0
            cf = other * rc
            par, ch, slot = t.edge_parent, t.edge_child, t.edge_slot
            mine = t.slot_seat[slot] == seat
            par, ch, slot = par[mine], ch[mine], slot[mine]
            v_child = sign * v0[ch]
            if is_hero:
                base = sign * v0[par]
                scale = np.ones(len(slot))
            else:
                y = self.current
                base_node = np.zeros(t.num_nodes)
                np.add.at(base_node, par, y[slot] * v_child)
                base = base_node[par]
                scale = 1.0 - self._c_slot[k][slot]
            regret += np.bincount(
                slot,
                weights=copy.weight * scale * cf[par] * (v_child - base),
                minlength=t.num_slots,
            )
            if first:  # own reach doesn't depend on the copy (opponent average: c = 0 copy only)
                avg += np.bincount(
                    slot, weights=own[par] * self.current[slot], minlength=t.num_slots
                )
                first = False
        mask = t.slot_seat == seat
        self.regret[mask] += regret[mask]
        if self.plus:
            self.regret[mask] = np.maximum(self.regret[mask], 0.0)
        self.avg[mask] += weight_t * avg[mask]

    def _refresh(self, seat: int | None) -> None:
        t = self.t
        positive = np.maximum(self.regret, 0.0)
        new = t.normalize(positive)
        if seat is None:
            self.current = new
        else:
            self.current = np.where(t.slot_seat == seat, new, self.current)

    def iterate(self, iterations: int) -> None:
        for _ in range(iterations):
            self.iterations += 1
            w = float(self.iterations) if self.linear else 1.0
            if self.alternating:
                for seat in (0, 1):
                    self._update_seat(seat, w)
                    self._refresh(seat)
            else:
                self._update_both(w)
                self._refresh(None)

    def _update_both(self, weight_t: float) -> None:
        saved = self.current.copy()
        for seat in (0, 1):
            self.current = saved
            self._update_seat(seat, weight_t)
        self.current = saved

    def result(self) -> SolveResult:
        return SolveResult(self.t.normalize(self.avg), self.current.copy(), self.iterations)
