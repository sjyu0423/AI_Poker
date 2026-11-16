"""Batched RNR solves and batched exact evaluation on a compiled tree (M7B).

M7A solved each Restricted Nash Response separately: two seats and two
copies, one ``p`` at a time. Each NumPy call touched about 10,000 numbers, so
Python and NumPy call overhead dominated. This module stacks **every**
``(p, hero seat, copy)`` problem into the rows of 2-D arrays and advances all
of them with the same calls.

**Equivalence.** :class:`BatchedRNR` performs the same floating-point
operations as :class:`~pokermind.robust.solver.RestrictedSolver` in CFR+
mode (alternating, regrets floored, linear averaging), in the same order:

* every sum uses ``np.bincount``, which accumulates in input order, as
  ``np.add.at`` does;
* a zero-weight copy adds exactly ``+0.0``.

Tests check that hero's average strategies are **bit-identical** to M7A's
:func:`~pokermind.robust.responses.rnr_strategy`.

The batch also keeps the free opponent's average strategy. It is
accumulated on the unrestricted copy, as it must be: the M7A solver did so on
its first copy, which it never used. That average lets the solver's own gap be
measured exactly in the modified game
(:func:`pokermind.robust.modified_game.rnr_nash_conv`).

:class:`TreeEvaluator` computes exact payoffs and exact best-response values
for many hero strategies at once by dynamic programming over the compiled
tree. Leduc's information sets each lie at a single depth (checked when the
evaluator is built), so a best response can be chosen level by level. Tests
match it to the M3 recursive evaluator to 1e-12. Selection uses it to *search*;
the strategy finally played is certified again by the M3 evaluator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pokermind.robust.tree import CompiledTree, FloatArray, IntArray
from pokermind.strategy import GameStrategy

Policy = dict[str, dict[int, float]]


def _rows_bincount(index: IntArray, weights: FloatArray, size: int) -> FloatArray:
    """Row-wise ``bincount``: ``weights`` has shape (rows, len(index))."""
    rows = weights.shape[0]
    flat = (np.arange(rows)[:, None] * size + index[None, :]).ravel()
    out: FloatArray = np.bincount(flat, weights=weights.ravel(), minlength=rows * size)
    return out.reshape(rows, size)


def _normalize_rows(tree: CompiledTree, vec: FloatArray) -> FloatArray:
    """:meth:`CompiledTree.normalize` applied to every row."""
    totals = _rows_bincount(tree.slot_iset, vec, tree.num_isets)
    counts = np.bincount(tree.slot_iset, minlength=tree.num_isets)
    t = totals[:, tree.slot_iset]
    out: FloatArray = np.where(t > 0, vec / np.where(t > 0, t, 1.0), 1.0 / counts[tree.slot_iset])
    return out


@dataclass(frozen=True, slots=True)
class BatchResult:
    ps: tuple[float, ...]
    hero: FloatArray
    """(len(ps), slots): hero's average strategy, both seats joined."""
    opponent: FloatArray
    """(len(ps), slots): the free opponent's average strategy, both seats joined."""
    iterations: int


class BatchedRNR:
    """CFR+ on the RNR game for many restriction probabilities ``p`` at once.

    Problem ``b = 2·j + r`` is ``p = ps[j]`` with hero in seat ``r``. Its two
    copies are the restricted copy (weight ``p``: the opponent plays
    ``model``) and the free copy (weight ``1 − p``).
    """

    def __init__(
        self,
        tree: CompiledTree,
        model: FloatArray,
        ps: Sequence[float],
        *,
        start: BatchedRNR | None = None,
    ) -> None:
        if not ps or any(not 0.0 <= p <= 1.0 for p in ps):
            raise ValueError("every p must be in [0, 1]")
        self.t = tree
        self.ps = tuple(float(p) for p in ps)
        nb = 2 * len(self.ps)
        self.responder = np.tile(np.array([0, 1]), len(self.ps))
        weights = np.array([[p, 1.0 - p] for p in self.ps for _ in (0, 1)])
        self.weight = weights  # (nb, 2)
        self.model = model
        self.hero_mask = tree.slot_seat[None, :] == self.responder[:, None]  # (nb, slots)
        self.regret = np.zeros((nb, tree.num_slots))
        self.avg = np.zeros((nb, tree.num_slots))
        self.current = np.repeat(tree.normalize(np.zeros(tree.num_slots))[None, :], nb, axis=0)
        self.iterations = 0
        if start is not None:
            self._warm(start)
        # static per-seat edge selections
        self._mine = {}
        for seat in (0, 1):
            m = tree.slot_seat[tree.edge_slot] == seat
            self._mine[seat] = (tree.edge_parent[m], tree.edge_child[m], tree.edge_slot[m])
        self._levels = [
            (lv, lv.slot >= 0, np.maximum(lv.slot, 0), lv.parent_seat == 0, lv.parent_seat == 1)
            for lv in tree.levels
        ]
        self._chance_reach = self._chance()

    def _warm(self, start: BatchedRNR) -> None:
        """Start from the nearest solved ``p`` of ``start``: its regrets and current
        strategy, with a fresh average (so the average reflects this solve only)."""
        for j, p in enumerate(self.ps):
            k = int(np.argmin([abs(p - q) for q in start.ps]))
            for r in (0, 1):
                self.regret[2 * j + r] = start.regret[2 * k + r]
                self.current[2 * j + r] = start.current[2 * k + r]

    def _chance(self) -> FloatArray:
        rc = np.ones(self.t.num_nodes)
        for lv, *_ in self._levels:
            rc[lv.child] = rc[lv.parent] * lv.chance_prob
        return rc

    # -- one pass over all rows ---------------------------------------------------------------

    def _sigma(self) -> FloatArray:
        """(nb, 2, slots): copy 0 restricted (model at opponent slots), copy 1 free."""
        restricted = np.where(self.hero_mask, self.current, 1.0 * self.model + 0.0 * self.current)
        return np.stack([restricted, self.current], axis=1)

    def _pass(self, sigma: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Reach of seat 0 and seat 1 and seat-0 values; rows are (problem, copy)."""
        t = self.t
        rows = sigma.shape[0]
        r0 = np.ones((rows, t.num_nodes))
        r1 = np.ones((rows, t.num_nodes))
        for lv, is_dec, slot0, s0, s1 in self._levels:
            factor = np.where(is_dec, sigma[:, slot0], 1.0)
            r0[:, lv.child] = r0[:, lv.parent] * np.where(s0, factor, 1.0)
            r1[:, lv.child] = r1[:, lv.parent] * np.where(s1, factor, 1.0)
        value = np.repeat(t.payoff0[None, :], rows, axis=0)
        for lv, is_dec, slot0, _, _ in reversed(self._levels):
            w = np.where(is_dec, sigma[:, slot0], lv.chance_prob)
            value += _rows_bincount(lv.parent, w * value[:, lv.child], t.num_nodes)
        return r0, r1, value

    def _update_seat(self, seat: int, weight_t: float) -> None:
        t = self.t
        nb = self.regret.shape[0]
        sigma = self._sigma().reshape(nb * 2, t.num_slots)
        r0, r1, v0 = self._pass(sigma)
        own = (r0 if seat == 0 else r1).reshape(nb, 2, t.num_nodes)
        other = (r1 if seat == 0 else r0).reshape(nb, 2, t.num_nodes)
        v = v0.reshape(nb, 2, t.num_nodes)
        sign = 1.0 if seat == 0 else -1.0
        par, ch, slot = self._mine[seat]
        hero_rows = self.responder == seat
        regret = np.zeros((nb, t.num_slots))
        for k in (0, 1):
            cf = other[:, k, :] * self._chance_reach[None, :]
            v_child = sign * v[:, k, ch]
            hero_base = sign * v[:, k, par]
            y = self.current
            opp_base_node = _rows_bincount(par, y[:, slot] * v_child, t.num_nodes)
            base = np.where(hero_rows[:, None], hero_base, opp_base_node[:, par])
            # opponent rows: restricted copy has scale 1 - 1 = 0, free copy 1 - 0 = 1
            scale = np.where(hero_rows[:, None], 1.0, 0.0 if k == 0 else 1.0)
            contrib = self.weight[:, k][:, None] * scale * cf[:, par] * (v_child - base)
            regret += _rows_bincount(slot, contrib, t.num_slots)
        # hero: own reach is the same in both copies (M7A used its first active copy);
        # opponent: the free copy, where the opponent plays its own strategy
        first = np.where(self.weight[:, 0] > 0, 0, 1)
        own_k = np.where(hero_rows[:, None], own[np.arange(nb), first, :], own[:, 1, :])
        avg = _rows_bincount(slot, own_k[:, par] * self.current[:, slot], t.num_slots)
        mask = t.slot_seat == seat
        self.regret[:, mask] += regret[:, mask]
        self.regret[:, mask] = np.maximum(self.regret[:, mask], 0.0)
        self.avg[:, mask] += weight_t * avg[:, mask]

    def _refresh(self, seat: int) -> None:
        new = _normalize_rows(self.t, np.maximum(self.regret, 0.0))
        self.current = np.where(self.t.slot_seat[None, :] == seat, new, self.current)

    def iterate(self, iterations: int) -> None:
        for _ in range(iterations):
            self.iterations += 1
            w = float(self.iterations)
            for seat in (0, 1):
                self._update_seat(seat, w)
                self._refresh(seat)

    def result(self) -> BatchResult:
        avg = _normalize_rows(self.t, self.avg)
        n = len(self.ps)
        hero = np.zeros((n, self.t.num_slots))
        opponent = np.zeros((n, self.t.num_slots))
        for j in range(n):
            for r in (0, 1):
                mask = self.t.slot_seat == r
                hero[j, mask] = avg[2 * j + r, mask]
                opponent[j, ~mask] = avg[2 * j + r, ~mask]
        return BatchResult(self.ps, hero, opponent, self.iterations)


def rnr_strategies(
    tree: CompiledTree, model: GameStrategy, ps: Sequence[float], iterations: int
) -> list[Policy]:
    """Hero's RNR strategy for every ``p`` (identical to M7A's ``rnr_strategy``)."""
    solver = BatchedRNR(tree, tree.to_vector(model), ps)
    solver.iterate(iterations)
    res = solver.result()
    return [tree.to_table(res.hero[j]) for j in range(len(ps))]


class TreeEvaluator:
    """Exact payoffs and best-response values for many hero strategies at once.

    Hero is seat-free: as seat 0 it plays its seat-0 slots, as seat 1 its
    seat-1 slots. Payoffs are hero's, averaged over the two seats, as in
    :func:`pokermind.adaptive.response.hero_value`.
    """

    def __init__(self, tree: CompiledTree) -> None:
        self.t = tree
        depth_of = np.full(tree.num_nodes, 0, dtype=np.int64)
        for d, lv in enumerate(tree.levels, start=1):
            depth_of[lv.child] = d
        dec = tree.iset >= 0
        iset_depth = np.full(tree.num_isets, -1, dtype=np.int64)
        for node in np.flatnonzero(dec):
            i, d = tree.iset[node], depth_of[node]
            if iset_depth[i] not in (-1, d):
                raise ValueError("an information set spans several depths")
            iset_depth[i] = d
        self.seat_mask = {s: tree.slot_seat == s for s in (0, 1)}

    def _profile(self, seat0: FloatArray, seat1: FloatArray) -> FloatArray:
        return np.where(self.seat_mask[0][None, :], seat0, seat1)

    def _values(self, sigma: FloatArray) -> FloatArray:
        """Seat-0 expected value at the root for each row of ``sigma``."""
        t = self.t
        value = np.repeat(t.payoff0[None, :], sigma.shape[0], axis=0)
        for lv in reversed(t.levels):
            w = np.where(lv.slot >= 0, sigma[:, np.maximum(lv.slot, 0)], lv.chance_prob)
            value += _rows_bincount(lv.parent, w * value[:, lv.child], t.num_nodes)
        out: FloatArray = value[:, 0]
        return out

    def hero_values(self, hero: FloatArray, opponents: FloatArray) -> FloatArray:
        """(H, O): hero row ``h``'s payoff against opponent row ``o``."""
        h, o = hero.shape[0], opponents.shape[0]
        hh = np.repeat(hero, o, axis=0)
        oo = np.tile(opponents, (h, 1))
        as0 = self._values(self._profile(hh, oo))
        as1 = -self._values(self._profile(oo, hh))
        out: FloatArray = ((as0 + as1) / 2).reshape(h, o)
        return out

    def best_response_values(self, strategy: FloatArray, responder: int) -> FloatArray:
        """Value to ``responder`` of a best response against each row of ``strategy``."""
        t = self.t
        rows = strategy.shape[0]
        sign = 1.0 if responder == 0 else -1.0
        reach = np.ones((rows, t.num_nodes))  # reach of everyone except the responder
        for lv in t.levels:
            other_dec = (lv.slot >= 0) & (lv.parent_seat != responder)
            factor = np.where(other_dec, strategy[:, np.maximum(lv.slot, 0)], 1.0)
            reach[:, lv.child] = reach[:, lv.parent] * factor * lv.chance_prob
        value = np.repeat(sign * t.payoff0[None, :], rows, axis=0)
        for lv in reversed(t.levels):
            mine = lv.parent_seat == responder
            w = np.where(lv.slot >= 0, strategy[:, np.maximum(lv.slot, 0)], lv.chance_prob)
            w = np.where(mine, 0.0, w)
            value += _rows_bincount(lv.parent, w * value[:, lv.child], t.num_nodes)
            if not mine.any():
                continue
            par, ch, slot = lv.parent[mine], lv.child[mine], lv.slot[mine]
            q = _rows_bincount(slot, reach[:, par] * value[:, ch], t.num_slots)
            uniq = np.unique(slot)
            iset_u = t.slot_iset[uniq]
            best = np.full((rows, t.num_isets), -np.inf)
            np.maximum.at(best.T, iset_u, q[:, uniq].T)
            # the first maximal slot of each information set (the value is the same for ties)
            candidate = np.where(q[:, uniq] >= best[:, iset_u], uniq[None, :], t.num_slots)
            first = np.full((rows, t.num_isets), t.num_slots, dtype=np.int64)
            np.minimum.at(first.T, iset_u, candidate.T)
            follow = first[:, t.slot_iset[slot]] == slot[None, :]
            value += _rows_bincount(par, np.where(follow, value[:, ch], 0.0), t.num_nodes)
        out: FloatArray = value[:, 0]
        return out

    def iset_reach(self, sigma: FloatArray) -> FloatArray:
        """(rows, isets): probability of reaching each information set (all players and
        chance) under each row's complete profile."""
        t = self.t
        reach = np.ones((sigma.shape[0], t.num_nodes))
        for lv in t.levels:
            w = np.where(lv.slot >= 0, sigma[:, np.maximum(lv.slot, 0)], lv.chance_prob)
            reach[:, lv.child] = reach[:, lv.parent] * w
        dec = np.flatnonzero(t.iset >= 0)
        return _rows_bincount(t.iset[dec], reach[:, dec], t.num_isets)

    def hero_reach_weights(self, hero: FloatArray, opponent: FloatArray) -> FloatArray:
        """Reach of each of hero's information sets when ``hero`` plays ``opponent``,
        seats alternating (each seat weighted ½); 0 at the opponent's sets."""
        prof0 = self._profile(hero[None, :], opponent[None, :])
        prof1 = self._profile(opponent[None, :], hero[None, :])
        r0, r1 = self.iset_reach(prof0)[0], self.iset_reach(prof1)[0]
        seat = np.array(self.t.iset_seat)
        out: FloatArray = 0.5 * np.where(seat == 0, r0, r1)
        return out

    def exploitability(self, strategy: FloatArray) -> FloatArray:
        """NashConv / 2 of each row (a seat-free strategy played by both seats)."""
        br0 = self.best_response_values(strategy, 0)
        br1 = self.best_response_values(strategy, 1)
        out: FloatArray = (br0 + br1) / 2
        return out


__all__ = ["BatchResult", "BatchedRNR", "TreeEvaluator", "rnr_strategies"]
