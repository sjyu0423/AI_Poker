"""A two-player game tree compiled into flat NumPy arrays (M7A).

The M3 solvers recurse through ``GameState`` objects (about 18 ms per Leduc
CFR iteration). RNR and DBR need many solves per opponent snapshot, so
:class:`CompiledTree` walks the game once and stores:

* nodes grouped by depth, with each node's parent, kind (chance, decision,
  terminal), acting seat and information set;
* every edge's chance probability or *slot*: the index of an
  (information set, action) pair, so a whole strategy is one vector;
* terminal payoffs for seat 0 (two-player zero-sum).

Information sets keep the game's own keys, so strategies convert to and from
the usual ``{key: {action: probability}}`` tables. Nothing about the
opponent, its model or any restriction is stored here; those are solver
inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from pokermind.games.base import Game, GameState

CHANCE_KIND: Final = 0
DECISION_KIND: Final = 1
TERMINAL_KIND: Final = 2
IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Level:
    """Edges whose child is at one depth."""

    child: IntArray
    parent: IntArray
    slot: IntArray
    """Slot of the action taken, or -1 for a chance edge."""
    chance_prob: FloatArray
    """Chance probability (1 for decision edges)."""
    parent_seat: IntArray
    """Seat acting at the parent (-1 for chance)."""


class CompiledTree:
    def __init__(self, game: Game) -> None:
        kind: list[int] = []
        seat: list[int] = []
        iset_of: list[int] = []
        payoff: list[float] = []
        depth: list[int] = []
        edges: list[tuple[int, int, int, float, int]] = []  # child, parent, slot, prob, seat
        self.keys: list[str] = []
        self.key_index: dict[str, int] = {}
        self.actions: list[tuple[int, ...]] = []
        self.iset_seat: list[int] = []
        slot_iset: list[int] = []
        slot_action: list[int] = []
        self.slot_of: dict[tuple[int, int], int] = {}

        def visit(
            state: GameState, d: int, parent: int, slot: int, prob: float, pseat: int
        ) -> None:
            node = len(kind)
            depth.append(d)
            if parent >= 0:
                edges.append((node, parent, slot, prob, pseat))
            if state.is_terminal:
                kind.append(TERMINAL_KIND)
                seat.append(-1)
                iset_of.append(-1)
                payoff.append(state.returns()[0])
                return
            if state.is_chance_node:
                kind.append(CHANCE_KIND)
                seat.append(-1)
                iset_of.append(-1)
                payoff.append(0.0)
                for outcome, p in state.chance_outcomes():
                    visit(state.apply(outcome), d + 1, node, -1, p, -1)
                return
            player = state.current_player
            key = state.information_state_key(player)
            legal = state.legal_actions()
            if key not in self.key_index:
                self.key_index[key] = len(self.keys)
                self.keys.append(key)
                self.actions.append(legal)
                self.iset_seat.append(player)
                for a in legal:
                    self.slot_of[(self.key_index[key], a)] = len(slot_iset)
                    slot_iset.append(self.key_index[key])
                    slot_action.append(a)
            i = self.key_index[key]
            if self.actions[i] != legal or self.iset_seat[i] != player:
                raise ValueError(f"inconsistent information set {key!r}")
            kind.append(DECISION_KIND)
            seat.append(player)
            iset_of.append(i)
            payoff.append(0.0)
            for a in legal:
                visit(state.apply(a), d + 1, node, self.slot_of[(i, a)], 1.0, player)

        visit(game.initial_state(), 0, -1, -1, 1.0, -1)
        self.num_nodes = len(kind)
        self.kind = np.array(kind, dtype=np.int64)
        self.seat = np.array(seat, dtype=np.int64)
        self.iset = np.array(iset_of, dtype=np.int64)
        self.payoff0 = np.array(payoff, dtype=np.float64)
        self.slot_iset = np.array(slot_iset, dtype=np.int64)
        self.slot_action = np.array(slot_action, dtype=np.int64)
        self.slot_seat = np.array([self.iset_seat[i] for i in slot_iset], dtype=np.int64)
        self.num_slots = len(slot_iset)
        self.num_isets = len(self.keys)
        e = np.array(edges, dtype=np.float64)
        child = e[:, 0].astype(np.int64)
        depth_arr = np.array(depth, dtype=np.int64)
        self.levels: list[Level] = []
        for d in range(1, int(depth_arr.max()) + 1):
            sel = depth_arr[child] == d
            self.levels.append(
                Level(
                    child=child[sel],
                    parent=e[sel, 1].astype(np.int64),
                    slot=e[sel, 2].astype(np.int64),
                    chance_prob=e[sel, 3],
                    parent_seat=e[sel, 4].astype(np.int64),
                )
            )
        # every decision edge, for regret and average updates
        dec = e[:, 2] >= 0
        self.edge_child = child[dec]
        self.edge_parent = e[dec, 1].astype(np.int64)
        self.edge_slot = e[dec, 2].astype(np.int64)

    # -- conversions ---------------------------------------------------------------------

    def to_vector(
        self, strategy: Mapping[str, Mapping[int, float]], seats: tuple[int, ...] = (0, 1)
    ) -> FloatArray:
        """Slot vector from a table; information sets of other seats are left at 0."""
        vec = np.zeros(self.num_slots)
        for s in range(self.num_slots):
            i = int(self.slot_iset[s])
            if self.iset_seat[i] in seats:
                vec[s] = strategy[self.keys[i]][int(self.slot_action[s])]
        return vec

    def to_table(
        self, vec: FloatArray, seats: tuple[int, ...] = (0, 1)
    ) -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = {}
        for s in range(self.num_slots):
            i = int(self.slot_iset[s])
            if self.iset_seat[i] in seats:
                out.setdefault(self.keys[i], {})[int(self.slot_action[s])] = float(vec[s])
        return out

    def iset_values(self, per_key: Mapping[str, float], default: float = 0.0) -> FloatArray:
        """Per-information-set vector from a ``{key: value}`` mapping."""
        return np.array([per_key.get(k, default) for k in self.keys], dtype=np.float64)

    def normalize(self, vec: FloatArray) -> FloatArray:
        """Normalize per information set; uniform where the total is 0."""
        totals = np.bincount(self.slot_iset, weights=vec, minlength=self.num_isets)
        counts = np.bincount(self.slot_iset, minlength=self.num_isets)
        t = totals[self.slot_iset]
        out: FloatArray = np.where(
            t > 0, vec / np.where(t > 0, t, 1.0), 1.0 / counts[self.slot_iset]
        )
        return out
