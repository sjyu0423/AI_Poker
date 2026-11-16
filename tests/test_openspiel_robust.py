"""M7A cross-check (optional; skipped without pyspiel): the vectorized CFR+ (restriction
zero) reproduces OpenSpiel's CFRPlusSolver on Leduc."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pokermind.games.base import GameState
from pokermind.games.leduc import LeducGame
from pokermind.robust.solver import Copy, RestrictedSolver
from pokermind.robust.tree import CompiledTree

pyspiel: Any = pytest.importorskip("pyspiel")
from open_spiel.python.algorithms import cfr as os_cfr  # noqa: E402

GAME = LeducGame()


@pytest.mark.parametrize("iterations", [1, 5, 20])
def test_cfr_plus_matches_openspiel(iterations: int) -> None:
    os_game = pyspiel.load_game("leduc_poker")
    key_map: dict[str, str] = {}

    def walk(ours: GameState, theirs: Any) -> None:
        if ours.is_terminal:
            return
        if ours.is_chance_node:
            for outcome, _ in ours.chance_outcomes():
                walk(ours.apply(outcome), theirs.child(outcome))
            return
        player = ours.current_player
        key_map[ours.information_state_key(player)] = theirs.information_state_string(player)
        for action in ours.legal_actions():
            walk(ours.apply(action), theirs.child(action))

    walk(GAME.initial_state(), os_game.new_initial_state())
    tree = CompiledTree(GAME)
    solver = RestrictedSolver(
        tree, 0, [Copy(1.0, np.zeros(tree.num_isets))], tree.normalize(np.zeros(tree.num_slots))
    )
    solver.iterate(iterations)
    ours = tree.to_table(solver.result().average)
    theirs = os_cfr.CFRPlusSolver(os_game)
    for _ in range(iterations):
        theirs.evaluate_and_update_policy()
    reference = theirs.average_policy()
    for key, dist in ours.items():
        row = reference.policy_for_key(key_map[key])
        for action, prob in dist.items():
            assert prob == pytest.approx(row[action], abs=1e-12), key
