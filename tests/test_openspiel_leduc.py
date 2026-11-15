"""Leduc cross-check against OpenSpiel's ``leduc_poker`` (optional; skipped if
OpenSpiel is not installed; see tests/test_openspiel_crosscheck.py).

Both games use the same card ids (rank = id // 2), the same action ids
(0 fold, 1 call/check, 2 raise/bet) and the same chance order, so the two
trees can be walked in lockstep with identical moves.
"""

from __future__ import annotations

from typing import Any

import pytest

from pokermind.evaluation.game_exact import exploitability_report, uniform_game_strategy
from pokermind.games.base import GameState
from pokermind.games.leduc import LeducGame
from pokermind.learning.game_cfr import GameCFRSolver
from pokermind.strategy import GameStrategy

pyspiel: Any = pytest.importorskip("pyspiel")
from open_spiel.python import policy as os_policy  # noqa: E402
from open_spiel.python.algorithms import cfr as os_cfr  # noqa: E402
from open_spiel.python.algorithms import exploitability as os_exploitability  # noqa: E402

GAME = LeducGame()


@pytest.fixture(scope="module")
def os_game() -> Any:
    return pyspiel.load_game("leduc_poker")


@pytest.fixture(scope="module")
def key_map(os_game: Any) -> dict[str, str]:
    """Walk both trees in lockstep, asserting identical structure; map keys."""
    mapping: dict[str, str] = {}
    nodes = [0]

    def walk(ours: GameState, theirs: Any) -> None:
        nodes[0] += 1
        assert ours.is_terminal == theirs.is_terminal()
        assert ours.is_chance_node == theirs.is_chance_node()
        if ours.is_terminal:
            assert ours.returns() == tuple(theirs.returns())
            return
        if ours.is_chance_node:
            assert list(ours.chance_outcomes()) == list(theirs.chance_outcomes())
            for outcome, _ in ours.chance_outcomes():
                walk(ours.apply(outcome), theirs.child(outcome))
            return
        player = ours.current_player
        assert player == theirs.current_player()
        assert ours.legal_actions() == tuple(theirs.legal_actions())
        key, os_key = ours.information_state_key(player), theirs.information_state_string(player)
        assert mapping.setdefault(key, os_key) == os_key
        for action in ours.legal_actions():
            walk(ours.apply(action), theirs.child(action))

    walk(GAME.initial_state(), os_game.new_initial_state())
    assert nodes[0] == 157 + 3_780 + 5_520
    return mapping


def to_os_policy(os_game: Any, strategy: GameStrategy, key_map: dict[str, str]) -> Any:
    tabular = os_policy.TabularPolicy(os_game)
    for key, dist in strategy.items():
        row = tabular.policy_for_key(key_map[key])
        row[:] = 0.0
        for action, prob in dist.items():
            row[action] = prob
    return tabular


def test_trees_identical_and_keys_bijective(os_game: Any, key_map: dict[str, str]) -> None:
    assert len(key_map) == 936
    assert len(set(key_map.values())) == 936
    assert set(key_map.values()) == set(os_policy.TabularPolicy(os_game).state_lookup)


def test_uniform_nash_conv_matches(os_game: Any, key_map: dict[str, str]) -> None:
    ours = exploitability_report(GAME, uniform_game_strategy(GAME)).nash_conv
    theirs = os_exploitability.nash_conv(os_game, os_policy.UniformRandomPolicy(os_game))
    assert ours == pytest.approx(theirs, abs=1e-12)


@pytest.mark.parametrize("iterations", [1, 10, 50])
def test_cfr_matches_openspiel_simultaneous_cfr(
    os_game: Any, key_map: dict[str, str], iterations: int
) -> None:
    ours = GameCFRSolver(GAME)
    ours.iterate(iterations)
    theirs = os_cfr._CFRSolver(
        os_game, regret_matching_plus=False, alternating_updates=False, linear_averaging=False
    )
    for _ in range(iterations):
        theirs.evaluate_and_update_policy()
    reference = theirs.average_policy()
    average = ours.average_strategy()
    for key, dist in average.items():
        row = reference.policy_for_key(key_map[key])
        for action, prob in dist.items():
            assert prob == pytest.approx(row[action], abs=1e-12), key
    os_nash_conv = os_exploitability.nash_conv(os_game, to_os_policy(os_game, average, key_map))
    assert exploitability_report(GAME, average).nash_conv == pytest.approx(os_nash_conv, abs=1e-12)
