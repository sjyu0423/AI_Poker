"""M6 cross-check against OpenSpiel (optional; skipped without pyspiel): hero best-response
values and the exact value of a realization mixture against a simulated opponent."""

from __future__ import annotations

from typing import Any

import pytest

from pokermind.adaptive.response import (
    hero_best_response,
    own_reach,
    realization_mixture,
)
from pokermind.evaluation.game_exact import best_response, expected_value, information_sets
from pokermind.games.base import GameState
from pokermind.games.leduc import LeducGame
from pokermind.opponents.population import ParametricParams, parametric_policy
from pokermind.strategy import GameStrategy

pyspiel: Any = pytest.importorskip("pyspiel")
from open_spiel.python import policy as os_policy  # noqa: E402
from open_spiel.python.algorithms import exploitability as os_exploitability  # noqa: E402
from open_spiel.python.algorithms.expected_game_score import policy_value  # noqa: E402

GAME = LeducGame()
SPECS = information_sets(GAME)


@pytest.fixture(scope="module")
def os_game() -> Any:
    return pyspiel.load_game("leduc_poker")


@pytest.fixture(scope="module")
def key_map(os_game: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}

    def walk(ours: GameState, theirs: Any) -> None:
        if ours.is_terminal:
            return
        if ours.is_chance_node:
            for outcome, _ in ours.chance_outcomes():
                walk(ours.apply(outcome), theirs.child(outcome))
            return
        player = ours.current_player
        mapping[ours.information_state_key(player)] = theirs.information_state_string(player)
        for action in ours.legal_actions():
            walk(ours.apply(action), theirs.child(action))

    walk(GAME.initial_state(), os_game.new_initial_state())
    return mapping


def to_os(os_game: Any, strategy: GameStrategy, key_map: dict[str, str]) -> Any:
    tabular = os_policy.TabularPolicy(os_game)
    for key, dist in strategy.items():
        row = tabular.policy_for_key(key_map[key])
        row[:] = 0.0
        for action, prob in dist.items():
            row[action] = prob
    return tabular


@pytest.fixture(scope="module")
def opponent() -> dict[str, dict[int, float]]:
    return parametric_policy(ParametricParams(0.1, 0.8, 0.3, 0.03), SPECS)


@pytest.mark.parametrize("player", [0, 1])
def test_best_response_value_matches_openspiel(
    os_game: Any, key_map: dict[str, str], opponent: dict[str, dict[int, float]], player: int
) -> None:
    ours = best_response(GAME, opponent, player, infosets=SPECS).value
    theirs = os_exploitability.best_response(os_game, to_os(os_game, opponent, key_map), player)
    assert ours == pytest.approx(theirs["best_response_value"], abs=1e-10)


def test_mixture_value_matches_openspiel(
    os_game: Any, key_map: dict[str, str], opponent: dict[str, dict[int, float]]
) -> None:
    uniform = {k: {a: 1.0 / len(s.actions) for a in s.actions} for k, s in SPECS.items()}
    br = hero_best_response(GAME, opponent, SPECS)
    mix = realization_mixture(uniform, br, 0.3, own_reach(GAME, uniform), own_reach(GAME, br))
    ours = expected_value(GAME, (mix, opponent), infosets=SPECS)
    # OpenSpiel: hero (the mixture) in seat 0, the opponent in seat 1.
    combined = {**opponent, **{k: v for k, v in mix.items() if SPECS[k].player == 0}}
    theirs = policy_value(os_game.new_initial_state(), [to_os(os_game, combined, key_map)] * 2)
    assert ours == pytest.approx(theirs[0], abs=1e-10)
