"""Cross-check against DeepMind's OpenSpiel (optional; skipped if not installed).

OpenSpiel is not a project dependency. To run these tests:

    pip install open_spiel        # or: pip install -e ".[crosscheck]"
    pytest tests/test_openspiel_crosscheck.py

Mapping: OpenSpiel's ``kuhn_poker`` uses cards 0 < 1 < 2 (J, Q, K) and two
actions, 0 = pass and 1 = bet. M0's CHECK and FOLD are "pass"; BET and CALL
are "bet". An information state string is the card digit followed by the
history, e.g. ``"1pb"`` == M0's ``"Q|xb"``. Payoffs use the same convention
(ante 1, bet 1, net chips).
"""

from __future__ import annotations

from typing import Any

import pytest

from pokermind.evaluation import exploitability_report, information_sets, uniform_strategy
from pokermind.games.kuhn import Action, Card
from pokermind.learning import CFRSolver
from pokermind.strategy import Strategy

pyspiel: Any = pytest.importorskip("pyspiel")
from open_spiel.python import policy as os_policy  # noqa: E402
from open_spiel.python.algorithms import cfr as os_cfr  # noqa: E402
from open_spiel.python.algorithms import exploitability as os_exploitability  # noqa: E402

PASS, BET = 0, 1
_TO_OS_ACTION = {Action.CHECK: PASS, Action.FOLD: PASS, Action.BET: BET, Action.CALL: BET}
_HISTORY_CODE = {"x": "p", "f": "p", "b": "b", "c": "b"}


def to_openspiel_key(key: str) -> str:
    card, history = key.split("|")
    return f"{int(Card[card])}{''.join(_HISTORY_CODE[ch] for ch in history)}"


def to_openspiel_policy(game: Any, strategy: Strategy) -> Any:
    tabular = os_policy.TabularPolicy(game)
    for key, dist in strategy.items():
        row = tabular.policy_for_key(to_openspiel_key(key))
        row[:] = 0.0
        for action, prob in dist.items():
            row[_TO_OS_ACTION[action]] = prob
    return tabular


def from_openspiel_policy(tabular: Any) -> dict[str, dict[Action, float]]:
    result = {}
    for key, spec in information_sets().items():
        row = tabular.policy_for_key(to_openspiel_key(key))
        result[key] = {a: float(row[_TO_OS_ACTION[a]]) for a in spec.actions}
    return result


@pytest.fixture(scope="module")
def game() -> Any:
    return pyspiel.load_game("kuhn_poker")


def test_key_mapping_covers_openspiel_information_states(game: Any) -> None:
    ours = {to_openspiel_key(k) for k in information_sets()}
    assert ours == set(os_policy.TabularPolicy(game).state_lookup)


def test_uniform_exploitability_matches(game: Any) -> None:
    report = exploitability_report(uniform_strategy())
    uniform = to_openspiel_policy(game, uniform_strategy())
    assert report.nash_conv == pytest.approx(os_exploitability.nash_conv(game, uniform), abs=1e-12)


def openspiel_simultaneous_cfr(game: Any) -> Any:
    """OpenSpiel's CFR configured like the original paper and like PokerMind.

    The public ``os_cfr.CFRSolver`` uses *alternating* updates; the base class
    with ``alternating_updates=False`` is the simultaneous-update vanilla CFR.
    """
    return os_cfr._CFRSolver(
        game, regret_matching_plus=False, alternating_updates=False, linear_averaging=False
    )


@pytest.mark.parametrize("iterations", [1, 10, 100, 1_000])
def test_cfr_average_policy_matches_openspiel_cfr(game: Any, iterations: int) -> None:
    """Vanilla CFR with simultaneous updates is deterministic, so the two
    independent implementations must agree to rounding error."""
    ours = CFRSolver()
    ours.iterate(iterations)
    solver = openspiel_simultaneous_cfr(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    theirs = from_openspiel_policy(solver.average_policy())
    avg = ours.average_strategy()
    for key in avg:
        for action in avg[key]:
            assert avg[key][action] == pytest.approx(theirs[key][action], abs=1e-10), key

    os_nash_conv = os_exploitability.nash_conv(game, to_openspiel_policy(game, avg))
    assert exploitability_report(avg).nash_conv == pytest.approx(os_nash_conv, abs=1e-12)


def test_openspiel_alternating_cfr_also_converges_to_the_same_equilibrium_set(game: Any) -> None:
    """Different update schedule, different iterates; both must approach exploitability 0."""
    solver = os_cfr.CFRSolver(game)  # alternating updates
    for _ in range(2_000):
        solver.evaluate_and_update_policy()
    theirs = from_openspiel_policy(solver.average_policy())
    ours = CFRSolver()
    ours.iterate(2_000)
    assert exploitability_report(theirs).exploitability < 0.01
    assert exploitability_report(ours.average_strategy()).exploitability < 0.01
