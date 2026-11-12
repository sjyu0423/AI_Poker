"""Vanilla CFR: regret matching, the update rule, averaging, determinism, convergence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pokermind.evaluation import (
    exploitability,
    exploitability_report,
    information_sets,
    kuhn_equilibrium,
)
from pokermind.evaluation.reference import PLAYER1_KEYS
from pokermind.games.kuhn import Action, Card, KuhnConfig
from pokermind.learning import CFRSolver, normalize_or_uniform, regret_matching, train_cfr
from pokermind.strategy import strategy_digest

X, B, C, F = Action.CHECK, Action.BET, Action.CALL, Action.FOLD

# -- regret matching ---------------------------------------------------------


def test_regret_matching_proportional_to_positive_regret() -> None:
    np.testing.assert_allclose(regret_matching(np.array([1.0, 3.0])), [0.25, 0.75])
    np.testing.assert_allclose(regret_matching(np.array([2.0, -5.0, 6.0])), [0.25, 0.0, 0.75])


@pytest.mark.parametrize("regrets", [[0.0, 0.0], [-1.0, -2.0], [0.0, -3.0], [-1e-300, 0.0]])
def test_regret_matching_uniform_without_positive_regret(regrets: list[float]) -> None:
    np.testing.assert_array_equal(regret_matching(np.array(regrets)), [0.5, 0.5])


def test_regret_matching_single_positive_is_pure() -> None:
    np.testing.assert_array_equal(regret_matching(np.array([-4.0, 1e-12])), [0.0, 1.0])


@pytest.mark.parametrize(
    "regrets", [np.array([]), np.array([[1.0, 2.0]]), np.array([np.nan, 1.0]), np.array([np.inf])]
)
def test_regret_matching_rejects_malformed_input(regrets: np.ndarray) -> None:
    with pytest.raises(ValueError):
        regret_matching(regrets)


@pytest.mark.parametrize("seed", range(20))
def test_regret_matching_output_is_a_distribution(seed: int) -> None:
    rng = np.random.default_rng(seed)
    regrets = rng.normal(size=int(rng.integers(1, 5)))
    sigma = regret_matching(regrets)
    assert np.all(sigma >= 0)
    assert math.isclose(sigma.sum(), 1.0)
    assert np.all(sigma[regrets <= 0] == 0) or np.all(regrets <= 0)


def test_normalize_or_uniform() -> None:
    np.testing.assert_allclose(normalize_or_uniform(np.array([1.0, 3.0])), [0.25, 0.75])
    np.testing.assert_array_equal(normalize_or_uniform(np.zeros(2)), [0.5, 0.5])


# -- structure and information hiding ----------------------------------------


def test_tables_cover_exactly_the_information_sets_with_legal_actions() -> None:
    solver = CFRSolver()
    specs = information_sets()
    assert set(solver.tables) == set(specs)
    assert len(solver.tables) == 12
    for key, table in solver.tables.items():
        assert table.actions == specs[key].actions  # legal-action masking by construction
        assert table.player == specs[key].player
        assert table.regret_sum.shape == table.strategy_sum.shape == (len(table.actions),)


def test_keys_only_contain_own_card_and_public_history() -> None:
    cards = {c.name for c in Card}
    for key in CFRSolver().tables:
        card, history = key.split("|")
        assert card in cards
        assert set(history) <= {a.code for a in Action}
        # A key has exactly one card: no slot for the opponent's.
        assert sum(ch in cards for ch in key) == 1


def test_initial_strategies_are_uniform() -> None:
    solver = CFRSolver()
    for strategy in (solver.current_strategy(), solver.average_strategy()):
        for dist in strategy.values():
            assert list(dist.values()) == [0.5, 0.5]


# -- the update rule: worked example from docs/concepts/cfr.md §10 -------------


def test_first_iteration_matches_hand_computed_regrets() -> None:
    solver = CFRSolver()
    solver.iterate(1)
    regrets = solver.cumulative_regrets()
    # K facing a bet: call +1/4, fold -1/4 (see the worked example).
    assert regrets["K|b"][C] == pytest.approx(0.25)
    assert regrets["K|b"][F] == pytest.approx(-0.25)
    assert solver.current_strategy()["K|b"] == {F: 0.0, C: 1.0}
    # The average after one uniform iteration is still uniform.
    assert solver.average_strategy()["K|b"] == {F: 0.5, C: 0.5}
    # J facing a bet: call always loses 2, fold loses 1. Weight per history 1/12.
    # r(call) = 2 * 1/12 * (-2 - (-1.5)) = -1/12, r(fold) = 2 * 1/12 * (-1 + 1.5) = +1/12
    assert regrets["J|b"][C] == pytest.approx(-1 / 12)
    assert regrets["J|b"][F] == pytest.approx(1 / 12)


def test_first_iteration_root_regret_for_player0() -> None:
    """P0 with K at the root, uniform opponent. Counterfactual weight per history
    = π_c = 1/6 (no opponent action before the root).

    bet:   P1 folds (½) -> +1, calls (½) -> +2            => 1.5
    check: P1 checks (½) -> +1; P1 bets (½) -> P0 then folds (½) -1 / calls (½) +2
           => ½·1 + ½·(½·-1 + ½·2) = 0.75
    node value = ½·1.5 + ½·0.75 = 1.125. Two histories (opponent J or Q).
    """
    solver = CFRSolver()
    solver.iterate(1)
    r = solver.cumulative_regrets()["K|"]
    assert r[B] == pytest.approx(2 * (1 / 6) * (1.5 - 1.125))
    assert r[X] == pytest.approx(2 * (1 / 6) * (0.75 - 1.125))


def test_strategy_is_frozen_within_an_iteration() -> None:
    """Both visits to an information set in one iteration use the same σ^t.

    P1's 'K|x' is visited twice in iteration 1: deal (J, K), then (Q, K). If σ
    were recomputed from regrets between the visits, the second visit would
    already play 'bet' with probability 1, its node value would be 1.5, and the
    total would differ from twice the uniform-σ contribution checked here.
    """
    solver = CFRSolver()
    solver.iterate(1)
    # Uniform σ^1. bet: P0 folds/calls vs K: ½·(+1) + ½·(+2) = 1.5; check: +1.
    # value = 1.25; weight per history = π_c·π_0(check) = 1/6·½.
    r = solver.cumulative_regrets()["K|x"]
    assert r[B] == pytest.approx(2 * (1 / 12) * (1.5 - 1.25))
    assert r[X] == pytest.approx(2 * (1 / 12) * (1.0 - 1.25))


# -- average strategy weighting ---------------------------------------------------


def _own_reach(strategy: dict[str, dict[Action, float]], key: str, player: int) -> float:
    """π_i(I): product of the player's own action probabilities on the path to I."""
    card, history = key.split("|")
    codes = {a.code: a for a in Action}
    reach = 1.0
    for j, code in enumerate(history):
        if j % 2 == player:
            reach *= strategy[f"{card}|{history[:j]}"][codes[code]]
    return reach


def test_average_strategy_is_reach_weighted_mean_of_current_strategies() -> None:
    solver = CFRSolver()
    specs = information_sets()
    history: list[dict[str, dict[Action, float]]] = []
    for _ in range(6):
        history.append(solver.current_strategy())  # σ^t used during the next iteration
        solver.iterate(1)
    average = solver.average_strategy()

    discriminating = False
    for key, spec in specs.items():
        weights = [_own_reach(sigma, key, spec.player) for sigma in history]
        for action in spec.actions:
            weighted = sum(w * s[key][action] for w, s in zip(weights, history, strict=True))
            expected = weighted / sum(weights) if sum(weights) else 1 / len(spec.actions)
            assert average[key][action] == pytest.approx(expected, abs=1e-12)
            plain = sum(s[key][action] for s in history) / len(history)
            discriminating |= abs(plain - expected) > 1e-3
    # The check would also pass for an unweighted average if no information set
    # had changing reach; make sure it does not.
    assert discriminating


# -- determinism ----------------------------------------------------------------


def test_training_is_bit_for_bit_deterministic() -> None:
    a, b = train_cfr(300), train_cfr(300)
    assert a.average_strategy() == b.average_strategy()
    assert a.cumulative_regrets() == b.cumulative_regrets()
    assert strategy_digest(a.average_strategy()) == strategy_digest(b.average_strategy())


def test_incremental_training_equals_one_shot() -> None:
    solver = CFRSolver()
    for _ in range(3):
        solver.iterate(100)
    assert solver.iterations == 300
    assert solver.average_strategy() == train_cfr(300).average_strategy()


def test_iterate_rejects_negative_count() -> None:
    with pytest.raises(ValueError):
        CFRSolver().iterate(-1)


# -- convergence ------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained() -> CFRSolver:
    return train_cfr(5_000)


def test_exploitability_decreases_substantially() -> None:
    solver = CFRSolver()
    values = []
    for target in (10, 100, 1_000, 5_000):
        solver.iterate(target - solver.iterations)
        values.append(exploitability(solver.average_strategy()))
    assert values == sorted(values, reverse=True)
    assert values[-1] < values[0] / 20


def test_average_strategy_approaches_equilibrium(trained: CFRSolver) -> None:
    avg = trained.average_strategy()
    report = exploitability_report(avg)
    assert report.exploitability < 0.004
    assert report.on_policy_values[0] == pytest.approx(-1 / 18, abs=1e-3)
    # Player 1's equilibrium is unique.
    target = kuhn_equilibrium(0)
    for key in PLAYER1_KEYS:
        for action, prob in target[key].items():
            assert avg[key][action] == pytest.approx(float(prob), abs=0.03), key
    # Player 0 sits on the α-family: the three implied values of α agree.
    alpha = avg["J|"][B]
    assert 0 <= alpha <= 1 / 3 + 0.01
    assert avg["K|"][B] / 3 == pytest.approx(alpha, abs=0.02)
    assert avg["Q|xb"][C] - 1 / 3 == pytest.approx(alpha, abs=0.03)
    assert avg["Q|"][B] < 0.01
    assert avg["J|xb"][C] < 0.01
    assert avg["K|xb"][C] > 0.99


def test_current_strategy_is_not_the_converged_object(trained: CFRSolver) -> None:
    """The current strategy is usually much more exploitable than the average."""
    assert exploitability(trained.current_strategy()) > exploitability(trained.average_strategy())


def test_cfr_converges_with_non_default_stakes() -> None:
    cfg = KuhnConfig(ante=2, bet_size=3)
    solver = CFRSolver(cfg)
    solver.iterate(50)
    early = exploitability(solver.average_strategy(), cfg)
    solver.iterate(1_950)
    late = exploitability(solver.average_strategy(), cfg)
    assert late < early / 5
