"""Exact evaluator, checked against an independent brute-force oracle.

The oracle below shares nothing with ``pokermind.evaluation`` or the CFR code
except the M0 engine. It computes expected values with exact ``Fraction``
arithmetic, and best responses by enumerating all 2^6 = 64 pure strategies of
the responding player.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from fractions import Fraction

import numpy as np
import pytest

from pokermind.agents import AlwaysBetAgent, AlwaysPassAgent, ThresholdAgent
from pokermind.evaluation import (
    GAME_VALUE,
    InvalidStrategyError,
    best_response,
    expected_value,
    expected_value_seat_averaged,
    exploitability,
    exploitability_report,
    information_sets,
    kuhn_equilibrium,
    max_deviation_from_equilibrium,
    nash_conv,
    uniform_strategy,
    validate_strategy,
)
from pokermind.evaluation.reference import PLAYER0_KEYS, PLAYER1_KEYS, as_float_strategy
from pokermind.experiments import deterministic_agent_strategy
from pokermind.games.kuhn import Action, KuhnConfig, KuhnState, all_deals, initial_state

X, B, C, F = Action.CHECK, Action.BET, Action.CALL, Action.FOLD
Number = Fraction | float
Table = Mapping[str, Mapping[Action, Number]]

# -- independent oracle ----------------------------------------------------------


def oracle_ev(seat0: Table, seat1: Table, config: KuhnConfig | None = None) -> Number:
    def ev(state: KuhnState) -> Number:
        if state.is_terminal:
            return Fraction(state.payoffs()[0])
        p = state.current_player
        table = seat0 if p == 0 else seat1
        dist = table[state.information_state(p).key]
        return sum((prob * ev(state.apply(a)) for a, prob in dist.items()), Fraction(0))

    deals = all_deals()
    return sum((ev(initial_state(c, config)) for c in deals), Fraction(0)) / len(deals)


def pure_strategies(player: int) -> list[dict[str, dict[Action, Fraction]]]:
    specs = {k: s for k, s in information_sets().items() if s.player == player}
    keys = sorted(specs)
    result = []
    for choice in itertools.product(*(specs[k].actions for k in keys)):
        result.append(
            {
                k: {a: Fraction(int(a is chosen)) for a in specs[k].actions}
                for k, chosen in zip(keys, choice, strict=True)
            }
        )
    return result


PURE = {0: pure_strategies(0), 1: pure_strategies(1)}


def oracle_best_response_value(strategy: Table, player: int) -> Number:
    if player == 0:
        return max(oracle_ev(pure, strategy) for pure in PURE[0])
    return max(-oracle_ev(strategy, pure) for pure in PURE[1])


def random_strategy(seed: int) -> dict[str, dict[Action, float]]:
    rng = np.random.default_rng(seed)
    table = {}
    for key, spec in information_sets().items():
        probs = rng.dirichlet(np.ones(len(spec.actions)) * 0.7)
        table[key] = {a: float(p) for a, p in zip(spec.actions, probs, strict=True)}
    return table


def test_oracle_enumerates_64_pure_strategies_per_player() -> None:
    assert len(PURE[0]) == len(PURE[1]) == 64


# -- expected value ------------------------------------------------------------


def test_uniform_self_play_value_is_one_eighth() -> None:
    u = uniform_strategy()
    assert expected_value((u, u)) == pytest.approx(1 / 8)
    assert oracle_ev(u, u) == Fraction(1, 8)


def test_random_vs_rule_exact_values() -> None:
    """Hand-checked in the M0 verification: seat 0 -1/4, seat 1 -1/6."""
    u, rule = uniform_strategy(), deterministic_agent_strategy(ThresholdAgent())
    assert expected_value((u, rule)) == pytest.approx(-1 / 4)
    assert -expected_value((rule, u)) == pytest.approx(-1 / 6)
    assert expected_value_seat_averaged(u, rule) == pytest.approx(-5 / 24)


@pytest.mark.parametrize("seed", range(10))
def test_expected_value_matches_oracle(seed: int) -> None:
    a, b = random_strategy(seed), random_strategy(seed + 100)
    assert expected_value((a, b)) == pytest.approx(float(oracle_ev(a, b)), abs=1e-12)


def test_expected_value_with_other_stakes() -> None:
    cfg = KuhnConfig(ante=2, bet_size=3)
    always_bet = deterministic_agent_strategy(AlwaysBetAgent(), cfg)
    # Every hand is bet-call: the higher card wins 5, so the value is 0 by symmetry.
    assert expected_value((always_bet, always_bet), cfg) == pytest.approx(0.0)
    passive = deterministic_agent_strategy(AlwaysPassAgent(), cfg)
    # Always-bet vs always-pass: seat 0 bets, seat 1 folds every hand -> +ante.
    assert expected_value((always_bet, passive), cfg) == pytest.approx(2.0)


# -- best response ---------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("player", [0, 1])
def test_best_response_matches_brute_force(seed: int, player: int) -> None:
    strategy = random_strategy(seed)
    br = best_response(strategy, player)
    assert br.value == pytest.approx(float(oracle_best_response_value(strategy, player)), abs=1e-12)
    # The returned policy actually achieves that value, and is one action per info set.
    pure = br.as_strategy()
    assert set(pure) == {k for k, s in information_sets().items() if s.player == player}
    achieved = (
        expected_value((pure, strategy)) if player == 0 else -expected_value((strategy, pure))
    )
    assert achieved == pytest.approx(br.value, abs=1e-12)


def exact_uniform() -> dict[str, dict[Action, Fraction]]:
    return {
        k: {a: Fraction(1, len(s.actions)) for a in s.actions}
        for k, s in information_sets().items()
    }


def test_best_response_to_uniform() -> None:
    u = uniform_strategy()
    assert best_response(u, 0).value == pytest.approx(1 / 2)
    assert best_response(u, 1).value == pytest.approx(5 / 12)
    assert oracle_best_response_value(exact_uniform(), 0) == Fraction(1, 2)
    assert oracle_best_response_value(exact_uniform(), 1) == Fraction(5, 12)


def test_best_response_exploits_the_honest_rule_agent() -> None:
    rule = deterministic_agent_strategy(ThresholdAgent())
    br = best_response(rule, 1)
    # The rule agent (P0) bets only with K, so P1 folds Q to a bet. After a
    # check P0 holds J or Q and calls with Q: bluffing J into a Q loses 2 instead
    # of 1, so P1 checks J. With Q after a check P0 holds J, which folds to a bet
    # (+1) and loses a showdown (+1): a tie, broken towards the passive action.
    assert br.policy["Q|b"] is F
    # 'K|b' is unreachable (P0 bets only with K, which P1 holds): every action is
    # worth 0 and the deterministic tie-break picks the passive one.
    assert br.policy["K|b"] is F
    assert br.policy["J|x"] is X
    assert br.policy["Q|x"] is X
    assert br.policy["K|x"] is B
    assert br.value == pytest.approx(float(oracle_best_response_value(rule, 1)))


def test_best_response_respects_information_sets() -> None:
    """One action per information set, never per history: a strategy that
    knew the opponent's card could beat the brute-force optimum, ours cannot."""
    for seed in range(4):
        s = random_strategy(seed)
        for p in (0, 1):
            assert best_response(s, p).value <= oracle_best_response_value(s, p) + 1e-12


# -- NashConv and exploitability -----------------------------------------------------


def test_uniform_nash_conv() -> None:
    report = exploitability_report(uniform_strategy())
    assert report.nash_conv == pytest.approx(11 / 12)
    assert report.exploitability == pytest.approx(11 / 24)
    assert report.best_response_values == pytest.approx((1 / 2, 5 / 12))
    assert report.on_policy_values == pytest.approx((1 / 8, -1 / 8))


@pytest.mark.parametrize("alpha", [Fraction(0), Fraction(1, 10), Fraction(1, 6), Fraction(1, 3)])
def test_known_equilibria_have_zero_exploitability(alpha: Fraction) -> None:
    exact = kuhn_equilibrium(alpha)
    # Exact arithmetic with the oracle: value -1/18, nobody can gain.
    assert oracle_ev(exact, exact) == GAME_VALUE
    assert oracle_best_response_value(exact, 0) == GAME_VALUE
    assert oracle_best_response_value(exact, 1) == -GAME_VALUE
    # The float evaluator agrees within rounding.
    eq = as_float_strategy(exact)
    report = exploitability_report(eq)
    assert abs(report.nash_conv) < 1e-12
    assert report.on_policy_values[0] == pytest.approx(-1 / 18, abs=1e-12)
    assert max_deviation_from_equilibrium(eq) < 1e-12


def test_perturbed_equilibrium_is_exploitable() -> None:
    eq = as_float_strategy(kuhn_equilibrium(0))
    eq["Q|b"] = {F: 0.4, C: 0.6}  # P1 calls too often with Q
    assert exploitability(eq) > 1e-3
    assert max_deviation_from_equilibrium(eq) == pytest.approx(0.6 - 1 / 3)


@pytest.mark.parametrize("seed", range(6))
def test_nash_conv_definition_and_bounds(seed: int) -> None:
    s = random_strategy(seed)
    report = exploitability_report(s)
    br0 = float(oracle_best_response_value(s, 0))
    br1 = float(oracle_best_response_value(s, 1))
    u0 = float(oracle_ev(s, s))
    assert report.nash_conv == pytest.approx((br0 - u0) + (br1 + u0), abs=1e-12)
    assert report.nash_conv == pytest.approx(br0 + br1, abs=1e-12)  # zero-sum identity
    assert report.exploitability == pytest.approx(report.nash_conv / 2)
    assert nash_conv(s) == report.nash_conv
    # Game-value bounds: nobody can do worse than the value by best-responding.
    assert br0 >= -1 / 18 - 1e-12
    assert br1 >= 1 / 18 - 1e-12


@pytest.mark.parametrize("seed", range(6))
def test_exploitability_bounds_worst_case_seat_averaged_loss(seed: int) -> None:
    """A strategy with exploitability ε loses at most ε per hand, seat-averaged,
    against any opponent."""
    s = random_strategy(seed)
    eps = exploitability(s)
    for opponent in (random_strategy(seed + 50), uniform_strategy()):
        assert expected_value_seat_averaged(s, opponent) >= -eps - 1e-12
    for p0, p1 in itertools.product(PURE[0][::7], PURE[1][::7]):
        opponent = as_float_strategy({**p0, **p1})
        assert expected_value_seat_averaged(s, opponent) >= -eps - 1e-12


# -- validation --------------------------------------------------------------------


def _uniform() -> dict[str, dict[Action, float]]:
    return uniform_strategy()


def test_valid_strategies_pass() -> None:
    validate_strategy(_uniform())
    validate_strategy(as_float_strategy(kuhn_equilibrium(Fraction(1, 6))))
    validate_strategy({k: v for k, v in _uniform().items() if k in PLAYER0_KEYS}, (0,))


def _bad(key: str, dist: dict[Action, float]) -> dict[str, dict[Action, float]]:
    s = _uniform()
    s[key] = dist
    return s


@pytest.mark.parametrize(
    ("strategy", "message"),
    [
        ({k: v for k, v in _uniform().items() if k != "K|xb"}, "missing"),
        ({**_uniform(), "K|J": {X: 1.0}}, "unknown"),
        ({**_uniform(), "K|xx": {X: 1.0}}, "unknown"),  # terminal history
        (_bad("K|", {X: 0.5, C: 0.5}), "legal"),  # CALL is illegal when not facing a bet
        (_bad("K|b", {F: 1.0}), "legal"),  # a legal action is missing
        (_bad("K|b", {F: 0.5, C: 0.5, B: 0.0}), "legal"),
        (_bad("Q|x", {X: 1.2, B: -0.2}), "invalid probability"),
        (_bad("Q|x", {X: math.nan, B: 0.5}), "invalid probability"),
        (_bad("Q|x", {X: math.inf, B: 0.0}), "invalid probability"),
        (_bad("Q|x", {X: 0.5, B: 0.4}), "sum"),
        (_bad("Q|x", {X: True, B: False}), "not a number"),
    ],
)
def test_malformed_strategies_rejected(
    strategy: dict[str, dict[Action, float]], message: str
) -> None:
    with pytest.raises(InvalidStrategyError, match=message):
        validate_strategy(strategy)


def test_evaluators_validate_their_inputs() -> None:
    bad = _bad("K|", {X: 0.9, B: 0.0})
    with pytest.raises(InvalidStrategyError):
        exploitability(bad)
    with pytest.raises(InvalidStrategyError):
        expected_value((bad, _uniform()))
    with pytest.raises(InvalidStrategyError):
        best_response(bad, 1)  # P1's best response reads P0's strategy
    # ...but a best response for P0 only needs P1's information sets.
    p1_only = {k: v for k, v in _uniform().items() if k in PLAYER1_KEYS}
    assert best_response(p1_only, 0).value == pytest.approx(1 / 2)


def test_information_sets_partition_by_player() -> None:
    specs = information_sets()
    assert {k for k, s in specs.items() if s.player == 0} == set(PLAYER0_KEYS)
    assert {k for k, s in specs.items() if s.player == 1} == set(PLAYER1_KEYS)
