"""M5: public contexts, records, simulated opponents, Bayesian model, change detection."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pokermind.cli import main
from pokermind.evaluation.game_exact import information_sets, validate_game_strategy
from pokermind.experiments.opponent_study import (
    OpponentStudyConfig,
    run_opponent_study,
    stream_id,
)
from pokermind.games.leduc import CALL, FOLD, RAISE, LeducGame
from pokermind.learning.game_cfr import GameCFRSolver
from pokermind.opponents.bayes import (
    ChangeDetectingModel,
    DetectorConfig,
    ModelConfig,
    OpponentModel,
    PopulationPrior,
    beta_cdf,
    beta_ppf,
    raw_frequency,
)
from pokermind.opponents.context import (
    PublicDecision,
    PublicHandRecord,
    action_set_name,
    context_levels,
    public_context,
    public_record,
)
from pokermind.opponents.exact import context_stats, expected_scores
from pokermind.opponents.play import play_hand, play_records
from pokermind.opponents.population import (
    PARAMETRIC_RANGES,
    Opponent,
    ParametricParams,
    build_population,
    parametric_distribution,
    showdown_equity,
)
from pokermind.strategy import game_strategy_digest, save_game_strategy

GAME = LeducGame()
SPECS = information_sets(GAME)
UNIFORM = {k: {a: 1.0 / len(s.actions) for a in s.actions} for k, s in SPECS.items()}


@pytest.fixture(scope="module")
def hero() -> dict[str, dict[int, float]]:
    solver = GameCFRSolver(GAME)
    solver.iterate(20)
    return solver.average_strategy()


@pytest.fixture(scope="module")
def population(hero: dict[str, dict[int, float]]) -> dict[str, list[Opponent]]:
    return build_population(GAME, hero, per_family=2, switching_per_split=2)


def record(hand: int, *decisions: tuple[str, tuple[int, ...], int]) -> PublicHandRecord:
    return PublicHandRecord(
        hand, hand % 2, tuple(PublicDecision(c, legal, a) for c, legal, a in decisions), None, True
    )


# -- contexts and public records -----------------------------------------------------------


def test_public_context_strings() -> None:
    state = GAME.state_from_moves((5, 0, RAISE))
    assert public_context(state) == "R1:r"
    state = GAME.state_from_moves((5, 0, CALL, RAISE, CALL, 1, CALL))
    assert public_context(state) == "R2:J:crc:c"
    with pytest.raises(ValueError):
        public_context(GAME.initial_state())


def test_context_levels_and_action_sets() -> None:
    assert context_levels("R2:J:crc:c", (CALL, RAISE)) == ("R2:J:crc:c", "R2:c", "R2:cr", "cr")
    assert context_levels("R1:r", (FOLD, CALL, RAISE)) == ("R1:r", "R1:r", "R1:fcr", "fcr")
    assert action_set_name((FOLD, CALL)) == "fc"
    with pytest.raises(ValueError):
        context_levels("R3:x", (CALL, RAISE))
    with pytest.raises(ValueError):
        action_set_name((RAISE,))


def test_there_are_96_opponent_contexts_with_fixed_action_sets() -> None:
    stats = context_stats(GAME, UNIFORM, UNIFORM)
    assert len(stats.weight) == 96
    assert sum(1 for c in stats.weight if c.startswith("R1")) == 6
    for ctx, legal in stats.legal.items():
        context_levels(ctx, legal)  # consistent action set at every level


def test_records_contain_only_public_information(hero: dict[str, dict[int, float]]) -> None:
    """Swapping the opponent's card never changes the record of a hand that ended by a fold;
    at a showdown the record reveals exactly the opponent's card."""
    folds = showdowns = 0
    for hand in range(300):
        moves = play_hand(GAME, hero, UNIFORM, hand, stream=7)
        opp = hand % 2
        rec = public_record(GAME, moves, opp, hand)
        assert all(d.context.startswith(("R1:", "R2:")) for d in rec.decisions)
        if rec.ended_by_fold:
            folds += 1
            assert rec.showdown_card is None
            public = None
            state = GAME.state_from_moves(moves)
            public = state.public
            for card in set(range(6)) - {
                moves[0],
                moves[1],
                *(() if public is None else (public,)),
            }:
                alt = list(moves)
                alt[opp] = card
                assert public_record(GAME, tuple(alt), opp, hand) == rec
        else:
            showdowns += 1
            assert rec.showdown_card == moves[opp]
    assert folds > 20 and showdowns > 20


def test_record_requires_complete_hand() -> None:
    with pytest.raises(ValueError):
        public_record(GAME, (5, 0, RAISE), 0)
    with pytest.raises(ValueError):
        public_record(GAME, (5, 0, FOLD), 2)


# -- opponents -------------------------------------------------------------------------------


def test_showdown_equity_values() -> None:
    assert [showdown_equity(c, None) for c in (0, 2, 4)] == pytest.approx([0.3, 0.5, 0.7])
    assert showdown_equity(0, 1) == 1.0  # J0 pairs J1
    assert showdown_equity(4, 0) == pytest.approx(0.625)  # K vs J1, Q0, Q1, K1


def test_parametric_tendencies() -> None:
    def dist(**kw: float) -> dict[int, float]:
        params = ParametricParams(**{"tightness": 0.35, "aggression": 0.5, "bluff": 0.1,
                                     "tremble": 0.0, **kw})  # fmt: skip
        return parametric_distribution(params, 0.4, (FOLD, CALL, RAISE))

    assert dist(tightness=0.65)[FOLD] > dist(tightness=0.05)[FOLD]
    assert dist(aggression=0.9)[RAISE] > dist(aggression=0.1)[RAISE]
    capped = parametric_distribution(ParametricParams(0.3, 0.9, 0.3, 0.0), 0.9, (FOLD, CALL))
    assert set(capped) == {FOLD, CALL} and sum(capped.values()) == pytest.approx(1.0)
    trembled = parametric_distribution(ParametricParams(0.3, 0.0, 0.0, 0.03), 0.9, (CALL, RAISE))
    assert trembled[RAISE] == pytest.approx(0.015)


def test_population_is_valid_deterministic_and_split(
    hero: dict[str, dict[int, float]],
    population: dict[str, list[Opponent]],
) -> None:
    again = build_population(GAME, hero, per_family=2, switching_per_split=2)
    digests: dict[str, set[str]] = {}
    for split, opponents in population.items():
        assert [o.name for o in again[split]] == [o.name for o in opponents]
        for o, o2 in zip(opponents, again[split], strict=True):
            assert o.policies == o2.policies
            for p in o.policies:
                validate_game_strategy(GAME, p, infosets=SPECS)
                digests.setdefault(split, set()).add(game_strategy_digest(p, GAME.action_codes))
            if o.family in PARAMETRIC_RANGES:
                for name, (lo, hi) in PARAMETRIC_RANGES[o.family].items():
                    assert lo <= o.parameters[name] <= hi
    assert not digests["development"] & digests["heldout"]
    names = {o.name for o in population["development"]} | {o.name for o in population["heldout"]}
    assert len(names) == len(population["development"]) + len(population["heldout"])


def test_switching_opponent_changes_policy(population: dict[str, list[Opponent]]) -> None:
    switching = [o for o in population["heldout"] if o.is_switching]
    assert switching
    for o in switching:
        t = o.switch_hand
        assert t is not None
        assert o.policy_at(t - 1) is o.policies[0]
        assert o.policy_at(t) is o.policies[1]
        assert o.policies[0] != o.policies[1]


# -- exact public marginals ---------------------------------------------------------------------


def test_exact_marginals_match_simulation(hero: dict[str, dict[int, float]]) -> None:
    stats = context_stats(GAME, hero, UNIFORM)
    for p in stats.marginal.values():
        assert p.sum() == pytest.approx(1.0)
    # Opponent in seat 0 opens every other hand: context "R1:" has weight 1/2.
    assert stats.weight["R1:"] == pytest.approx(0.5)
    assert stats.marginal["R1:"] == pytest.approx([0.5, 0.5])
    records = play_records(GAME, hero, lambda h: UNIFORM, 20_000, stream=3)
    counts: dict[str, int] = {}
    for r in records:
        for d in r.decisions:
            counts[d.context] = counts.get(d.context, 0) + 1
    total = sum(counts.values()) / len(records)
    assert total == pytest.approx(stats.decisions_per_hand, rel=0.02)
    for ctx in ("R1:", "R1:c", "R1:r", "R1:cr"):
        assert counts[ctx] / len(records) == pytest.approx(stats.weight[ctx], abs=0.01)


def test_oracle_has_zero_excess_and_card_aware_bound(hero: dict[str, dict[int, float]]) -> None:
    pop = build_population(GAME, hero, per_family=1, switching_per_split=0)
    opponent = pop["heldout"][0].policies[0]
    stats = context_stats(GAME, hero, opponent)
    oracle = expected_scores(stats, lambda c, legal: stats.marginal[c])
    assert oracle.excess_log_loss == pytest.approx(0.0, abs=1e-12)
    assert oracle.card_aware_log_loss <= oracle.oracle_log_loss + 1e-12
    uniform = expected_scores(stats, lambda c, legal: np.full(len(legal), 1 / len(legal)))
    assert uniform.excess_log_loss > 0


# -- the Bayesian model -------------------------------------------------------------------------


def test_beta_functions_against_closed_forms() -> None:
    for x in (0.1, 0.37, 0.9):
        assert beta_cdf(x, 1, 1) == pytest.approx(x, abs=1e-12)
        assert beta_cdf(x, 2, 1) == pytest.approx(x * x, abs=1e-12)
        assert beta_cdf(x, 1, 3) == pytest.approx(1 - (1 - x) ** 3, abs=1e-12)
    assert beta_ppf(0.05, 1, 1) == pytest.approx(0.05, abs=1e-10)
    assert beta_ppf(0.5, 3, 1) == pytest.approx(0.5 ** (1 / 3), abs=1e-10)
    assert beta_ppf(beta_cdf(0.3, 2.5, 7.5), 2.5, 7.5) == pytest.approx(0.3, abs=1e-10)
    assert beta_cdf(0.3, 4, 6) == pytest.approx(1 - beta_cdf(0.7, 6, 4), abs=1e-12)


def test_zero_observations_return_the_prior() -> None:
    model = OpponentModel(ModelConfig(4, 4))
    assert model.predict("R1:r", (FOLD, CALL, RAISE)) == pytest.approx([1 / 3] * 3)
    prior = PopulationPrior(
        (
            {"R1:r": np.array([0.2, 0.5, 0.3])},
            {"R1:r": np.array([0.3, 0.4, 0.3])},
            {"R1:fcr": np.array([0.25, 0.5, 0.25])},
            {"fcr": np.array([0.3, 0.5, 0.2])},
        )
    )
    with_prior = OpponentModel(ModelConfig(4, 4), prior)
    assert with_prior.predict("R1:r", (FOLD, CALL, RAISE)) == pytest.approx([0.2, 0.5, 0.3])
    assert with_prior.decisions_observed == 0
    flat = OpponentModel(ModelConfig(5, 1, hierarchical=False), prior)
    assert flat.posterior_parameters("R1:r", (FOLD, CALL, RAISE)) == pytest.approx([1, 2.5, 1.5])


def test_hierarchical_update_matches_hand_calculation() -> None:
    model = OpponentModel(ModelConfig(1.0, 1.0))
    model.observe_hand(record(0, ("R1:", (CALL, RAISE), CALL)))
    # every level has counts [1, 0]; base uniform -> 3/4, 7/8, 15/16, 31/32
    assert model.predict("R1:", (CALL, RAISE)) == pytest.approx([31 / 32, 1 / 32])
    # sparse fallback: an unseen context sharing only the action set "cr"
    assert model.predict("R2:J:cc:", (CALL, RAISE)) == pytest.approx([0.75, 0.25])
    # unrelated action set: still uniform
    assert model.predict("R1:r", (FOLD, CALL, RAISE)) == pytest.approx([1 / 3] * 3)
    assert model.opportunities("R1:") == 1 and model.opportunities("R1:c") == 0


def test_flat_dirichlet_posterior_and_uncertainty() -> None:
    model = OpponentModel(ModelConfig(3.0, 1.0, hierarchical=False))
    for hand, action in enumerate([FOLD, CALL, CALL, RAISE, CALL]):
        model.observe_hand(record(hand, ("R1:r", (FOLD, CALL, RAISE), action)))
    params = model.posterior_parameters("R1:r", (FOLD, CALL, RAISE))
    assert params == pytest.approx([2, 4, 2])
    assert model.predict("R1:r", (FOLD, CALL, RAISE)) == pytest.approx([0.25, 0.5, 0.25])
    sd = model.posterior_std("R1:r", (FOLD, CALL, RAISE))
    assert sd[1] == pytest.approx(math.sqrt(0.5 * 0.5 / 9))
    lo, hi = model.credible_interval("R1:r", (FOLD, CALL, RAISE))
    assert lo[1] == pytest.approx(beta_ppf(0.05, 4, 4)) and hi[1] == pytest.approx(
        beta_ppf(0.95, 4, 4)
    )
    inside = model.interval_contains("R1:r", (FOLD, CALL, RAISE), np.array([0.25, 0.5, 0.25]))
    assert inside.all()
    assert not model.interval_contains("R1:r", (FOLD, CALL, RAISE), np.array([0.98, 0.01, 0.01]))[0]
    # uncertainty shrinks with data
    fresh = OpponentModel(ModelConfig(3.0, 1.0, hierarchical=False))
    assert (fresh.posterior_std("R1:r", (FOLD, CALL, RAISE)) > sd).all()


def test_raw_frequency_and_illegal_updates() -> None:
    model = OpponentModel(ModelConfig())
    assert raw_frequency(model, "R1:", (CALL, RAISE)) == pytest.approx([0.5, 0.5])
    model.update("R1:", (CALL, RAISE), RAISE)
    assert raw_frequency(model, "R1:", (CALL, RAISE)) == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError):
        model.update("R1:", (CALL, RAISE), FOLD)


def test_decay_and_showdown_cards_are_ignored() -> None:
    model = OpponentModel(ModelConfig(1.0, 1.0, decay=0.5))
    model.observe_hand(record(0, ("R1:", (CALL, RAISE), CALL)))
    model.observe_hand(record(1, ("R1:", (CALL, RAISE), CALL)))
    assert model.counts[0]["R1:"] == pytest.approx([1.5, 0.0])
    a, b = OpponentModel(ModelConfig()), OpponentModel(ModelConfig())
    decision = (PublicDecision("R1:", (CALL, RAISE), CALL),)
    a.observe_hand(PublicHandRecord(0, 0, decision, None, True))
    b.observe_hand(PublicHandRecord(0, 0, decision, 5, False))
    assert {k: v.tolist() for k, v in a.counts[0].items()} == {
        k: v.tolist() for k, v in b.counts[0].items()
    }


def test_change_detector_alarms_after_a_switch_only() -> None:
    model = ChangeDetectingModel(ModelConfig(2.0, 2.0), DetectorConfig(0.8, 4.0, 0.0))
    for hand in range(120):
        action = CALL if hand < 60 else RAISE
        model.observe_hand(record(hand, ("R1:", (CALL, RAISE), action)))
    assert model.alarms and all(a >= 60 for a in model.alarms)
    assert model.alarms[0] < 70
    assert model.predict("R1:", (CALL, RAISE))[1] > 0.8
    with pytest.raises(ValueError):
        ChangeDetectingModel(ModelConfig(decay=0.9), DetectorConfig())


def test_cumulative_model_learns_a_stationary_opponent(
    hero: dict[str, dict[int, float]],
    population: dict[str, list[Opponent]],
) -> None:
    opponent = next(o for o in population["heldout"] if o.family == "aggressive")
    stats = context_stats(GAME, hero, opponent.policies[0])
    model = OpponentModel(ModelConfig(8, 64, hierarchical=False))
    excess = [expected_scores(stats, model.predict).excess_log_loss]
    for r in play_records(GAME, hero, opponent.policy_at, 2_000, stream=11):
        model.observe_hand(r)
    excess.append(expected_scores(stats, model.predict).excess_log_loss)
    assert excess[1] < 0.5 * excess[0]


# -- simulation and study -----------------------------------------------------------------------


def test_play_is_repeatable_and_streams_are_independent(hero: dict[str, dict[int, float]]) -> None:
    a = play_records(GAME, hero, lambda h: UNIFORM, 50, stream=1)
    assert a == play_records(GAME, hero, lambda h: UNIFORM, 50, stream=1)
    assert a != play_records(GAME, hero, lambda h: UNIFORM, 50, stream=2)
    assert play_records(GAME, hero, lambda h: UNIFORM, 10, stream=1, start=40) == a[40:]
    assert stream_id("x", 0, "estimate") != stream_id("x", 0, "evaluate")


def test_study_smoke_and_cli(hero: dict[str, dict[int, float]], tmp_path: Path) -> None:
    path = tmp_path / "hero.json"
    save_game_strategy(path, GAME.name, GAME.action_codes, hero)
    config = OpponentStudyConfig(
        hero_path=path, seeds=2, tuning_seeds=1, per_family=1, switching_per_split=1,
        estimation_hands=60, evaluation_hands=40, budgets=(0, 10, 60),
        alpha_fine_grid=(2.0,), alpha_coarse_grid=(4.0,), decay_grid=(0.95,),
        detector_grid=(DetectorConfig(0.9, 5.0, 0.0),),
    )  # fmt: skip
    doc = run_opponent_study(config, tmp_path / "run")
    assert (tmp_path / "run" / "metrics.json").is_file()
    held = doc["sample_efficiency"]["heldout"]
    assert set(held) >= {"uniform", "population_prior", "laplace", "main"}
    assert held["main"]["0"]["excess_log_loss"]["runs"] == 6 * 2
    names = {o["name"] for o in doc["population"]["heldout"]}
    assert not names & {o["name"] for o in doc["population"]["development"]}
    assert main(["opponent-study", "--hero", str(tmp_path / "missing.json")]) == 2
