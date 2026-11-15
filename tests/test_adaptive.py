"""M6: card-conditional reconstruction, best responses, budgeted mixtures, adaptation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from pokermind.adaptive.adapter import (
    AdaptationSettings,
    Adapter,
    ChangeAwareObserver,
    Priors,
    build_priors,
)
from pokermind.adaptive.reconstruction import (
    CardConditionalModel,
    HandEvidence,
    ObserverHandRecord,
    canonical_infoset,
    canonical_mean,
    evidence,
    expand,
    infoset_contexts,
    key_for,
    marginal_copy,
    observer_record,
    opponent_infoset_weights,
    reconstruction_error,
)
from pokermind.adaptive.response import (
    Mixer,
    exploitability,
    hero_best_response,
    hero_value,
    own_reach,
    realization_mixture,
)
from pokermind.adaptive.robust_opponents import (
    OUT_OF_POPULATION,
    make_bait_switch,
    make_out_of_population,
    trapper_policy,
)
from pokermind.cli import main
from pokermind.evaluation.game_exact import best_response, information_sets, validate_game_strategy
from pokermind.experiments.adaptive_study import build_world
from pokermind.games.leduc import CALL, FOLD, RAISE, LeducGame, LeducState
from pokermind.learning.game_cfr import GameCFRSolver
from pokermind.opponents.bayes import ModelConfig, OpponentModel, PopulationPrior
from pokermind.opponents.context import PublicDecision, PublicHandRecord, public_context
from pokermind.opponents.exact import context_stats
from pokermind.opponents.play import play_hand
from pokermind.opponents.population import Opponent, build_population, showdown_equity
from pokermind.strategy import save_game_strategy

GAME = LeducGame()
SPECS = information_sets(GAME)
UNIFORM = {k: {a: 1.0 / len(s.actions) for a in s.actions} for k, s in SPECS.items()}


def random_policy(seed: int) -> dict[str, dict[int, float]]:
    rng = np.random.default_rng(seed)
    out = {}
    for key, spec in SPECS.items():
        p = rng.dirichlet(np.ones(len(spec.actions)))
        out[key] = {a: float(x) for a, x in zip(spec.actions, p, strict=True)}
    return out


@pytest.fixture(scope="module")
def cfr() -> dict[str, dict[int, float]]:
    solver = GameCFRSolver(GAME)
    solver.iterate(30)
    return solver.average_strategy()


@pytest.fixture(scope="module")
def population(cfr: dict[str, dict[int, float]]) -> dict[str, list[Opponent]]:
    return build_population(GAME, cfr, per_family=1, switching_per_split=1)


@pytest.fixture(scope="module")
def priors(cfr: dict[str, dict[int, float]], population: dict[str, list[Opponent]]) -> Priors:
    dev = [o for o in population["development"] if not o.is_switching]
    public = PopulationPrior.from_opponents(
        [context_stats(GAME, cfr, o.policies[0]) for o in dev], "dev"
    )
    return build_priors(
        GAME, SPECS, cfr, [o.policies[0] for o in dev], public, infoset_contexts(GAME)
    )


def decision_states() -> Iterator[LeducState]:
    def walk(state: LeducState) -> Iterator[LeducState]:
        if state.is_terminal:
            return
        if state.is_chance_node:
            for card, _ in state.chance_outcomes():
                yield from walk(state.apply(card))
            return
        yield state
        for a in state.legal_actions():
            yield from walk(state.apply(a))

    yield from walk(GAME.initial_state())


# -- reconstruction ---------------------------------------------------------------------------


def test_key_for_matches_canonical_information_sets() -> None:
    for state in decision_states():
        player = state.current_player
        key = state.information_state_key(player)
        assert key_for(state.private[player], public_context(state)) == canonical_infoset(key)


def test_observer_records_hide_the_opponent_card_until_showdown(
    cfr: dict[str, dict[int, float]],
) -> None:
    sizes = set()
    for hand in range(200):
        moves = play_hand(GAME, cfr, UNIFORM, hand, stream=4)
        opp = hand % 2
        rec = observer_record(GAME, moves, opp, hand)
        assert rec.hero_card == moves[1 - opp]
        ev = evidence(rec)
        sizes.add(len(ev.candidates))
        if rec.public.showdown_card is None:
            assert moves[opp] in ev.candidates
            state = GAME.state_from_moves(moves)
            used = {moves[0], moves[1], *(() if state.public is None else (state.public,))}
            for card in set(range(6)) - used:
                alt = list(moves)
                alt[opp] = card
                assert observer_record(GAME, tuple(alt), opp, hand) == rec
        else:
            assert ev.candidates == (moves[opp],)
    assert sizes == {1, 4, 5}


def test_zero_observations_return_the_prior(priors: Priors) -> None:
    fit = CardConditionalModel(priors.population_theta, 4.0).fit([])
    for k, v in fit.theta.items():
        assert v == pytest.approx(priors.population_theta[k])


def test_em_step_matches_hand_calculation(priors: Priors) -> None:
    """Hero holds J0; opponent (seat 0) raised in round 1 and hero folded."""
    prior = priors.population_theta
    hand = HandEvidence((1, 2, 3, 4, 5), (("R1:", (CALL, RAISE), RAISE),))
    model = CardConditionalModel(prior, 3.0)
    fit = model.fit([hand], max_iterations=1)
    lik = {c: prior[key_for(c, "R1:")][1] for c in hand.candidates}
    total = sum(lik.values())
    counts = {"J|": 0.0, "Q|": 0.0, "K|": 0.0}
    for c, v in lik.items():
        counts[key_for(c, "R1:")] += v / total
    for k, n in counts.items():
        expected = (np.array([0.0, n]) + 3.0 * prior[k]) / (n + 3.0)
        assert fit.theta[k] == pytest.approx(expected)
        assert fit.expected_counts[k] == pytest.approx([0.0, n])
    # a revealed card puts the whole count on that rank
    shown = model.fit([HandEvidence((4,), (("R1:", (CALL, RAISE), CALL),))], max_iterations=1)
    assert shown.expected_counts["K|"] == pytest.approx([1.0, 0.0])
    assert shown.expected_counts["Q|"].sum() == 0.0


def test_reconstruction_improves_with_data(
    cfr: dict[str, dict[int, float]], priors: Priors, population: dict[str, list[Opponent]]
) -> None:
    opponent = next(o for o in population["heldout"] if o.family == "loose")
    truth = opponent.policies[0]
    weights = opponent_infoset_weights(GAME, cfr, truth)
    hands = [
        evidence(observer_record(GAME, play_hand(GAME, cfr, truth, h, 8), h % 2, h))
        for h in range(400)
    ]
    model = CardConditionalModel(priors.population_theta, 2.0)
    errors = [
        reconstruction_error(expand(model.fit(hands[:n]).theta, SPECS), truth, weights)
        for n in (0, 400)
    ]
    assert errors[1] < 0.7 * errors[0]
    assert reconstruction_error(truth, truth, weights) == 0.0


def test_gibbs_samples_are_deterministic_and_valid(priors: Priors) -> None:
    hands = [HandEvidence((1, 2, 3, 4, 5), (("R1:", (CALL, RAISE), RAISE),))] * 5
    model = CardConditionalModel(priors.population_theta, 2.0)
    a = model.posterior_samples(hands, 3, np.random.default_rng(1))
    b = model.posterior_samples(hands, 3, np.random.default_rng(1))
    for x, y in zip(a, b, strict=True):
        for k in x:
            assert np.array_equal(x[k], y[k])
            assert x[k].sum() == pytest.approx(1.0) and (x[k] > 0).all()


def test_marginal_copy_ignores_the_card(priors: Priors) -> None:
    model = OpponentModel(ModelConfig(16, 64), priors.public_prior)
    naive = marginal_copy(model.predict, SPECS, priors.infoset_contexts)
    validate_game_strategy(GAME, naive, infosets=SPECS)
    assert naive["K0|"] == naive["J1|"] == naive["Q0|"]


def test_canonical_mean_and_expand_round_trip(cfr: dict[str, dict[int, float]]) -> None:
    theta = canonical_mean([cfr], SPECS)
    assert len(theta) == 288
    validate_game_strategy(GAME, expand(theta, SPECS), infosets=SPECS)


# -- best responses and mixtures ----------------------------------------------------------------


def test_best_response_dominates_and_matches_the_evaluator() -> None:
    opponent = random_policy(3)
    br = hero_best_response(GAME, opponent, SPECS)
    validate_game_strategy(GAME, br, infosets=SPECS)
    value = hero_value(GAME, br, opponent, SPECS)
    seat_values = [best_response(GAME, opponent, p, infosets=SPECS).value for p in (0, 1)]
    assert value == pytest.approx(sum(seat_values) / 2, abs=1e-12)
    for seed in range(5):
        assert hero_value(GAME, random_policy(100 + seed), opponent, SPECS) <= value + 1e-12


def test_realization_mixture_is_normalized_and_linear(cfr: dict[str, dict[int, float]]) -> None:
    response = hero_best_response(GAME, random_policy(5), SPECS)
    rb, rr = own_reach(GAME, cfr), own_reach(GAME, response)
    opponent = random_policy(6)
    ends = [hero_value(GAME, p, opponent, SPECS) for p in (cfr, response)]
    expl = []
    for alpha in (0.0, 0.3, 0.6, 1.0):
        mix = realization_mixture(cfr, response, alpha, rb, rr)
        validate_game_strategy(GAME, mix, infosets=SPECS)
        value = hero_value(GAME, mix, opponent, SPECS)
        assert value == pytest.approx((1 - alpha) * ends[0] + alpha * ends[1], abs=1e-12)
        expl.append(exploitability(GAME, mix, SPECS))
    assert expl[1] <= 0.5 * (expl[0] + expl[2]) + 1e-12  # convex along 0, 0.3, 0.6
    assert expl[2] <= 0.5 * (expl[1] + expl[3]) + 1e-12
    with pytest.raises(ValueError):
        realization_mixture(cfr, response, 1.5, rb, rr)


def test_budget_search_enforces_the_budget_exactly(cfr: dict[str, dict[int, float]]) -> None:
    response = hero_best_response(GAME, random_policy(9), SPECS)
    base = exploitability(GAME, cfr, SPECS)
    mixer = Mixer(GAME, SPECS, cfr, response, base)
    previous = 0.0
    for extra in (0.01, 0.05, 0.2):
        search = mixer.largest_alpha(base + extra)
        assert search.exploitability <= base + extra + 1e-12
        assert exploitability(GAME, mixer.policy(search.alpha), SPECS) == pytest.approx(
            search.exploitability
        )
        assert search.alpha >= previous
        previous = search.alpha
    assert mixer.largest_alpha(base - 1e-6).alpha == 0.0  # base itself over budget
    assert mixer.largest_alpha(10.0).alpha == 1.0


# -- the adapter -----------------------------------------------------------------------------------


def test_adapter_zero_observations_and_determinism(
    cfr: dict[str, dict[int, float]], priors: Priors
) -> None:
    adapter = Adapter(GAME, SPECS, priors, AdaptationSettings(frontier_budgets=(0.05,)))
    snap = adapter.snapshot([])
    assert snap.hands == 0 and snap.prior == ("population", 2.0)
    assert snap.estimate == expand(priors.population_theta, SPECS)
    for name, policy in snap.policies.items():
        validate_game_strategy(GAME, policy, infosets=SPECS)
        assert 0.0 <= snap.alphas[name] <= 1.0
    budget = priors.cfr_exploitability + adapter.settings.budget
    for name in ("br_map", "br_confidence"):
        assert snap.responses[name].budget.exploitability <= budget + 1e-12
    if not snap.gate["open"]:
        assert snap.policies["gated"] == cfr
    records = [
        observer_record(GAME, play_hand(GAME, cfr, UNIFORM, h, 2), h % 2, h) for h in range(15)
    ]
    a, b = adapter.snapshot(records), adapter.snapshot(records)
    assert (
        a.policies == b.policies
        and a.gate["candidates"]["budget_max"]["gains"]
        == b.gate["candidates"]["budget_max"]["gains"]
    )
    assert set(a.prior_scores) == {"population:2", "population:8", "cfr:2", "cfr:8"}


def test_confidence_weights_follow_counts(priors: Priors) -> None:
    adapter = Adapter(GAME, SPECS, priors, AdaptationSettings())
    theta = {k: np.full(len(v), 1.0 / len(v)) for k, v in priors.cfr_theta.items()}
    counts = {k: np.zeros(len(v)) for k, v in theta.items()}
    counts["K|"] = np.array([30.0, 0.0])
    conf = adapter.confidence_theta(theta, counts, 10.0)
    assert conf["Q|"] == pytest.approx(priors.cfr_theta["Q|"])  # no data: CFR
    assert conf["K|"] == pytest.approx(0.75 * theta["K|"] + 0.25 * priors.cfr_theta["K|"])


def test_change_reset_keeps_model_and_data_consistent(priors: Priors) -> None:
    def rec(h: int, action: int) -> ObserverHandRecord:
        pub = PublicHandRecord(h, 0, (PublicDecision("R1:", (CALL, RAISE), action),), None, True)
        return ObserverHandRecord(pub, 0, None)

    observer = ChangeAwareObserver(priors.public_prior, retain=10)
    for h in range(120):
        observer.observe(rec(h, CALL if h < 60 else RAISE))
    assert observer.resets and all(r >= 60 for r in observer.resets)
    retained = observer.retained()
    assert len(retained) <= 120 - observer.resets[0] + 10
    fresh = OpponentModel(ModelConfig(16, 64), priors.public_prior)
    for r in retained:
        fresh.observe_hand(r.public)
    assert observer.public_model().predict("R1:", (CALL, RAISE)) == pytest.approx(
        fresh.predict("R1:", (CALL, RAISE))
    )
    off = ChangeAwareObserver(priors.public_prior, detect=False)
    for h in range(30):
        off.observe(rec(h, CALL))
    assert len(off.retained()) == 30 and not off.resets


# -- opponents and the study ---------------------------------------------------------------------


def test_robust_opponents_are_valid_and_behave_as_named() -> None:
    for family in OUT_OF_POPULATION:
        o = make_out_of_population(family, 0, SPECS)
        validate_game_strategy(GAME, o.policies[0], infosets=SPECS)
    trapper = trapper_policy(SPECS, 0.3, 0.9, 0.0)
    assert showdown_equity(0, None) < showdown_equity(4, None)
    assert trapper["J0|"][RAISE] > trapper["K0|"][RAISE]  # bets weak, checks strong
    blind = make_out_of_population("card_blind", 0, SPECS).policies[0]
    assert blind["J0|r"] == blind["K1|r"]
    station = make_out_of_population("station", 0, SPECS).policies[0]
    assert station["J0|r"][FOLD] < 0.05
    for i in (0, 1):
        o = make_bait_switch(i, SPECS)
        assert o.switch_hand == 100 and o.policies[0] != o.policies[1]


def test_world_groups_are_disjoint(cfr: dict[str, dict[int, float]], tmp_path: Path) -> None:
    path = tmp_path / "hero.json"
    save_game_strategy(path, GAME.name, GAME.action_codes, cfr)
    world = build_world(path, per_family=1, oop_per_family=1, switching=1)
    groups = {k: set(v) for k, v in world.groups.items()}
    names = [n for v in world.groups.values() for n in v]
    assert len(names) == len(set(names))
    dev_families = {world.opponents[n].family for n in groups["development"]}
    assert not dev_families & {world.opponents[n].family for n in groups["out_of_population"]}
    assert main(["adaptive-study", "--hero", str(tmp_path / "missing.json")]) == 2
