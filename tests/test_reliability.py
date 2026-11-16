"""M7B: batched RNR, exact batched evaluation, RNR-game gap, selection, calibration
helpers, tempering, the bait-and-switch population and the research configuration."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from pokermind.adaptive.reconstruction import CardConditionalModel, HandEvidence
from pokermind.adaptive.response import exploitability, hero_value
from pokermind.adaptive.robust_opponents import BAIT_STYLES, make_bait_switch_population
from pokermind.evaluation.game_exact import (
    exploitability_report,
    information_sets,
    validate_game_strategy,
)
from pokermind.experiments import reliability_study as rs
from pokermind.experiments.adaptive_study import observe
from pokermind.experiments.robust_study import _grid, _select
from pokermind.games.leduc import LeducGame
from pokermind.research import (
    DEFAULT_CONFIG_PATH,
    ResearchConfig,
    SelectionConfig,
    load_research_config,
    save_research_config,
)
from pokermind.robust.batch import BatchedRNR, TreeEvaluator, rnr_strategies
from pokermind.robust.modified_game import RNRGame, rnr_bounds
from pokermind.robust.responses import rnr_strategy
from pokermind.robust.selection import SelectionRule, Selector, stop_rule
from pokermind.robust.tree import CompiledTree

GAME = LeducGame()
SPECS = information_sets(GAME)
TREE = CompiledTree(GAME)
EVAL = TreeEvaluator(TREE)


def random_policy(seed: int) -> dict[str, dict[int, float]]:
    rng = np.random.default_rng(seed)
    return {
        k: {
            a: float(x)
            for a, x in zip(s.actions, rng.dirichlet(np.ones(len(s.actions))), strict=True)
        }
        for k, s in SPECS.items()
    }


MODEL = random_policy(0)
OTHER = random_policy(1)


# -- batched solver --------------------------------------------------------------------------


def test_batched_rnr_is_bit_identical_to_m7a() -> None:
    ps = (0.0, 0.3, 1.0)
    batched = rnr_strategies(TREE, MODEL, ps, 40)
    for p, table in zip(ps, batched, strict=True):
        assert table == rnr_strategy(TREE, MODEL, p, 40)  # exact equality, not approximate


def test_batched_rnr_validates_and_warm_starts() -> None:
    with pytest.raises(ValueError):
        BatchedRNR(TREE, TREE.to_vector(MODEL), [])
    with pytest.raises(ValueError):
        BatchedRNR(TREE, TREE.to_vector(MODEL), [1.2])
    cold = BatchedRNR(TREE, TREE.to_vector(MODEL), [0.2, 0.6])
    cold.iterate(30)
    warm = BatchedRNR(TREE, TREE.to_vector(OTHER), [0.25, 0.6], start=cold)
    # nearest p of the start problem: regrets copied, average fresh
    assert np.array_equal(warm.regret, cold.regret)
    assert not warm.avg.any()
    warm.iterate(5)
    res = warm.result()
    validate_game_strategy(GAME, TREE.to_table(res.hero[0]), infosets=SPECS)


# -- RNR-game gap -----------------------------------------------------------------------------


def test_rnr_gap_equals_nash_conv_of_the_explicit_game() -> None:
    p = 0.3
    solver = BatchedRNR(TREE, TREE.to_vector(MODEL), [p])
    solver.iterate(20)
    res = solver.result()
    hero, free = TREE.to_table(res.hero[0]), TREE.to_table(res.opponent[0])
    bounds = rnr_bounds(GAME, SPECS, hero, free, MODEL, p)
    game = RNRGame(MODEL, p)
    specs = information_sets(game)
    profile = {("H|" + k): v for k, v in hero.items()} | {("O|" + k): v for k, v in free.items()}
    report = exploitability_report(game, {k: profile[k] for k in specs}, infosets=specs)
    assert bounds.gap == pytest.approx(report.nash_conv, abs=1e-12)
    assert bounds.lower <= bounds.upper


def test_rnr_gap_shrinks_with_iterations() -> None:
    p = 0.5
    solver = BatchedRNR(TREE, TREE.to_vector(MODEL), [p])
    gaps = []
    for its in (10, 90):
        solver.iterate(its)
        res = solver.result()
        hero, free = TREE.to_table(res.hero[0]), TREE.to_table(res.opponent[0])
        gaps.append(rnr_bounds(GAME, SPECS, hero, free, MODEL, p).gap)
    assert 0 <= gaps[1] < gaps[0]


# -- batched exact evaluation ----------------------------------------------------------------


def test_tree_evaluator_matches_the_m3_evaluator() -> None:
    policies = [MODEL, OTHER, random_policy(2)]
    mat = np.array([TREE.to_vector(x) for x in policies])
    expl = EVAL.exploitability(mat)
    values = EVAL.hero_values(mat, mat)
    for i, x in enumerate(policies):
        assert expl[i] == pytest.approx(exploitability(GAME, x, SPECS), abs=1e-12)
        for j, y in enumerate(policies):
            assert values[i, j] == pytest.approx(hero_value(GAME, x, y, SPECS), abs=1e-12)


def test_tree_evaluator_on_pure_strategies_and_reach() -> None:
    # pure strategies create exact ties in best responses; the value must not depend on them
    pure = {k: {a: float(a == s.actions[0]) for a in s.actions} for k, s in SPECS.items()}
    vec = TREE.to_vector(pure)[None, :]
    assert EVAL.exploitability(vec)[0] == pytest.approx(
        exploitability(GAME, pure, SPECS), abs=1e-12
    )
    w = EVAL.hero_reach_weights(TREE.to_vector(MODEL), TREE.to_vector(OTHER))
    seat = np.array(TREE.iset_seat)
    # hero's first decision as seat 0 (after the deal) is reached with probability 1 in total
    roots = [i for i, k in enumerate(TREE.keys) if k.count("|") == 1 and k.endswith("|")]
    assert sum(w[i] for i in roots if seat[i] == 0) == pytest.approx(0.5)


# -- selection --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def selector() -> Selector:
    cfr = rnr_strategy(TREE, MODEL, 0.0, 300)
    return Selector(GAME, SPECS, TREE, cfr, exploitability(GAME, cfr, SPECS))


def test_point_rule_reproduces_m7a_selection(selector: Selector) -> None:
    from pokermind.experiments.adaptive_study import World

    world = World(GAME, SPECS, selector.cfr, None, {}, {}, {})  # type: ignore[arg-type]
    stop = selector.cfr_exploitability + 0.25
    ps = (0.0, 0.2, 0.5, 1.0)
    cands, _ = selector.candidates(MODEL, stop, ps, iterations=60)
    ref, _ = _grid(world, "rnr", ps, lambda p: rnr_strategy(TREE, MODEL, p, 60), MODEL, stop)
    for d in (0.0, 0.05, 0.25):
        mine = selector.select(cands, selector.cfr_exploitability + d, SelectionRule())
        theirs = _select(ref, selector.cfr_exploitability + d)
        assert mine.p == (None if theirs is None else theirs.parameter)
        if theirs is not None:
            assert mine.exploitability == pytest.approx(theirs.exploitability, abs=1e-15)
            assert mine.certificates >= 1


def test_lcb_rule_falls_back_and_needs_samples(selector: Selector) -> None:
    cands, _ = selector.candidates(MODEL, selector.cfr_exploitability + 0.25, (0.0, 0.5), 40)
    with pytest.raises(ValueError):
        selector.select(cands, 1.0, SelectionRule("lcb", 1.0))
    selector.add_samples(cands, [MODEL, OTHER, random_policy(3)])
    assert cands.sample_gain is not None and cands.sample_gain.shape == (len(cands.ps), 3)
    # an enormous z makes every lower bound negative: play the CFR baseline
    s = selector.select(cands, 10.0, SelectionRule("lcb", z=1e6))
    assert s.index is None and selector.policy(cands, s) is selector.cfr
    with pytest.raises(ValueError):
        SelectionRule("other")


def test_selection_never_exceeds_the_budget(selector: Selector) -> None:
    cands, _ = selector.candidates(MODEL, selector.cfr_exploitability + 0.25, iterations=40)
    for d in (0.0, 0.02, 0.1):
        budget = selector.cfr_exploitability + d
        s = selector.select(cands, budget, SelectionRule())
        policy = selector.policy(cands, s)
        assert exploitability(GAME, policy, SPECS) <= budget + 1e-12


def test_stop_rule() -> None:
    assert stop_rule(np.array([0.0, 0.1, 0.5, 0.6, 0.2]), 0.3) == 4
    assert stop_rule(np.array([0.0, 0.5, 0.1, 0.6]), 0.3) == 4
    assert stop_rule(np.array([0.0, 0.1]), 0.3) == 2


def test_acting_policy_ignores_the_true_opponent() -> None:
    """The decision (chosen p) for one snapshot is the same whatever truth evaluates it."""
    hero = Path("runs/m4-leduc-teacher/strategies/strategy_100000.json")
    if not hero.is_file():
        pytest.skip("M4 teacher strategy not available (git-ignored run output)")
    rs._init(hero, 2)
    world, _, sel, adapter = rs._ctx()
    opponent = world.opponents[world.groups["heldout"][0]]
    rec = rs.reconstruct(world, adapter, observe(world, opponent, 0, 20))
    config = rs.ReliabilityConfig(Path("x"), hands=(20,), deltas=(0.1,))
    research = ResearchConfig(
        selection=SelectionConfig(rule="lcb", z=1.0, inflation=1.5),
        response=dataclasses.replace(ResearchConfig().response, iterations=30),
    )
    research = dataclasses.replace(
        research,
        reconstruction=dataclasses.replace(research.reconstruction, posterior_samples=4),
    )
    rows = rs._snapshot_decisions(
        world, sel, rec, research, config, [MODEL, OTHER], (1, 2), with_mix=False
    )
    for pol in ("point", "frozen"):
        assert rows[0][pol]["0.1"]["p"] == rows[1][pol]["0.1"]["p"]


# -- tempering --------------------------------------------------------------------------------


def test_tempering_widens_the_posterior() -> None:
    prior = {"K|": np.array([0.5, 0.5])}
    model = CardConditionalModel(prior, 2.0)
    # 30 shown-down hands (card 4 is a king) that always took action 1 at "K|"
    hands = [HandEvidence((4,), (("R1:", (1, 2), 1),))] * 30

    def spread(tau: float) -> float:
        rng = np.random.default_rng(0)
        draws = model.posterior_samples(hands, 200, rng, temperature=tau)
        return float(np.std([d["K|"][0] for d in draws]))

    assert spread(0.25) > spread(1.0)
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    a = model.posterior_samples(hands, 3, rng_a)
    b = model.posterior_samples(hands, 3, rng_b, temperature=1.0)
    assert all(np.array_equal(x["K|"], y["K|"]) for x, y in zip(a, b, strict=True))
    with pytest.raises(ValueError):
        model.posterior_samples(hands, 3, np.random.default_rng(0), temperature=0.0)


# -- calibration helpers ------------------------------------------------------------------------


def _synthetic(sd_scale: float, n: int = 400) -> list[dict[str, object]]:
    """Rows whose sample gains are N(truth + noise, 1) draws, noise ~ N(0, sd_scale²)."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        truth = float(rng.normal())
        center = truth + float(rng.normal(scale=sd_scale))
        samples = {
            str(t): [[0.0] * 400, (center + rng.normal(size=400)).tolist()] for t in rs.TEMPERATURES
        }
        rows.append(
            {
                "hands": 20,
                "ps": [0.0, 0.5],
                "exploitability": [0.0, 0.1],
                "model_value": [0.0, center],
                "true_gain": [0.0, truth],
                "sample_gain": samples,
                "parameter_coverage": {},
                "timings": {},
                "i": i,
            }
        )
    return [{"opponent": "o", "rows": rows}]


def test_gain_coverage_detects_over_confidence() -> None:
    calibrated = rs.gain_coverage(_synthetic(1.0))["1.0"]
    over = rs.gain_coverage(_synthetic(2.0))["1.0"]
    cov = calibrated["by_inflation"]["1.0"]["two_sided"]["0.9"]
    assert cov == pytest.approx(0.9, abs=0.05)
    assert over["by_inflation"]["1.0"]["two_sided"]["0.9"] < 0.75
    assert over["empirical_inflation_90"] == pytest.approx(2.0, rel=0.2)
    choice = rs.choose_calibration(rs.gain_coverage(_synthetic(2.0)))
    assert choice["inflation"] in (1.5, 2.0, 2.5)


def test_choose_rule_prefers_fewer_losses_near_the_best() -> None:
    def sim(mean: float, harm: float) -> dict[str, object]:
        return {"0.1": {"gain": {"mean": mean}, "harm_rate": harm}}

    simulated = {"point": sim(1.0, 0.3), "a": sim(0.95, 0.1), "b": sim(0.5, 0.0)}
    assert rs.choose_rule(simulated) == "a"  # b is below 90 % of the best mean
    assert rs.choose_rule({"point": sim(1.0, 0.3)}) == "point"


# -- opponents and configuration ------------------------------------------------------------


def test_bait_switch_population() -> None:
    pop = make_bait_switch_population(24, SPECS)
    assert len(pop) == 24 and len({o.name for o in pop}) == 24
    pairs = {o.phase_families for o in pop}
    assert len(pairs) == 24 and all(a != b and a in BAIT_STYLES for a, b in pairs)
    assert {o.switch_hand for o in pop} <= {50, 100, 200}
    for o in pop[:6]:
        for policy in o.policies:
            validate_game_strategy(GAME, policy, infosets=SPECS)
    assert [o.name for o in make_bait_switch_population(24, SPECS)] == [o.name for o in pop]


def test_research_config_round_trip_and_digest(tmp_path: Path) -> None:
    config = ResearchConfig(selection=SelectionConfig(rule="lcb", z=1.28, inflation=2.0))
    path = tmp_path / "c.json"
    save_research_config(config, path)
    loaded = load_research_config(path)
    assert loaded == config and loaded.digest() == config.digest()
    doc = json.loads(path.read_text())
    doc["selection"]["z"] = 0.5  # edited without re-freezing
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="digest"):
        load_research_config(path)
    with pytest.raises(ValueError):
        ResearchConfig(selection=SelectionConfig(rule="other"))
    with pytest.raises(ValueError):
        ResearchConfig(selection=SelectionConfig(inflation=0.0))


def test_committed_research_config_is_valid() -> None:
    if not DEFAULT_CONFIG_PATH.is_file():
        pytest.skip("research configuration not frozen yet")
    config = load_research_config(DEFAULT_CONFIG_PATH)
    assert config.version.count(".") == 2
    assert config.provenance.get("evidence") == "development opponents only"
