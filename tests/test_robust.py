"""M7A: compiled trees, restricted CFR, RNR, DBR, selection under exact budgets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pokermind.adaptive.reconstruction import evidence, observer_record
from pokermind.adaptive.response import exploitability, hero_best_response, hero_value
from pokermind.evaluation.game_exact import information_sets, tree_counts, validate_game_strategy
from pokermind.experiments.robust_study import Candidate, _select
from pokermind.games.leduc import LeducGame
from pokermind.learning.game_cfr import GameCFRSolver
from pokermind.opponents.play import play_hand
from pokermind.robust.modified_game import RNRGame, hero_table
from pokermind.robust.responses import (
    DBRModel,
    confidence,
    dbr_model,
    dbr_strategy,
    rnr_strategy,
)
from pokermind.robust.solver import Copy, RestrictedSolver
from pokermind.robust.tree import CompiledTree

GAME = LeducGame()
SPECS = information_sets(GAME)
TREE = CompiledTree(GAME)
ACTIONS = {k: s.actions for k, s in SPECS.items()}


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


# -- compiled tree and solver ----------------------------------------------------------------


def test_compiled_tree_shape_and_round_trip() -> None:
    assert TREE.num_nodes == 157 + 3_780 + 5_520
    assert TREE.num_isets == 936 and TREE.num_slots == 2_184
    root = TREE.levels[0]
    assert root.chance_prob.sum() == pytest.approx(1.0)
    vec = TREE.to_vector(MODEL)
    back = TREE.to_table(vec)
    assert all(back[k][a] == MODEL[k][a] for k in MODEL for a in MODEL[k])
    uniform = TREE.to_table(TREE.normalize(np.zeros(TREE.num_slots)))
    assert uniform["K0|r"] == pytest.approx({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})


def test_vanilla_mode_matches_the_m3_solver() -> None:
    zeros = np.zeros(TREE.num_isets)
    solver = RestrictedSolver(
        TREE,
        0,
        [Copy(1.0, zeros)],
        TREE.to_vector(MODEL),
        plus=False,
        alternating=False,
        linear=False,
    )
    solver.iterate(6)
    ours = TREE.to_table(solver.result().average)
    ref = GameCFRSolver(GAME)
    ref.iterate(6)
    avg = ref.average_strategy()
    assert max(abs(ours[k][a] - avg[k][a]) for k in avg for a in avg[k]) < 1e-12


def test_solver_rejects_bad_inputs() -> None:
    zeros = np.zeros(TREE.num_isets)
    with pytest.raises(ValueError):
        RestrictedSolver(TREE, 2, [Copy(1.0, zeros)], TREE.to_vector(MODEL))
    with pytest.raises(ValueError):
        RestrictedSolver(TREE, 0, [Copy(0.5, zeros)], TREE.to_vector(MODEL))
    with pytest.raises(ValueError):
        RestrictedSolver(TREE, 0, [Copy(1.0, zeros + 2)], TREE.to_vector(MODEL))


# -- the modified game ---------------------------------------------------------------------------


def test_rnr_modified_game_rules_and_chance() -> None:
    game = RNRGame(MODEL, 0.3)
    root = game.initial_state()
    outcomes = dict(root.chance_outcomes())
    assert outcomes == pytest.approx({0: 0.35, 1: 0.15, 2: 0.35, 3: 0.15})
    counts = tree_counts(game)
    # 4 copies of Leduc; restricted opponent decisions (1,890 per seat) become chance nodes
    assert counts.terminal == 4 * 5_520
    assert counts.decision == 2 * 3_780 + 2 * 1_890
    assert counts.chance == 1 + 4 * 157 + 2 * 1_890
    # hero's key hides the coin; restricted opponent nodes follow the model
    free = root.apply(0).apply(5).apply(0)  # hero seat 0, free; hero K1, opponent J0
    restricted = root.apply(1).apply(5).apply(0)
    assert free.information_state_key(0) == restricted.information_state_key(0) == "H|K1|"
    after = restricted.apply(1)  # hero checks; the restricted opponent is chance
    assert after.is_chance_node
    assert dict(after.chance_outcomes()) == pytest.approx(MODEL["J0|c"])
    assert root.apply(0).apply(5).apply(0).apply(1).information_state_key(1) == "O|J0|c"
    assert dict(RNRGame(MODEL, 0.0).initial_state().chance_outcomes()) == {0: 0.5, 2: 0.5}
    with pytest.raises(ValueError):
        RNRGame(MODEL, 1.5)


def test_vectorized_rnr_matches_cfr_on_the_explicit_game() -> None:
    p = 0.3
    ref = GameCFRSolver(RNRGame(MODEL, p))
    ref.iterate(5)
    hero = hero_table(ref.average_strategy())
    out = np.zeros(TREE.num_slots)
    copies = [Copy(p, np.ones(TREE.num_isets)), Copy(1 - p, np.zeros(TREE.num_isets))]
    for r in (0, 1):
        s = RestrictedSolver(
            TREE, r, copies, TREE.to_vector(MODEL), plus=False, alternating=False, linear=False
        )
        s.iterate(5)
        mask = TREE.slot_seat == r
        out[mask] = s.result().average[mask]
    ours = TREE.to_table(out)
    assert max(abs(ours[k][a] - hero[k][a]) for k in hero for a in hero[k]) < 1e-12


# -- RNR and DBR ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nash() -> dict[str, dict[int, float]]:
    return rnr_strategy(TREE, MODEL, 0.0, 400)


def test_rnr_limits(nash: dict[str, dict[int, float]]) -> None:
    validate_game_strategy(GAME, nash, infosets=SPECS)
    assert exploitability(GAME, nash, SPECS) < 0.006  # approximately Nash
    full = rnr_strategy(TREE, MODEL, 1.0, 400)
    validate_game_strategy(GAME, full, infosets=SPECS)
    br_value = hero_value(GAME, hero_best_response(GAME, MODEL, SPECS), MODEL, SPECS)
    assert hero_value(GAME, full, MODEL, SPECS) == pytest.approx(br_value, abs=0.01)
    mid = rnr_strategy(TREE, MODEL, 0.3, 400)
    values = [hero_value(GAME, s, MODEL, SPECS) for s in (nash, mid, full)]
    expl = [exploitability(GAME, s, SPECS) for s in (nash, mid, full)]
    assert values[0] < values[1] < values[2]
    assert expl[0] < expl[1] < expl[2]
    assert rnr_strategy(TREE, MODEL, 0.3, 50) == rnr_strategy(TREE, MODEL, 0.3, 50)  # deterministic


def test_dbr_confidence_and_model() -> None:
    assert confidence(0.0, 0.8, 10.0) == 0.0
    assert confidence(5.0, 0.8, 10.0) == pytest.approx(0.4)
    assert confidence(50.0, 0.8, 10.0) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        confidence(1.0, 1.5, 10.0)
    from pokermind.adaptive.reconstruction import canonical_actions

    counts = {k: np.zeros(len(a)) for k, a in canonical_actions(SPECS).items()}
    counts["K|r"] = np.array([1.0, 2.0, 1.0])
    model = dbr_model(counts, ACTIONS)
    assert model.counts["K0|r"] == model.counts["K1|r"] == 4.0  # suits share counts
    assert model.frequencies["K1|r"] == pytest.approx({0: 0.25, 1: 0.5, 2: 0.25})
    assert model.counts["Q0|r"] == 0.0 and model.frequencies["Q0|r"] == pytest.approx(
        {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    )


def test_dbr_limits(nash: dict[str, dict[int, float]]) -> None:
    observed = DBRModel(MODEL, dict.fromkeys(MODEL, 100.0))
    unobserved = DBRModel(MODEL, dict.fromkeys(MODEL, 0.0))
    # no confidence anywhere: plain CFR+, identical to RNR with p = 0
    assert dbr_strategy(TREE, unobserved, 1.0, 10.0, 400) == nash
    assert dbr_strategy(TREE, observed, 0.0, 10.0, 400) == nash
    # full confidence everywhere: the same game as RNR with p = 1
    assert dbr_strategy(TREE, observed, 1.0, 10.0, 200) == rnr_strategy(TREE, MODEL, 1.0, 200)
    partial = dbr_strategy(TREE, observed, 0.5, 10.0, 200)
    validate_game_strategy(GAME, partial, infosets=SPECS)


def test_selection_respects_the_exact_budget() -> None:
    def cand(p: float, e: float, v: float) -> Candidate:
        return Candidate("rnr", p, {}, e, v)

    cands = [cand(0.0, 0.001, 0.0), cand(0.2, 0.02, 0.1), cand(0.4, 0.06, 0.3), cand(1.0, 1.0, 0.9)]
    assert _select(cands, 0.05).parameter == 0.2  # type: ignore[union-attr]
    assert _select(cands, 0.06).parameter == 0.4  # type: ignore[union-attr]
    assert _select(cands, 0.0005) is None
    assert math.isclose(_select(cands, 10.0).model_value, 0.9)  # type: ignore[union-attr]


def test_inputs_are_public_only() -> None:
    """Swapping the opponent's hidden card in a folded hand leaves the evidence, and so
    every reconstruction-driven strategy, unchanged."""
    for hand in range(100):
        moves = play_hand(GAME, MODEL, MODEL, hand, stream=9)
        rec = observer_record(GAME, moves, hand % 2, hand)
        if rec.public.showdown_card is not None:
            continue
        state = GAME.state_from_moves(moves)
        used = {moves[0], moves[1], *(() if state.public is None else (state.public,))}
        for card in set(range(6)) - used:
            alt = list(moves)
            alt[hand % 2] = card
            assert evidence(observer_record(GAME, tuple(alt), hand % 2, hand)) == evidence(rec)
