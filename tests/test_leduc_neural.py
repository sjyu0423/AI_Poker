"""M4: Leduc encoding, dataset splits, masked policies, training and evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from pokermind.cli import main
from pokermind.evaluation.game_exact import exploitability_report, information_sets
from pokermind.experiments.leduc_neural_study import (
    baseline_rows,
    distribution_metrics,
    model_rows,
    table_from_rows,
)
from pokermind.games.leduc import CALL, FOLD, RAISE, LeducGame, LeducObservation, LeducState
from pokermind.learning.game_cfr import GameCFRSolver
from pokermind.neural.game_model import (
    GamePolicyConfig,
    GamePolicyNetwork,
    legal_probabilities,
    load_checkpoint,
    save_checkpoint,
)
from pokermind.neural.game_training import Arrays, train_game_policy
from pokermind.neural.leduc_agent import LeducNeuralAgent
from pokermind.neural.leduc_dataset import (
    HELD_OUT_PUBLIC_COMBOS,
    SPLIT_NAMES,
    LeducDataset,
    Split,
    build_leduc_dataset,
    make_split,
    split_coverage,
)
from pokermind.neural.leduc_encoding import (
    FEATURE_NAMES,
    NUM_ACTIONS,
    NUM_FEATURES,
    canonical_key,
    encode_observation,
    legal_action_mask,
    max_contribution,
)
from pokermind.neural.training import TrainingConfig, state_dict_digest
from pokermind.strategy import save_game_strategy

GAME = LeducGame()


def decision_states() -> Iterator[tuple[tuple[int, ...], LeducState]]:
    """Every decision node with the moves that reach it."""

    def walk(
        state: LeducState, moves: tuple[int, ...]
    ) -> Iterator[tuple[tuple[int, ...], LeducState]]:
        if state.is_terminal:
            return
        if state.is_chance_node:
            for card, _ in state.chance_outcomes():
                yield from walk(state.apply(card), (*moves, card))
            return
        yield moves, state
        for action in state.legal_actions():
            yield from walk(state.apply(action), (*moves, action))

    yield from walk(GAME.initial_state(), ())


@pytest.fixture(scope="module")
def teacher() -> dict[str, dict[int, float]]:
    solver = GameCFRSolver(GAME)
    solver.iterate(20)
    return solver.average_strategy()


@pytest.fixture(scope="module")
def dataset(teacher: dict[str, dict[int, float]]) -> LeducDataset:
    return build_leduc_dataset(teacher, {"source": "test"})


def arrays(ds: LeducDataset, idx: tuple[int, ...]) -> Arrays:
    return Arrays(ds.features(idx), ds.masks(idx), ds.targets(idx))


# -- encoding --------------------------------------------------------------------------------


def test_feature_layout() -> None:
    assert NUM_FEATURES == len(FEATURE_NAMES) == 30
    assert NUM_ACTIONS == 3
    assert max_contribution(GAME.config) == 13


def test_known_encoding() -> None:
    # K1 vs J0, round 1 "r r c", public J1, round 2 "r": player 1 to act, facing 4 more.
    state = GAME.state_from_moves((5, 0, RAISE, RAISE, CALL, 1, RAISE))
    x = encode_observation(state.observation(1))
    on = {FEATURE_NAMES[i] for i in np.flatnonzero(x)}
    assert on == {
        "own=J", "public=J", "pair", "round=2",
        "r1.a0=r", "r1.a1=r", "r1.a2=c", "r2.a0=r",
        "acting_seat=1", "facing_bet", "raises_this_round",
        "own_contribution", "opponent_contribution",
    }  # fmt: skip
    assert x[FEATURE_NAMES.index("raises_this_round")] == pytest.approx(0.5)
    assert x[FEATURE_NAMES.index("own_contribution")] == pytest.approx(5 / 13)
    assert x[FEATURE_NAMES.index("opponent_contribution")] == pytest.approx(9 / 13)


def test_encoding_agrees_with_engine_at_every_decision_node() -> None:
    names = FEATURE_NAMES
    top = max_contribution(GAME.config)
    count = 0
    for _, state in decision_states():
        seat = state.current_player
        x = encode_observation(state.observation(seat))
        contrib = state.contributions
        legal = state.legal_actions()
        assert x[names.index("acting_seat=1")] == seat
        assert x[names.index("round=2")] == state.round_index
        assert x[names.index("own_contribution")] == pytest.approx(contrib[seat] / top)
        assert x[names.index("opponent_contribution")] == pytest.approx(contrib[1 - seat] / top)
        assert bool(x[names.index("facing_bet")]) == (FOLD in legal)
        assert (x[names.index("raises_this_round")] < 1.0) == (RAISE in legal)
        count += 1
    assert count == 3_780


def test_encoding_never_depends_on_the_opponent_card() -> None:
    """Replace the opponent's card by every other unseen card at every decision node."""
    checked = 0
    for moves, state in decision_states():
        seat = state.current_player
        x = encode_observation(state.observation(seat))
        opp_index = 1 - seat  # moves[0] is P0's card, moves[1] P1's
        used = {moves[0], moves[1], *(() if state.public is None else (state.public,))}
        for card in sorted(set(range(6)) - used):
            alt = list(moves)
            alt[opp_index] = card
            other = GAME.state_from_moves(tuple(alt))
            assert np.array_equal(encode_observation(other.observation(seat)), x)
            checked += 1
    assert checked > 3_780


def test_encoder_rejects_full_states_and_impossible_observations() -> None:
    state = GAME.state_from_moves((5, 0))
    with pytest.raises(TypeError):
        encode_observation(state)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not the one to act"):
        encode_observation(LeducObservation(1, 5, None, ((),)))
    with pytest.raises(ValueError, match="public card"):
        encode_observation(LeducObservation(0, 5, None, ((CALL, CALL), ())))
    with pytest.raises(ValueError, match="fold"):
        encode_observation(LeducObservation(0, 5, None, ((RAISE, FOLD),)))


def test_suits_are_dropped_and_canonical_states_are_distinguished(dataset: LeducDataset) -> None:
    k0 = GAME.state_from_moves((4, 0)).observation(0)
    k1 = GAME.state_from_moves((5, 0)).observation(0)
    assert np.array_equal(encode_observation(k0), encode_observation(k1))
    assert canonical_key(k0) == canonical_key(k1) == "K|"
    rows = {x.tobytes() for x in dataset.features()}
    assert len(rows) == len(dataset) == 288  # injective on canonical states


def test_legal_action_mask() -> None:
    assert legal_action_mask((CALL, RAISE)).tolist() == [False, True, True]
    assert legal_action_mask((FOLD, CALL)).tolist() == [True, True, False]
    with pytest.raises(ValueError):
        legal_action_mask(())
    with pytest.raises(ValueError):
        legal_action_mask((3,))


# -- dataset and splits ----------------------------------------------------------------------


def test_dataset_covers_every_information_set_once(dataset: LeducDataset) -> None:
    members = [m for e in dataset.examples for m in e.members]
    assert len(members) == len(set(members)) == 936
    assert set(members) == set(information_sets(GAME))
    targets, masks = dataset.targets(), dataset.masks()
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert (targets[~masks] == 0).all()
    assert dataset.max_symmetry_deviation == 0.0


def test_suit_asymmetric_teacher_is_rejected(teacher: dict[str, dict[int, float]]) -> None:
    broken = {k: dict(v) for k, v in teacher.items()}
    broken["K0|"] = {CALL: 0.9, RAISE: 0.1}
    broken["K1|"] = {CALL: 0.1, RAISE: 0.9}
    with pytest.raises(ValueError, match="suit-symmetric"):
        build_leduc_dataset(broken)


@pytest.mark.parametrize("name", SPLIT_NAMES)
def test_splits_partition_and_never_share_feature_vectors(dataset: LeducDataset, name: str) -> None:
    split = make_split(dataset, name)
    assert make_split(dataset, name) == split  # deterministic
    parts = [set(split.train), set(split.validation), set(split.test)]
    assert sum(len(p) for p in parts) == len(dataset)
    assert set().union(*parts) == set(range(len(dataset)))
    feats = dataset.features()
    fit = {feats[i].tobytes() for i in (*split.train, *split.validation)}
    assert not any(feats[i].tobytes() in fit for i in split.test)
    fit_keys = {m for i in (*split.train, *split.validation) for m in dataset.examples[i].members}
    test_keys = {m for i in split.test for m in dataset.examples[i].members}
    assert not fit_keys & test_keys
    coverage = split_coverage(dataset, split)
    assert coverage["overlap"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
        "test_feature_vectors_in_fit": 0,
    }


def test_split_definitions(dataset: LeducDataset) -> None:
    ex = dataset.examples
    public = make_split(dataset, "public")
    assert {(ex[i].own_rank, ex[i].public_rank) for i in public.test} == set(HELD_OUT_PUBLIC_COMBOS)
    assert len(public.test) == 60
    assert all(
        (ex[i].own_rank, ex[i].public_rank) not in HELD_OUT_PUBLIC_COMBOS
        for i in (*public.train, *public.validation)
    )
    history = make_split(dataset, "history")
    assert len(history.test) == 3 + 9 * 6
    assert all(ex[i].histories[0].startswith("rr") for i in history.test)
    random = make_split(dataset, "random")
    assert len(random.test) == round(0.2 * 288)
    assert make_split(dataset, "random", seed=1).test != random.test
    full = make_split(dataset, "full")
    assert len(full.train) == 288 and not full.test and not full.validation
    with pytest.raises(ValueError):
        make_split(dataset, "nope")


def test_split_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Split("bad", "", (0, 1), (), (1,))
    with pytest.raises(ValueError, match="duplicate"):
        Split("bad", "", (0, 0), (), ())


# -- model, masking, training ----------------------------------------------------------------


def test_masked_probabilities_are_normalized_and_zero_on_illegal(dataset: LeducDataset) -> None:
    for hidden in [(), (16,)]:
        model = GamePolicyNetwork(GamePolicyConfig(NUM_FEATURES, NUM_ACTIONS, hidden, 3))
        probs = legal_probabilities(model, dataset.features(), dataset.masks())
        assert np.abs(probs.sum(axis=1) - 1.0).max() < 1e-12
        assert (probs[~dataset.masks()] == 0.0).all()
        assert (probs[dataset.masks()] > 0.0).all()
    with pytest.raises(ValueError):
        legal_probabilities(model, dataset.features()[:1], np.zeros((1, 3), dtype=bool))


def test_training_is_reproducible(dataset: LeducDataset) -> None:
    split = make_split(dataset, "random")
    config = GamePolicyConfig(NUM_FEATURES, NUM_ACTIONS, (16,), 7)
    training = TrainingConfig(epochs=60, learning_rate=0.01, lr_schedule="cosine", keep_best=True)
    runs = [
        train_game_policy(
            arrays(dataset, split.train), config, training, arrays(dataset, split.validation)
        )
        for _ in range(2)
    ]
    assert state_dict_digest(runs[0][0]) == state_dict_digest(runs[1][0])  # type: ignore[arg-type]
    assert runs[0][1].train_kl == runs[1][1].train_kl


def test_validation_selection_picks_lowest_validation_kl(dataset: LeducDataset) -> None:
    split = make_split(dataset, "history")
    config = GamePolicyConfig(NUM_FEATURES, NUM_ACTIONS, (16,), 0)
    training = TrainingConfig(epochs=300, learning_rate=0.03, lr_schedule="cosine", keep_best=True)
    model, history = train_game_policy(
        arrays(dataset, split.train), config, training, arrays(dataset, split.validation)
    )
    assert history.selection == "validation_kl"
    best = int(np.argmin(history.validation_kl))
    assert history.selected_epoch == history.epoch[best]
    probs = model_rows(model, dataset)[list(split.validation)]
    metrics = distribution_metrics(
        probs, dataset.targets(split.validation), dataset.masks(split.validation)
    )
    assert metrics["kl"] == pytest.approx(history.validation_kl[best], abs=1e-5)


def test_training_reduces_the_loss(dataset: LeducDataset) -> None:
    full = make_split(dataset, "full")
    config = GamePolicyConfig(NUM_FEATURES, NUM_ACTIONS, (32,), 0)
    training = TrainingConfig(epochs=400, learning_rate=0.01, lr_schedule="cosine", keep_best=True)
    _, history = train_game_policy(arrays(dataset, full.train), config, training)
    assert history.selection == "train_kl"
    assert min(history.train_kl) < 0.25 * history.train_kl[0]


def test_arrays_validation(dataset: LeducDataset) -> None:
    targets = dataset.targets((0,)).copy()
    masks = dataset.masks((0,)).copy()
    masks[0] = False
    masks[0, int(np.argmax(targets[0]))] = True
    targets[0] = 0.0
    targets[0, int(np.flatnonzero(~masks[0])[0])] = 1.0
    with pytest.raises(ValueError, match="illegal"):
        Arrays(dataset.features((0,)), masks, targets)


def test_checkpoint_round_trip(dataset: LeducDataset, tmp_path: Path) -> None:
    model = GamePolicyNetwork(GamePolicyConfig(NUM_FEATURES, NUM_ACTIONS, (16,), 5))
    save_checkpoint(tmp_path / "m.pt", model, {"note": "x"})
    loaded, metadata = load_checkpoint(tmp_path / "m.pt")
    assert metadata == {"note": "x"}
    assert np.array_equal(model_rows(model, dataset), model_rows(loaded, dataset))


# -- evaluation ------------------------------------------------------------------------------


def test_distribution_metrics_handle_zeros() -> None:
    target = np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]])
    masks = np.array([[False, True, True], [False, True, True]])
    same = distribution_metrics(target, target, masks)
    assert same["kl"] == 0.0 and same["total_variation"] == 0.0 and same["floored_entries"] == 0
    predicted = np.array([[0.0, 0.0, 1.0], [0.0, 0.5, 0.5]])  # q = 0 where p = 1
    m = distribution_metrics(predicted, target, masks)
    assert np.isfinite(m["kl"]) and m["kl"] == pytest.approx(-np.log(1e-12) / 2)
    assert m["floored_entries"] == 1
    assert m["total_variation"] == pytest.approx(0.5)
    assert m["argmax_agreement"] == 0.5


def test_full_policy_exploitability_of_expanded_tables(
    dataset: LeducDataset, teacher: dict[str, dict[int, float]]
) -> None:
    specs = information_sets(GAME)
    rebuilt = dataset.expand(table_from_rows(dataset, dataset.targets()))
    assert rebuilt.keys() == teacher.keys()
    expected = exploitability_report(GAME, teacher, infosets=specs).exploitability
    assert exploitability_report(GAME, rebuilt, infosets=specs).exploitability == pytest.approx(
        expected, abs=1e-12
    )
    model = GamePolicyNetwork(GamePolicyConfig(NUM_FEATURES, NUM_ACTIONS, (16,), 1))
    strategy = dataset.expand(table_from_rows(dataset, model_rows(model, dataset)))
    report = exploitability_report(GAME, strategy, infosets=specs)  # validates all 936 sets
    assert report.exploitability > expected


def test_baselines_only_change_test_states(dataset: LeducDataset) -> None:
    split = make_split(dataset, "public")
    targets = dataset.targets()
    for name, (rows, _) in baseline_rows(dataset, split).items():
        fitted = list(split.train) + list(split.validation)
        assert np.array_equal(rows[fitted], targets[fitted]), name
        assert np.allclose(rows.sum(axis=1), 1.0)
        assert (rows[~dataset.masks()] == 0).all()
    uniform = baseline_rows(dataset, split)["uniform"][0]
    masks = dataset.masks()
    for i in split.test:
        assert np.allclose(uniform[i][masks[i]], 1 / masks[i].sum())


def test_neural_agent_matches_model_table(dataset: LeducDataset) -> None:
    model = GamePolicyNetwork(GamePolicyConfig(NUM_FEATURES, NUM_ACTIONS, (16,), 2))
    agent = LeducNeuralAgent(model)
    rows = model_rows(model, dataset)
    rng = np.random.default_rng(0)
    for i, e in enumerate(dataset.examples[::17]):
        dist = agent.distribution(e.observation, e.legal_actions)
        for a in e.legal_actions:
            # float32: one row and a batch of 288 use different kernels (~1e-8 apart)
            assert dist[a] == pytest.approx(rows[i * 17, a], abs=1e-6)
        assert agent.act(e.observation, e.legal_actions, rng) in e.legal_actions
    with pytest.raises(TypeError):
        agent.act(GAME.state_from_moves((5, 0)), (CALL, RAISE), rng)  # type: ignore[arg-type]


def test_cli_leduc_neural_smoke(
    tmp_path: Path, teacher: dict[str, dict[int, float]], capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "teacher.json"
    save_game_strategy(path, GAME.name, GAME.action_codes, teacher)
    out = tmp_path / "run"
    code = main(
        [
            "leduc-neural", "--teacher", str(path), "--seeds", "1", "--epochs", "20",
            "--splits", "full", "public", "--simulation-hands", "200",
            "--benchmark-number", "50", "--output-dir", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    doc = json.loads((out / "metrics.json").read_text())
    assert doc["dataset"]["canonical_states"] == 288
    assert {r["split"] for r in doc["runs"]} == {"full", "public"}
    assert doc["coverage"]["public"]["overlap"]["test_feature_vectors_in_fit"] == 0
    assert set(doc["baselines"]["public"]) == {"uniform", "nearest", "history_mean"}
    assert "teacher exploitability" in capsys.readouterr().out
    assert main(["leduc-neural", "--teacher", str(tmp_path / "missing.json")]) == 2


def test_torch_not_left_in_deterministic_mode() -> None:
    assert not torch.are_deterministic_algorithms_enabled()
