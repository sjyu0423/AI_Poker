"""M2A neural policies: encoding, masking, training, serialization, agent, evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from pokermind.agents import Agent, RandomAgent, StrategyAgent, ThresholdAgent
from pokermind.cli import main
from pokermind.evaluation import (
    expected_value_seat_averaged,
    exploitability,
    information_sets,
    uniform_strategy,
    validate_strategy,
)
from pokermind.experiments.cfr_experiment import deterministic_agent_strategy
from pokermind.experiments.neural_experiment import (
    NeuralExperimentConfig,
    compare_to_teacher,
    exploitability_attribution,
    run_neural_experiment,
    table_from_dataset,
)
from pokermind.games.kuhn import (
    DECK,
    Action,
    Card,
    InfoState,
    KuhnState,
    all_deals,
    initial_state,
)
from pokermind.neural import (
    ACTIONS,
    FEATURE_NAMES,
    NUM_FEATURES,
    NeuralPolicyAgent,
    PolicyConfig,
    PolicyNetwork,
    TrainingConfig,
    build_teacher_dataset,
    encode_info_state,
    legal_action_mask,
    load_checkpoint,
    load_teacher,
    masked_softmax,
    model_strategy,
    parameter_count,
    save_checkpoint,
    train_policy,
)
from pokermind.neural.dataset import TeacherDataset
from pokermind.neural.model import masked_log_softmax
from pokermind.neural.training import backward, state_dict_digest
from pokermind.simulation import SimulationConfig, run_simulation

X, B, C, F = Action.CHECK, Action.BET, Action.CALL, Action.FOLD
M1_TEACHER_DIGEST = "b8217a0c74657df799530f51315c1f8fa3d4c299a07f196af8474e65ffc0becc"


def walk(state: KuhnState) -> Iterator[KuhnState]:
    yield state
    for action in state.legal_actions():
        yield from walk(state.apply(action))


@pytest.fixture(scope="module")
def teacher() -> TeacherDataset:
    return load_teacher(iterations=50_000)


@pytest.fixture(scope="module")
def trained(teacher: TeacherDataset) -> PolicyNetwork:
    model, _ = train_policy(teacher, PolicyConfig((16,), seed=0), TrainingConfig(1_500, 0.03))
    return model


# -- encoding ------------------------------------------------------------------------------


def test_encoding_layout() -> None:
    assert NUM_FEATURES == 7
    assert FEATURE_NAMES == (
        "card=J",
        "card=Q",
        "card=K",
        "history=-",
        "history=x",
        "history=b",
        "history=xb",
    )
    assert ACTIONS == (X, B, C, F)
    np.testing.assert_array_equal(
        encode_info_state(InfoState(0, Card.Q, (X, B))), [0, 1, 0, 0, 0, 0, 1]
    )
    np.testing.assert_array_equal(
        encode_info_state(InfoState(0, Card.K, ())), [0, 0, 1, 1, 0, 0, 0]
    )
    assert encode_info_state(InfoState(1, Card.J, (B,))).dtype == np.float32


def test_every_information_set_has_a_distinct_one_hot_code() -> None:
    codes = set()
    for key, (info, _) in _info_states().items():
        x = encode_info_state(info)
        assert x[:3].sum() == 1 and x[3:].sum() == 1
        codes.add(tuple(x))
        assert key == info.key
    assert len(codes) == 12


def _info_states() -> dict[str, tuple[InfoState, tuple[Action, ...]]]:
    from pokermind.neural.dataset import decision_info_states

    return decision_info_states()


def test_encoding_rejects_non_info_states_and_terminal_histories() -> None:
    with pytest.raises(TypeError):
        encode_info_state(initial_state((Card.J, Card.Q)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decision history"):
        encode_info_state(InfoState(0, Card.K, (X, X)))


def test_legal_action_mask() -> None:
    np.testing.assert_array_equal(legal_action_mask((X, B)), [True, True, False, False])
    np.testing.assert_array_equal(legal_action_mask((F, C)), [False, False, True, True])
    with pytest.raises(ValueError):
        legal_action_mask(())


# -- hidden information ----------------------------------------------------------------------


def test_features_and_agent_independent_of_opponent_card(trained: PolicyNetwork) -> None:
    agent = NeuralPolicyAgent(trained)
    checked = 0
    for cards in all_deals():
        for state in walk(initial_state(cards)):
            if state.is_terminal:
                continue
            p = state.current_player
            own = state.cards[p]
            views = set()
            for opp in DECK:
                if opp == own:
                    continue
                alt = KuhnState((own, opp) if p == 0 else (opp, own), state.history)
                info = alt.information_state(p)
                rng = np.random.default_rng(9)
                actions = tuple(agent.act(info, alt.legal_actions(), rng) for _ in range(10))
                views.add((tuple(encode_info_state(info)), actions))
            assert len(views) == 1
            checked += 1
    assert checked == 24


def test_dataset_has_no_opponent_card_and_no_duplicates(teacher: TeacherDataset) -> None:
    assert len(teacher.examples) == 12
    assert len(set(teacher.keys)) == 12
    assert len({tuple(e.features) for e in teacher.examples}) == 12
    for e in teacher.examples:
        assert set(type(e).__dataclass_fields__) == {
            "key",
            "info_state",
            "features",
            "mask",
            "target",
        }
        assert type(e.info_state) is InfoState
        assert e.features.shape == (NUM_FEATURES,)


# -- dataset ------------------------------------------------------------------------------------


def test_teacher_reproduces_m1_strategy(teacher: TeacherDataset) -> None:
    assert teacher.teacher["digest"] == M1_TEACHER_DIGEST
    assert teacher.teacher["iterations"] == 50_000
    table = table_from_dataset(teacher)
    validate_strategy(table)
    assert exploitability(table) == pytest.approx(0.000999, abs=1e-6)


def test_targets_are_distributions_over_legal_actions(teacher: TeacherDataset) -> None:
    targets, masks = teacher.targets(), teacher.masks()
    np.testing.assert_allclose(targets.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(targets[~masks] == 0)
    assert np.all(targets >= 0)


def test_dataset_from_file_matches(tmp_path: Path, teacher: TeacherDataset) -> None:
    from pokermind.strategy import save_strategy

    path = tmp_path / "teacher.json"
    save_strategy(path, table_from_dataset(teacher))
    from_file = load_teacher(path)
    assert from_file.teacher["digest"] == teacher.teacher["digest"]
    np.testing.assert_array_equal(from_file.targets(), teacher.targets())


def test_dataset_rejects_invalid_teacher() -> None:
    bad = uniform_strategy()
    bad["K|"] = {X: 0.9, B: 0.0}
    with pytest.raises(ValueError):
        build_teacher_dataset(bad)


# -- model, masking and normalization --------------------------------------------------------


@pytest.mark.parametrize(
    ("hidden", "params"), [((), 7 * 4 + 4), ((16,), 7 * 16 + 16 + 16 * 4 + 4), ((16, 16), 468)]
)
def test_parameter_counts(hidden: tuple[int, ...], params: int) -> None:
    model = PolicyNetwork(PolicyConfig(hidden))
    assert parameter_count(model) == params


def test_masked_softmax_normalizes_and_zeroes_illegal_actions() -> None:
    logits = torch.tensor([[5.0, -2.0, 100.0, 3.0], [0.0, 0.0, -50.0, 50.0]])
    mask = torch.tensor([[True, True, False, False], [False, False, True, True]])
    q = masked_softmax(logits, mask)
    torch.testing.assert_close(q.sum(dim=-1), torch.ones(2))
    assert torch.all(q[~mask] == 0)
    # Only differences between legal logits matter.
    torch.testing.assert_close(q[0, :2], torch.softmax(torch.tensor([5.0, -2.0]), dim=0))
    log_q = masked_log_softmax(logits, mask)
    assert torch.isfinite(log_q).all()  # finite masking value: no 0 * -inf = nan in the loss


def test_masking_validates_its_input() -> None:
    with pytest.raises(TypeError):
        masked_softmax(torch.zeros(1, 4), torch.ones(1, 4))
    with pytest.raises(ValueError):
        masked_softmax(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool))


def test_illegal_logits_receive_no_gradient(teacher: TeacherDataset) -> None:
    model = PolicyNetwork(PolicyConfig((16,)))
    x = torch.from_numpy(teacher.features())
    mask = torch.from_numpy(teacher.masks())
    p = torch.from_numpy(teacher.targets()).float()
    logits = model(x)
    logits.retain_grad()
    loss = -(p * masked_log_softmax(logits, mask)).sum(dim=-1).mean()
    backward(loss)
    assert logits.grad is not None
    assert torch.all(logits.grad[~mask] == 0)
    # Closed form for softmax + cross-entropy: dL/dz = (q - p) / N on legal actions.
    q = masked_softmax(logits.detach(), mask)
    torch.testing.assert_close(logits.grad, (q - p) / len(p))


def test_backpropagation_matches_finite_differences(teacher: TeacherDataset) -> None:
    model = PolicyNetwork(PolicyConfig((4,), seed=3)).double()
    x = torch.from_numpy(teacher.features()).double()
    mask = torch.from_numpy(teacher.masks())
    p = torch.from_numpy(teacher.targets())

    def loss() -> torch.Tensor:
        return -(p * masked_log_softmax(model(x), mask)).sum(dim=-1).mean()

    model.zero_grad()
    backward(loss())
    layer = model.layers[0]
    assert isinstance(layer, torch.nn.Linear)
    weight = layer.weight
    assert weight.grad is not None
    eps = 1e-6
    for i, j in [(0, 0), (1, 3), (3, 6)]:
        with torch.no_grad():
            weight[i, j] += eps
            up = float(loss())
            weight[i, j] -= 2 * eps
            down = float(loss())
            weight[i, j] += eps
        assert float(weight.grad[i, j]) == pytest.approx((up - down) / (2 * eps), abs=1e-8)


def test_model_strategy_is_a_valid_strategy(trained: PolicyNetwork) -> None:
    table = model_strategy(trained)
    validate_strategy(table)  # float64 renormalization keeps sums within 1e-9
    for key, spec in information_sets().items():
        assert set(table[key]) == set(spec.actions)
        assert abs(sum(table[key].values()) - 1) < 1e-12


# -- initialization and training determinism ---------------------------------------------------


def test_initialization_is_deterministic_and_independent_of_global_rng() -> None:
    torch.manual_seed(123)
    a = PolicyNetwork(PolicyConfig((16,), seed=5))
    torch.manual_seed(999)
    _ = torch.rand(100)
    b = PolicyNetwork(PolicyConfig((16,), seed=5))
    c = PolicyNetwork(PolicyConfig((16,), seed=6))
    assert state_dict_digest(a) == state_dict_digest(b)
    assert state_dict_digest(a) != state_dict_digest(c)


def test_training_is_bit_for_bit_deterministic(teacher: TeacherDataset) -> None:
    cfg = (PolicyConfig((16, 16), seed=1), TrainingConfig(epochs=200, learning_rate=0.03))
    m1, h1 = train_policy(teacher, *cfg)
    m2, h2 = train_policy(teacher, *cfg)
    assert state_dict_digest(m1) == state_dict_digest(m2)
    assert h1.cross_entropy == h2.cross_entropy
    assert model_strategy(m1) == model_strategy(m2)


def test_training_reduces_kl_and_records_history(teacher: TeacherDataset) -> None:
    _, history = train_policy(teacher, PolicyConfig((16,)), TrainingConfig(300, 0.03))
    assert history.epoch[0] == 0 and history.epoch[-1] == 300
    assert len(history.kl) == 301
    assert history.kl[-1] < history.kl[0] / 100
    assert all(k >= -1e-6 for k in history.kl)  # KL >= 0 up to float32 rounding
    # KL = CE - H(p): the gap is the same constant at every epoch.
    gaps = {round(ce - kl, 5) for ce, kl in zip(history.cross_entropy, history.kl, strict=True)}
    assert len(gaps) == 1


def test_training_config_validation() -> None:
    for kwargs in ({"epochs": 0}, {"learning_rate": 0.0}, {"epochs": 5, "checkpoint_epochs": (6,)}):
        with pytest.raises(ValueError):
            TrainingConfig(**kwargs)
    with pytest.raises(ValueError):
        PolicyConfig((0,))


def test_training_checkpoints(tmp_path: Path, teacher: TeacherDataset) -> None:
    cfg = TrainingConfig(epochs=50, learning_rate=0.03, checkpoint_epochs=(10, 50))
    model, history = train_policy(teacher, PolicyConfig((16,)), cfg, tmp_path)
    assert sorted(history.checkpoints) == [10, 50]
    final, meta = load_checkpoint(Path(history.checkpoints[50]))
    assert meta["epoch"] == 50
    assert state_dict_digest(final) == state_dict_digest(model)
    with pytest.raises(ValueError):
        train_policy(teacher, PolicyConfig((16,)), cfg, None)


def test_hidden_layer_fits_teacher_but_linear_cannot(
    teacher: TeacherDataset, trained: PolicyNetwork
) -> None:
    """The linear model's BET-vs-CHECK logit gap is additive in card and history,
    which the teacher's log-odds are not (docs §9): an irreducible error."""
    teacher_table = table_from_dataset(teacher)
    mlp = compare_to_teacher(model_strategy(trained), teacher_table)
    assert mlp["max_abs_error"] < 0.01
    linear, _ = train_policy(teacher, PolicyConfig(()), TrainingConfig(3_000, 0.03))
    lin = compare_to_teacher(model_strategy(linear), teacher_table)
    assert lin["max_abs_error"] > 0.05
    assert exploitability(model_strategy(linear)) > 10 * exploitability(teacher_table)


# -- serialization ------------------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path: Path, trained: PolicyNetwork) -> None:
    path = tmp_path / "m.pt"
    size = save_checkpoint(path, trained, {"note": "test"})
    assert size == path.stat().st_size
    loaded, meta = load_checkpoint(path)
    assert meta == {"note": "test"}
    assert loaded.config == trained.config
    assert state_dict_digest(loaded) == state_dict_digest(trained)
    assert model_strategy(loaded) == model_strategy(trained)


def test_checkpoint_rejects_foreign_files(tmp_path: Path) -> None:
    path = tmp_path / "x.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


# -- agent and evaluator integration ------------------------------------------------------------


def test_agent_protocol_and_seeded_sampling(trained: PolicyNetwork) -> None:
    agent = NeuralPolicyAgent(trained)
    assert isinstance(agent, Agent)
    info = InfoState(1, Card.Q, (B,))
    picks = [agent.act(info, (F, C), np.random.default_rng(4)) for _ in range(3)]
    assert len(set(picks)) == 1
    dist = agent.distribution(info, (F, C))
    assert dist == model_strategy(trained)["Q|b"]


def test_agent_matches_table_agent_with_same_probabilities(trained: PolicyNetwork) -> None:
    """Same probabilities + shared sampler + same seed => identical simulated hands."""
    cfg = SimulationConfig(num_hands=3_000, seed=8)
    neural = run_simulation((NeuralPolicyAgent(trained), RandomAgent()), cfg)
    table = run_simulation((StrategyAgent(model_strategy(trained)), RandomAgent()), cfg)
    assert neural.digest == table.digest


@pytest.mark.parametrize("opponent", ["random", "rule"])
def test_agent_plays_through_simulator(trained: PolicyNetwork, opponent: str) -> None:
    opp: Agent = RandomAgent() if opponent == "random" else ThresholdAgent()
    table = uniform_strategy() if opponent == "random" else deterministic_agent_strategy(opp)
    result = run_simulation((NeuralPolicyAgent(trained), opp), SimulationConfig(20_000, seed=2))
    assert result.invalid_actions == (0, 0)
    exact = expected_value_seat_averaged(model_strategy(trained), table)
    assert abs(result.mean_returns[0] - exact) < 4 * result.std_errors[0]


def test_exploitability_attribution(teacher: TeacherDataset) -> None:
    table = table_from_dataset(teacher)
    assert all(v == 0 for v in exploitability_attribution(table, table).values())
    student = {k: dict(v) for k, v in table.items()}
    student["Q|b"] = {F: 0.4, C: 0.6}
    attribution = exploitability_attribution(student, table)
    assert next(iter(attribution)) == "Q|b"
    assert attribution["Q|b"] > 0.01


# -- experiment runner and CLI ---------------------------------------------------------------------


def test_experiment_runner(tmp_path: Path) -> None:
    cfg = NeuralExperimentConfig(
        architectures=("linear", "mlp16"),
        learning_rates=(0.03,),
        seeds=(0,),
        epochs=200,
        teacher_iterations=2_000,
        simulation_hands=2_000,
        benchmark=True,
        benchmark_calls=50,
        benchmark_repeats=2,
    )
    doc = run_neural_experiment(cfg, tmp_path)
    assert {r["run_id"] for r in doc["runs"]} == {"linear_lr0.03_seed0", "mlp16_lr0.03_seed0"}
    for r in doc["runs"]:
        assert (tmp_path / r["checkpoint"]).is_file()
        loaded, _ = load_checkpoint(tmp_path / r["checkpoint"])
        assert exploitability(model_strategy(loaded)) == pytest.approx(r["exploitability"])
    assert (tmp_path / "training_curves.png").stat().st_size > 10_000
    latency = doc["selected"]["mlp16"]["latency"]
    assert latency["conditions"]["torch_threads"] == 1
    assert latency["table_lookup_ns"] > 0
    assert json.loads((tmp_path / "metrics.json").read_text())["teacher"]["table_entries"] == 12
    again = run_neural_experiment(cfg, tmp_path / "again")
    assert [r["weights_digest"] for r in again["runs"]] == [
        r["weights_digest"] for r in doc["runs"]
    ]


def test_cli_train_and_simulate_neural(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run"
    args = ["train-neural", "--output-dir", str(out), "--epochs", "100", "--seeds", "0"]
    args += ["--learning-rates", "0.03", "--architectures", "mlp16", "--teacher-iterations", "500"]
    assert main([*args, "--simulation-hands", "0", "--no-benchmark"]) == 0
    assert "mlp16_lr0.03_seed0" in capsys.readouterr().out
    model = out / "models" / "mlp16_lr0.03_seed0.pt"
    sim = ["simulate", "--agents", "neural", "rule", "--model", str(model), "--hands", "500"]
    assert main([*sim, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["result"]["invalid_actions"] == [0, 0]
    assert report["config"]["model"] == str(model)
    assert main(["simulate", "--agents", "neural", "rule"]) == 2
    assert main([*sim, "--ante", "2"]) == 2
    assert main(["train-neural", "--architectures", "huge", "--output-dir", str(out)]) == 2
