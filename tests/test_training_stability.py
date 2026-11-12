"""M2B: learning-rate schedules, best-checkpoint selection, early stopping, studies,
and benchmark methodology."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pokermind.cli import main
from pokermind.evaluation import exploitability
from pokermind.experiments.latency import benchmark_policies, time_statement
from pokermind.experiments.neural_experiment import table_from_dataset
from pokermind.experiments.stability_study import (
    CAPACITY_ARCHITECTURES,
    PROCEDURES,
    StudyConfig,
    count_spikes,
    run_studies,
    training_config,
)
from pokermind.neural import PolicyConfig, PolicyNetwork, TrainingConfig, load_teacher, train_policy
from pokermind.neural.dataset import TeacherDataset
from pokermind.neural.model import model_strategy, parameter_count
from pokermind.neural.training import learning_rate_at, state_dict_digest

BASELINE = (
    Path(__file__).resolve().parents[1] / "docs" / "experiments" / "m2a_baseline_results.json"
)


@pytest.fixture(scope="module")
def teacher() -> TeacherDataset:
    return load_teacher(iterations=50_000)


@pytest.fixture(scope="module")
def m2a_runs() -> dict[str, dict[str, object]]:
    data = json.loads(BASELINE.read_text())
    return {r["run_id"]: r for r in data["runs"]}


# -- regression: the defaults are exactly M2A -------------------------------------------------


@pytest.mark.parametrize(
    ("run_id", "hidden", "lr", "seed"),
    [
        ("mlp16_lr0.01_seed2", (16,), 0.01, 2),
        ("mlp16x16_lr0.03_seed0", (16, 16), 0.03, 0),  # the run that ended on a spike
    ],
)
def test_default_training_reproduces_m2a_weights(
    teacher: TeacherDataset,
    m2a_runs: dict[str, dict[str, object]],
    run_id: str,
    hidden: tuple[int, ...],
    lr: float,
    seed: int,
) -> None:
    model, history = train_policy(teacher, PolicyConfig(hidden, seed), TrainingConfig(3_000, lr))
    assert state_dict_digest(model) == m2a_runs[run_id]["weights_digest"]
    assert history.selected_epoch == 3_000
    assert not history.stopped_early
    assert set(history.learning_rate[:-1]) == {lr}


# -- learning-rate schedule --------------------------------------------------------------------


def test_cosine_schedule_values() -> None:
    cfg = TrainingConfig(epochs=100, learning_rate=0.03, lr_schedule="cosine", min_lr_factor=0.1)
    assert learning_rate_at(cfg, 0) == pytest.approx(0.03)
    assert learning_rate_at(cfg, 50) == pytest.approx(0.003 + 0.5 * 0.027)
    assert learning_rate_at(cfg, 100) == pytest.approx(0.003)
    values = [learning_rate_at(cfg, t) for t in range(101)]
    assert values == sorted(values, reverse=True)


def test_cosine_schedule_is_applied_to_the_optimizer(teacher: TeacherDataset) -> None:
    cfg = TrainingConfig(epochs=50, learning_rate=0.03, lr_schedule="cosine")
    _, history = train_policy(teacher, PolicyConfig((4,)), cfg)
    assert history.learning_rate[:-1] == [learning_rate_at(cfg, t) for t in range(50)]
    assert math.isnan(history.learning_rate[-1])


def test_cosine_removes_the_m2a_failure(teacher: TeacherDataset) -> None:
    """The M2A run that ended on a loss spike is stable with cosine decay."""
    policy = PolicyConfig((16, 16), seed=0)
    base, _ = train_policy(teacher, policy, TrainingConfig(3_000, 0.03))
    cos, hist = train_policy(teacher, policy, TrainingConfig(3_000, 0.03, lr_schedule="cosine"))
    assert exploitability(model_strategy(base)) > 0.01
    assert exploitability(model_strategy(cos)) < 0.0011
    assert count_spikes(hist.kl) < count_spikes(
        train_policy(teacher, policy, TrainingConfig(3_000, 0.03))[1].kl
    )


# -- best checkpoint -----------------------------------------------------------------------------


def test_keep_best_returns_the_lowest_recorded_kl(teacher: TeacherDataset) -> None:
    policy = PolicyConfig((16, 16), seed=0)
    cfg = TrainingConfig(3_000, 0.03, keep_best=True)
    model, history = train_policy(teacher, policy, cfg)
    best = min(range(len(history.kl)), key=lambda i: history.kl[i])
    assert history.selected_epoch == history.epoch[best]
    assert history.selected_kl == min(history.kl)
    # The returned weights really are that epoch's weights: retraining for exactly
    # that many constant-rate updates gives the same parameters.
    replay, _ = train_policy(teacher, policy, TrainingConfig(history.selected_epoch, 0.03))
    assert state_dict_digest(model) == state_dict_digest(replay)


def test_keep_best_can_select_the_initial_parameters() -> None:
    """With a learning rate so large that the first step is worse, epoch 0 wins."""
    ds = load_teacher(iterations=200)
    model, history = train_policy(
        ds, PolicyConfig((), seed=0), TrainingConfig(3, 50.0, keep_best=True)
    )
    assert history.kl[1] > history.kl[0]  # the first step overshoots (KL 0.43 -> 68)
    assert history.selected_epoch == 0
    assert state_dict_digest(model) == state_dict_digest(PolicyNetwork(PolicyConfig((), 0)))
    assert history.selected_kl == min(history.kl)


# -- early stopping -------------------------------------------------------------------------------


def test_early_stopping_stops_and_is_deterministic(teacher: TeacherDataset) -> None:
    cfg = TrainingConfig(
        3_000, 0.03, keep_best=True, early_stopping_patience=50, min_relative_improvement=0.5
    )
    m1, h1 = train_policy(teacher, PolicyConfig((8,), seed=1), cfg)
    m2, h2 = train_policy(teacher, PolicyConfig((8,), seed=1), cfg)
    assert h1.stopped_early
    assert h1.epoch[-1] < 3_000
    assert state_dict_digest(m1) == state_dict_digest(m2)
    assert h1.kl == h2.kl
    # The stop happened exactly `patience` epochs after the last qualifying improvement.
    last = h1.epoch[-1]
    window = h1.kl[: last - 50 + 1]
    assert min(h1.kl[last - 50 + 1 :]) >= min(window) * 0.5


def test_training_config_validation() -> None:
    for kwargs in (
        {"lr_schedule": "step"},
        {"min_lr_factor": 0.0},
        {"min_lr_factor": 1.5},
        {"early_stopping_patience": 0},
        {"min_relative_improvement": 1.0},
    ):
        with pytest.raises(ValueError):
            TrainingConfig(**kwargs)


# -- study helpers ----------------------------------------------------------


def test_count_spikes() -> None:
    assert count_spikes([1.0, 0.5, 0.1, 0.05]) == 0
    assert count_spikes([1.0, 0.01, 0.5, 0.4, 0.005, 0.001, 0.2]) == 2
    assert count_spikes([1.0, 0.01, 0.2, 0.3, 0.02]) == 1  # one sustained excursion
    assert count_spikes([]) == 0


def test_procedures_and_capacity_architectures() -> None:
    assert training_config("baseline", 10, 0.01) == TrainingConfig(10, 0.01)
    assert training_config("cosine_best", 10, 0.01).keep_best
    assert set(PROCEDURES) == {
        "baseline",
        "best_checkpoint",
        "cosine",
        "cosine_best",
        "early_stop_best",
    }
    counts = {
        a: parameter_count(PolicyNetwork(PolicyConfig(h)))
        for a, h in CAPACITY_ARCHITECTURES.items()
    }
    assert counts == {"linear": 32, "mlp2": 28, "mlp4": 52, "mlp8": 100, "mlp16": 196}
    with pytest.raises(ValueError):
        StudyConfig(capacity_procedure="nope")


def test_studies_run_and_are_reproducible(tmp_path: Path) -> None:
    cfg = StudyConfig(
        seeds=(0, 1),
        learning_rates=(0.03,),
        epochs=60,
        teacher_iterations=300,
        benchmark_number=200,
        benchmark_repeats=3,
    )
    a = run_studies(cfg, tmp_path / "a", ("stability", "capacity"))
    b = run_studies(cfg, tmp_path / "b", ("stability", "capacity"))
    for study in ("stability", "capacity"):
        rows_a = [{k: v for k, v in r.items() if k != "train_seconds"} for r in a[study]["rows"]]
        rows_b = [{k: v for k, v in r.items() if k != "train_seconds"} for r in b[study]["rows"]]
        assert rows_a == rows_b
    assert len(a["stability"]["rows"]) == 2 * 1 * len(PROCEDURES) * 2
    assert len(a["capacity"]["rows"]) == len(CAPACITY_ARCHITECTURES) * 2
    for name in ("stability.json", "capacity.json", "stability.png", "capacity.png"):
        assert (tmp_path / "a" / name).is_file()
    pooled = a["stability"]["pooled"]["baseline"]
    assert pooled["runs"] == 4 and {"min", "median", "max"} <= set(pooled["exploitability"])


# -- benchmark methodology ----------------------------------------------------------


def test_time_statement_reports_uncertainty() -> None:
    t = time_statement(
        "sum", "sum of a small list", "_ = sum(xs)", {"xs": list(range(50))}, number=500, repeats=5
    )
    assert t.repeats == 5
    assert t.min_ns <= t.q1_ns <= t.median_ns <= t.q3_ns <= t.max_ns
    assert t.relative_iqr >= 0
    empty = time_statement("empty", "pass", "pass", {}, number=500, repeats=5)
    assert empty.median_ns < t.median_ns


def test_benchmark_separates_the_four_operations(teacher: TeacherDataset) -> None:
    model, _ = train_policy(teacher, PolicyConfig((4,)), TrainingConfig(20, 0.03))
    result = benchmark_policies(
        model, table_from_dataset(teacher), number=200, repeats=3, include_mps=False
    )
    timings = result["timings"]
    assert set(timings) == {
        "empty_statement",
        "dict_lookup",
        "table_agent_decision",
        "nn_forward_cpu",
        "neural_agent_decision",
    }
    for name, t in timings.items():
        assert t["repeats"] == 3 and t["median_ns"] > 0, name
        if name != "empty_statement":
            assert t["net_median_ns"] == pytest.approx(
                t["median_ns"] - timings["empty_statement"]["median_ns"]
            )
    # Coarse ordering that holds on any machine: a full decision costs more than
    # the bare operation it contains.
    assert timings["table_agent_decision"]["median_ns"] > timings["dict_lookup"]["median_ns"]
    assert timings["neural_agent_decision"]["median_ns"] > timings["nn_forward_cpu"]["median_ns"]
    assert result["conditions"]["torch_threads"] == 1
    assert result["mps"] == {"available": False, "reason": "disabled"}


def test_cli_neural_study(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "neural-study",
        "--studies",
        "capacity",
        "--seeds",
        "1",
        "--epochs",
        "30",
        "--learning-rates",
        "0.03",
        "--teacher-iterations",
        "200",
        "--output-dir",
        str(tmp_path),
    ]
    assert main(args) == 0
    assert "mlp2|lr0.03" in capsys.readouterr().out
    assert (tmp_path / "capacity.json").is_file()
    assert main([*args[:3], "--seeds", "0", "--output-dir", str(tmp_path)]) == 2
