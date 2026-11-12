"""CFR experiment runner: artifacts, reproducibility, CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from pokermind.agents import RandomAgent, ThresholdAgent
from pokermind.cli import main
from pokermind.evaluation import exploitability
from pokermind.experiments import (
    CFRExperimentConfig,
    deterministic_agent_strategy,
    run_cfr_experiment,
)
from pokermind.strategy import load_strategy, strategy_digest

TIMING = {"train_seconds_cumulative", "eval_seconds"}


def _strip_timing(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoints": [
            {k: v for k, v in cp.items() if k not in TIMING} for cp in doc["checkpoints"]
        ],
        "simulation_check": doc["simulation_check"],
        "config": doc["config"],
    }


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    out = tmp_path_factory.mktemp("cfr")
    cfg = CFRExperimentConfig(checkpoints=(1, 10, 100, 500), seed=5, simulation_hands=5_000)
    return out, run_cfr_experiment(cfg, out)


def test_artifacts_written(experiment: tuple[Path, dict[str, Any]]) -> None:
    out, doc = experiment
    assert (out / "metrics.json").is_file()
    assert (out / "metrics.csv").is_file()
    assert (out / "convergence.png").stat().st_size > 10_000
    assert sorted(p.name for p in (out / "strategies").iterdir()) == [
        "strategy_000001.json",
        "strategy_000010.json",
        "strategy_000100.json",
        "strategy_000500.json",
    ]
    with (out / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["iteration"]) for r in rows] == [1, 10, 100, 500]
    on_disk = json.loads((out / "metrics.json").read_text())
    assert on_disk["checkpoints"] == doc["checkpoints"]


def test_recorded_fields(experiment: tuple[Path, dict[str, Any]]) -> None:
    _, doc = experiment
    assert doc["config"]["seed"] == 5
    assert doc["config"]["checkpoints"] == [1, 10, 100, 500]
    for key in ("python_version", "numpy_version", "matplotlib_version", "platform", "git"):
        assert key in doc["metadata"]
    for cp in doc["checkpoints"]:
        for key in (
            "iteration",
            "exploitability",
            "nash_conv",
            "self_play_value_seat0",
            "expected_payoff_vs",
            "average_strategy",
            "train_seconds_cumulative",
            "strategy_digest",
            "alpha",
            "max_deviation_from_equilibrium",
        ):
            assert key in cp
        assert cp["nash_conv"] == pytest.approx(2 * cp["exploitability"])


def test_exported_strategies_match_metrics(experiment: tuple[Path, dict[str, Any]]) -> None:
    out, doc = experiment
    for cp in doc["checkpoints"]:
        strategy = load_strategy(out / cp["strategy_file"])
        assert strategy_digest(strategy) == cp["strategy_digest"]
        # Re-evaluated independently from the file.
        assert exploitability(strategy) == pytest.approx(cp["exploitability"], abs=1e-15)


def test_exploitability_decreases_across_checkpoints(
    experiment: tuple[Path, dict[str, Any]],
) -> None:
    _, doc = experiment
    values = [cp["exploitability"] for cp in doc["checkpoints"]]
    assert values == sorted(values, reverse=True)


def test_simulation_check_agrees_with_exact_ev(experiment: tuple[Path, dict[str, Any]]) -> None:
    _, doc = experiment
    assert {s["opponent"] for s in doc["simulation_check"]} == {"random", "rule"}
    for sim in doc["simulation_check"]:
        assert sim["invalid_actions"] == [0, 0]
        assert abs(sim["z_score"]) < 4


def test_experiment_is_reproducible(
    tmp_path: Path, experiment: tuple[Path, dict[str, Any]]
) -> None:
    _, first = experiment
    cfg = CFRExperimentConfig(checkpoints=(1, 10, 100, 500), seed=5, simulation_hands=5_000)
    second = run_cfr_experiment(cfg, tmp_path)
    assert _strip_timing(first) == _strip_timing(second)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checkpoints": ()},
        {"checkpoints": (10, 5)},
        {"checkpoints": (0, 5)},
        {"checkpoints": (5, 5)},
        {"seed": -1},
        {"simulation_hands": -1},
    ],
)
def test_invalid_experiment_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CFRExperimentConfig(**kwargs)


def test_deterministic_agent_strategy() -> None:
    rule = deterministic_agent_strategy(ThresholdAgent())
    assert len(rule) == 12
    assert all(sorted(d.values()) == [0.0, 1.0] for d in rule.values())
    with pytest.raises(ValueError, match="not deterministic"):
        deterministic_agent_strategy(RandomAgent())


def test_cli_train_cfr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run"
    args = ["train-cfr", "--checkpoints", "5", "50", "--simulation-hands", "0"]
    assert main([*args, "--output-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert "exploitability" in text
    doc = json.loads((out / "metrics.json").read_text())
    assert [cp["iteration"] for cp in doc["checkpoints"]] == [5, 50]
    assert doc["simulation_check"] == []
    assert main(["train-cfr", "--checkpoints", "50", "5", "--output-dir", str(out)]) == 2
