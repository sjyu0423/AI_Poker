"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokermind.cli import main


def test_simulate_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "--agents", "random", "rule", "--hands", "500", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "500 hands, seed=3" in out
    assert "zero-sum violations: 0" in out
    assert "result digest:" in out


def test_simulate_json_output_and_reproducibility(tmp_path: Path) -> None:
    args = ["simulate", "--agents", "random", "random", "--hands", "2000", "--seed", "21"]
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    assert main([*args, "--output", str(out1)]) == 0
    assert main([*args, "--output", str(out2)]) == 0
    r1, r2 = json.loads(out1.read_text()), json.loads(out2.read_text())
    assert r1["config"] == r2["config"]
    for r in (r1, r2):
        del r["result"]["elapsed_seconds"], r["result"]["hands_per_second"]
    assert r1["result"] == r2["result"]
    meta = r1["metadata"]
    for key in ("python_version", "numpy_version", "platform", "timestamp_utc", "git", "command"):
        assert key in meta
    assert r1["config"]["seed"] == 21
    assert r1["result"]["zero_sum_violations"] == 0
    assert r1["result"]["invalid_actions"] == [0, 0]


def test_simulate_print_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", "--hands", "10", "--json", "--no-alternate-seats"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["config"]["alternate_seats"] is False


@pytest.mark.parametrize(
    "args", [["--hands", "0"], ["--seed", "-1"], ["--ante", "0"], ["--bet-size", "-2"]]
)
def test_simulate_rejects_bad_config(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", *args]) == 2
    assert "error:" in capsys.readouterr().err


def test_unknown_agent_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["simulate", "--agents", "random", "nope"])


def test_list_agents(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["agents"]) == 0
    out = capsys.readouterr().out
    for name in ("random", "rule", "always-pass", "always-bet"):
        assert name in out
