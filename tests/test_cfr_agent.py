"""CFR / strategy agent: sampling, file round-trip, simulator and CLI integration."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from pokermind.agents import Agent, RandomAgent, StrategyAgent, ThresholdAgent
from pokermind.cli import main
from pokermind.evaluation import expected_value_seat_averaged, uniform_strategy
from pokermind.experiments import deterministic_agent_strategy
from pokermind.games.kuhn import DECK, Action, Card, InfoState, KuhnState, all_deals, initial_state
from pokermind.learning import train_cfr
from pokermind.simulation import SimulationConfig, run_simulation
from pokermind.strategy import (
    StrategyFormatError,
    load_strategy,
    save_strategy,
    strategy_digest,
)

X, B, C, F = Action.CHECK, Action.BET, Action.CALL, Action.FOLD


@pytest.fixture(scope="module")
def cfr_strategy() -> dict[str, dict[Action, float]]:
    return train_cfr(2_000).average_strategy()


def walk(state: KuhnState) -> Iterator[KuhnState]:
    yield state
    for action in state.legal_actions():
        yield from walk(state.apply(action))


def test_satisfies_agent_protocol(cfr_strategy: dict[str, dict[Action, float]]) -> None:
    assert isinstance(StrategyAgent(cfr_strategy), Agent)


def test_sampling_frequencies_match_probabilities() -> None:
    strategy = uniform_strategy()
    strategy["Q|b"] = {F: 2 / 3, C: 1 / 3}
    agent, rng = StrategyAgent(strategy), np.random.default_rng(0)
    info = InfoState(player=1, private_card=Card.Q, history=(B,))
    n = 30_000
    counts = Counter(agent.act(info, (F, C), rng) for _ in range(n))
    assert set(counts) == {F, C}
    assert abs(counts[C] / n - 1 / 3) < 4 * (1 / 3 * 2 / 3 / n) ** 0.5


def test_pure_entries_are_deterministic_and_zero_probability_never_sampled() -> None:
    strategy = uniform_strategy()
    strategy["K|b"] = {F: 0.0, C: 1.0}
    agent, rng = StrategyAgent(strategy), np.random.default_rng(3)
    info = InfoState(player=1, private_card=Card.K, history=(B,))
    assert {agent.act(info, (F, C), rng) for _ in range(2_000)} == {C}


def test_seeded_sampling_is_reproducible(cfr_strategy: dict[str, dict[Action, float]]) -> None:
    info = InfoState(player=0, private_card=Card.J, history=())

    def picks(seed: int) -> list[Action]:
        agent, rng = StrategyAgent(cfr_strategy), np.random.default_rng(seed)
        return [agent.act(info, (X, B), rng) for _ in range(500)]

    assert picks(7) == picks(7)
    assert picks(7) != picks(8)


def test_rejects_unknown_info_set_and_mismatched_legal_actions() -> None:
    agent, rng = StrategyAgent(uniform_strategy()), np.random.default_rng(0)
    with pytest.raises(KeyError):
        agent.act(InfoState(0, Card.K, (X, X)), (X, B), rng)
    with pytest.raises(ValueError, match="do not match"):
        agent.act(InfoState(0, Card.K, ()), (F, C), rng)


def test_rejects_invalid_probabilities() -> None:
    strategy = uniform_strategy()
    strategy["K|"] = {X: 0.7, B: 0.7}
    with pytest.raises(ValueError, match="invalid probabilities"):
        StrategyAgent(strategy).act(InfoState(0, Card.K, ()), (X, B), np.random.default_rng(0))


def test_agent_copies_its_table(cfr_strategy: dict[str, dict[Action, float]]) -> None:
    table = {k: dict(v) for k, v in cfr_strategy.items()}
    agent = StrategyAgent(table)
    table["K|b"][C] = 0.0
    assert agent.strategy["K|b"][C] == cfr_strategy["K|b"][C]


# -- files ----------------------------------------------------------------------


def test_strategy_file_round_trip_is_exact(
    tmp_path: Path, cfr_strategy: dict[str, dict[Action, float]]
) -> None:
    path = tmp_path / "s.json"
    save_strategy(path, cfr_strategy, {"iterations": 2_000})
    loaded = load_strategy(path)
    assert loaded == cfr_strategy  # bit-identical floats
    assert strategy_digest(loaded) == strategy_digest(cfr_strategy)
    assert StrategyAgent.from_file(path).strategy == cfr_strategy


def test_tampered_or_malformed_files_rejected(
    tmp_path: Path, cfr_strategy: dict[str, dict[Action, float]]
) -> None:
    path = tmp_path / "s.json"
    save_strategy(path, cfr_strategy)
    doc = json.loads(path.read_text())
    doc["strategy"]["K|b"]["c"] = 0.5
    path.write_text(json.dumps(doc))
    with pytest.raises(StrategyFormatError, match="digest"):
        load_strategy(path)
    for bad in ("not json", json.dumps({"format": "other"}), json.dumps([1, 2])):
        path.write_text(bad)
        with pytest.raises(StrategyFormatError):
            load_strategy(path)
    doc = json.loads(
        json.dumps(
            {"format": "pokermind.strategy.v1", "game": "kuhn", "strategy": {"K|": {"z": 1.0}}}
        )
    )
    path.write_text(json.dumps(doc))
    with pytest.raises(StrategyFormatError, match="unknown action"):
        load_strategy(path)


# -- hidden information -----------------------------------------------------------


def test_action_distribution_independent_of_opponent_card(
    cfr_strategy: dict[str, dict[Action, float]],
) -> None:
    """For every reachable decision, swapping the opponent's card leaves the
    agent's sampled actions unchanged under the same generator seed."""
    agent = StrategyAgent(cfr_strategy)
    checked = 0
    for cards in all_deals():
        for state in walk(initial_state(cards)):
            if state.is_terminal:
                continue
            p = state.current_player
            own = state.cards[p]
            outputs = set()
            for opp in DECK:
                if opp == own:
                    continue
                alt = KuhnState((own, opp) if p == 0 else (opp, own), state.history)
                rng = np.random.default_rng(42)
                outputs.add(
                    tuple(
                        agent.act(alt.information_state(p), alt.legal_actions(), rng)
                        for _ in range(20)
                    )
                )
            assert len(outputs) == 1
            checked += 1
    assert checked == 6 * 4


# -- simulator and CLI ---------------------------------------------------------------


@pytest.mark.parametrize("opponent", ["random", "rule"])
def test_plays_through_simulator_and_matches_exact_ev(
    opponent: str, cfr_strategy: dict[str, dict[Action, float]]
) -> None:
    opp_agent: Agent = RandomAgent() if opponent == "random" else ThresholdAgent()
    opp_table = (
        uniform_strategy() if opponent == "random" else deterministic_agent_strategy(opp_agent)
    )
    result = run_simulation(
        (StrategyAgent(cfr_strategy), opp_agent), SimulationConfig(num_hands=50_000, seed=12)
    )
    assert result.invalid_actions == (0, 0)
    assert result.zero_sum_violations == 0
    exact = expected_value_seat_averaged(cfr_strategy, opp_table)
    assert abs(result.mean_returns[0] - exact) < 4 * result.std_errors[0]


def test_simulation_with_cfr_agent_is_reproducible(
    cfr_strategy: dict[str, dict[Action, float]],
) -> None:
    cfg = SimulationConfig(num_hands=5_000, seed=3)
    a = run_simulation((StrategyAgent(cfr_strategy), RandomAgent()), cfg)
    b = run_simulation((StrategyAgent(cfr_strategy), RandomAgent()), cfg)
    assert a.digest == b.digest


def test_cli_simulate_with_cfr_strategy(
    tmp_path: Path, cfr_strategy: dict[str, dict[Action, float]], capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "s.json"
    save_strategy(path, cfr_strategy)
    args = ["simulate", "--agents", "cfr", "rule", "--strategy", str(path), "--hands", "2000"]
    assert main([*args, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config"]["agents"] == ["cfr", "rule"]
    assert report["config"]["strategy"] == str(path)
    assert report["result"]["invalid_actions"] == [0, 0]


def test_cli_cfr_requires_valid_strategy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["simulate", "--agents", "cfr", "random"]) == 2
    assert "requires --strategy" in capsys.readouterr().err
    assert main(["simulate", "--agents", "cfr", "random", "--strategy", str(tmp_path / "x")]) == 2
    incomplete = tmp_path / "incomplete.json"
    save_strategy(incomplete, {"K|": {X: 0.5, B: 0.5}})
    assert main(["simulate", "--agents", "cfr", "random", "--strategy", str(incomplete)]) == 2
    assert "missing information sets" in capsys.readouterr().err


def test_cli_lists_cfr_agent(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["agents"]) == 0
    assert "cfr" in capsys.readouterr().out
