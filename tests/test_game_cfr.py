"""Generic CFR and exact evaluation through the game interface (M3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pokermind.agents.game_agents import GameStrategyAgent, UniformGameAgent
from pokermind.cli import main
from pokermind.evaluation import exploitability_report as legacy_report
from pokermind.evaluation import uniform_strategy
from pokermind.evaluation.game_exact import (
    InvalidGameStrategyError,
    best_response,
    expected_value,
    expected_value_seat_averaged,
    exploitability_report,
    information_sets,
    tree_counts,
    uniform_game_strategy,
    validate_game_strategy,
)
from pokermind.evaluation.reference import as_float_strategy, kuhn_equilibrium
from pokermind.games.base import Game, GameState
from pokermind.games.kuhn_game import (
    KUHN_ACTIONS,
    KuhnGame,
    from_action_strategy,
    to_action_strategy,
)
from pokermind.games.leduc import LeducGame
from pokermind.learning import CFRSolver, train_cfr
from pokermind.learning.game_cfr import GameCFRSolver, train_game_cfr
from pokermind.simulation.game_runner import GameSimulationConfig, run_game_simulation
from pokermind.strategy import (
    StrategyFormatError,
    game_strategy_digest,
    load_game_strategy,
    load_strategy,
    save_game_strategy,
    strategy_digest,
)

M1_DIGEST_50K = "b8217a0c74657df799530f51315c1f8fa3d4c299a07f196af8474e65ffc0becc"
KUHN, LEDUC = KuhnGame(), LeducGame()


def random_strategy(game: Game, seed: int) -> dict[str, dict[int, float]]:
    rng = np.random.default_rng(seed)
    out = {}
    for key, spec in information_sets(game).items():
        probs = rng.dirichlet(np.ones(len(spec.actions)) * 0.7)
        out[key] = {a: float(p) for a, p in zip(spec.actions, probs, strict=True)}
    return out


# -- Kuhn regression: the generic path reproduces M1 bit for bit ----------------------------


def test_kuhn_adapter_structure() -> None:
    counts = tree_counts(KUHN)
    assert (counts.chance, counts.decision, counts.terminal) == (1, 24, 30)
    assert set(information_sets(KUHN)) == set(CFRSolver().tables)
    assert KUHN.action_codes == ("x", "b", "c", "f")
    assert [a.code for a in KUHN_ACTIONS] == list(KUHN.action_codes)


def test_generic_cfr_reproduces_m1_full_digest() -> None:
    """Acceptance criterion: the complete M1 digest, not a prefix."""
    generic = train_game_cfr(KUHN, 50_000)
    avg = generic.average_strategy()
    assert game_strategy_digest(avg, KUHN.action_codes) == M1_DIGEST_50K
    assert strategy_digest(to_action_strategy(avg)) == M1_DIGEST_50K


@pytest.mark.parametrize("iterations", [1, 7, 300])
def test_generic_cfr_tables_bit_identical_to_m1(iterations: int) -> None:
    legacy, generic = train_cfr(iterations), train_game_cfr(KUHN, iterations)
    assert to_action_strategy(generic.average_strategy()) == legacy.average_strategy()
    assert to_action_strategy(generic.current_strategy()) == legacy.current_strategy()
    assert to_action_strategy(generic.cumulative_regrets()) == legacy.cumulative_regrets()


@pytest.mark.parametrize("seed", range(5))
def test_generic_evaluator_matches_m1_evaluator_on_kuhn(seed: int) -> None:
    from pokermind.evaluation.exact import information_sets as legacy_specs

    rng = np.random.default_rng(seed)
    legacy = {
        k: dict(zip(s.actions, map(float, rng.dirichlet(np.ones(2))), strict=True))
        for k, s in legacy_specs().items()
    }
    old = legacy_report(legacy)
    new = exploitability_report(KUHN, from_action_strategy(legacy))
    # Best responses and NashConv are bit-identical.
    assert new.best_response_values == old.best_response_values
    assert new.nash_conv == old.nash_conv
    # The on-policy value is computed as Σ(1/6)·v instead of Σv/6: equal to rounding.
    assert new.on_policy_values[0] == pytest.approx(old.on_policy_values[0], abs=1e-15)


def test_kuhn_known_values_through_generic_evaluator() -> None:
    assert exploitability_report(KUHN, from_action_strategy(uniform_strategy())).nash_conv == (
        pytest.approx(11 / 12)
    )
    eq = from_action_strategy(as_float_strategy(kuhn_equilibrium(0)))
    report = exploitability_report(KUHN, eq)
    assert abs(report.nash_conv) < 1e-12
    assert report.on_policy_values[0] == pytest.approx(-1 / 18)


def test_generic_strategy_files_interoperate_with_m1(tmp_path: Path) -> None:
    avg = train_game_cfr(KUHN, 200).average_strategy()
    path = tmp_path / "kuhn.json"
    save_game_strategy(path, "kuhn", KUHN.action_codes, avg)
    assert load_strategy(path) == to_action_strategy(avg)  # M1 loader reads it
    assert load_game_strategy(path, "kuhn", KUHN.action_codes) == avg
    with pytest.raises(StrategyFormatError):
        load_game_strategy(path, "leduc", LEDUC.action_codes)


# -- generic CFR on Leduc ---------------------------------------------------------------------


def test_leduc_tables_cover_all_information_sets() -> None:
    solver = GameCFRSolver(LEDUC)
    specs = information_sets(LEDUC)
    assert set(solver.tables) == set(specs)
    for key, table in solver.tables.items():
        assert table.actions == specs[key].actions
        assert table.player == specs[key].player


def _independent_first_iteration_regret(game: Game, key: str, player: int) -> dict[int, float]:
    """r(I, a) = Σ_{h∈I} π_c(h)·π_{-i}(h)·(u_i(h·a) − u_i(h)) under the uniform
    strategy, computed by direct tree enumeration (no CFR code involved)."""
    uniform = uniform_game_strategy(game)

    def value(state: GameState) -> float:
        if state.is_terminal:
            return state.returns()[player]
        if state.is_chance_node:
            return sum(p * value(state.apply(o)) for o, p in state.chance_outcomes())
        dist = uniform[state.information_state_key(state.current_player)]
        return sum(p * value(state.apply(a)) for a, p in dist.items())

    regrets: dict[int, float] = {}

    def visit(state: GameState, weight: float) -> None:
        if state.is_terminal:
            return
        if state.is_chance_node:
            for o, p in state.chance_outcomes():
                visit(state.apply(o), weight * p)
            return
        actor = state.current_player
        k = state.information_state_key(actor)
        if actor == player and k == key:
            v = value(state)
            for a in state.legal_actions():
                regrets[a] = regrets.get(a, 0.0) + weight * (value(state.apply(a)) - v)
        for a, p in uniform[k].items():
            visit(state.apply(a), weight * (p if actor != player else 1.0))

    visit(game.initial_state(), 1.0)
    return regrets


@pytest.mark.parametrize("key", ["K1|", "J0|cr", "Q1|rr", "K0|J1|rc|", "J0|J1|crrc|crr"])
def test_leduc_first_iteration_regrets(key: str) -> None:
    solver = GameCFRSolver(LEDUC)
    solver.iterate(1)
    player = information_sets(LEDUC)[key].player
    expected = _independent_first_iteration_regret(LEDUC, key, player)
    for action, value in solver.cumulative_regrets()[key].items():
        assert value == pytest.approx(expected[action], abs=1e-12)


def test_leduc_training_is_deterministic() -> None:
    a, b = train_game_cfr(LEDUC, 15), train_game_cfr(LEDUC, 15)
    assert a.average_strategy() == b.average_strategy()
    assert a.cumulative_regrets() == b.cumulative_regrets()
    assert a.decision_visits == b.decision_visits == 15 * 3_780


def test_leduc_exploitability_decreases() -> None:
    solver = GameCFRSolver(LEDUC)
    values = []
    for target in (1, 10, 40):
        solver.iterate(target - solver.iterations)
        values.append(exploitability_report(LEDUC, solver.average_strategy()).exploitability)
    assert values == sorted(values, reverse=True)
    assert values[-1] < values[0] / 5


# -- exact evaluation on Leduc -------------------------------------------------------------------


@pytest.fixture(scope="module")
def leduc_specs() -> dict[str, object]:
    return information_sets(LEDUC)  # type: ignore[return-value]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("player", [0, 1])
def test_best_response_policy_achieves_its_value(seed: int, player: int) -> None:
    specs = information_sets(LEDUC)
    strategy = random_strategy(LEDUC, seed)
    br = best_response(LEDUC, strategy, player, infosets=specs)
    pure = br.as_strategy(specs)
    profile = (pure, strategy) if player == 0 else (strategy, pure)
    achieved = expected_value(LEDUC, profile, infosets=specs)
    assert (achieved if player == 0 else -achieved) == pytest.approx(br.value, abs=1e-12)


@pytest.mark.parametrize("player", [0, 1])
def test_best_response_dominates_other_strategies(player: int) -> None:
    specs = information_sets(LEDUC)
    target = random_strategy(LEDUC, 11)
    br = best_response(LEDUC, target, player, infosets=specs).value
    for seed in range(20, 30):
        mine = random_strategy(LEDUC, seed)
        profile = (mine, target) if player == 0 else (target, mine)
        v = expected_value(LEDUC, profile, infosets=specs)
        assert (v if player == 0 else -v) <= br + 1e-12


def _omniscient_value(game: Game, strategy: dict[str, dict[int, float]], player: int) -> float:
    """A 'cheating' best response that sees every card: max per *history*."""

    def value(state: GameState) -> float:
        if state.is_terminal:
            return state.returns()[player]
        if state.is_chance_node:
            return sum(p * value(state.apply(o)) for o, p in state.chance_outcomes())
        actor = state.current_player
        if actor == player:
            return max(value(state.apply(a)) for a in state.legal_actions())
        dist = strategy[state.information_state_key(actor)]
        return sum(p * value(state.apply(a)) for a, p in dist.items())

    return value(game.initial_state())


@pytest.mark.parametrize("player", [0, 1])
def test_best_response_does_not_use_hidden_information(player: int) -> None:
    """Against uniform play, the information-set best response (2.0875 / 2.6597)
    is clearly below a best response that could see the opponent's card
    (2.375 / 2.925). It is respecting information sets."""
    uniform = uniform_game_strategy(LEDUC)
    br = best_response(LEDUC, uniform, player).value
    cheat = _omniscient_value(LEDUC, uniform, player)
    assert (br, cheat) == pytest.approx(
        ((2.0875, 2.375), (2.659722222222222, 2.925))[player], abs=1e-12
    )
    assert cheat - br > 0.25


def test_reference_best_response_values_for_uniform() -> None:
    report = exploitability_report(LEDUC, uniform_game_strategy(LEDUC))
    assert report.best_response_values == pytest.approx((2.0875, 2.659722222222222), abs=1e-12)
    assert report.exploitability == pytest.approx(4.747222222222222 / 2, abs=1e-12)


def test_validation_rejects_malformed_strategies() -> None:
    good = uniform_game_strategy(LEDUC)
    validate_game_strategy(LEDUC, good)
    bad = {k: dict(v) for k, v in good.items()}
    bad["K1|"] = {1: 0.5, 2: 0.5, 0: 0.0}  # fold is illegal at the opening
    with pytest.raises(InvalidGameStrategyError, match="legal"):
        validate_game_strategy(LEDUC, bad)
    bad = {k: dict(v) for k, v in good.items() if k != "K1|"}
    with pytest.raises(InvalidGameStrategyError, match="missing"):
        validate_game_strategy(LEDUC, bad)
    bad = {**good, "K1|K1|cc|": {1: 0.5, 2: 0.5}}  # K1 can't be both private and public
    with pytest.raises(InvalidGameStrategyError, match="unknown"):
        validate_game_strategy(LEDUC, bad)


def test_expected_value_symmetric_uniform() -> None:
    u = uniform_game_strategy(LEDUC)
    assert expected_value(LEDUC, (u, u)) == pytest.approx(-0.078125)
    assert expected_value_seat_averaged(LEDUC, u, u) == pytest.approx(0.0, abs=1e-15)


# -- simulation ---------------------------------------------------------------------------------


def test_simulation_is_reproducible_and_agrees_with_exact_value() -> None:
    avg = train_game_cfr(LEDUC, 20).average_strategy()
    cfg = GameSimulationConfig(num_hands=40_000, seed=3)
    agents = (GameStrategyAgent(avg), UniformGameAgent())
    a = run_game_simulation(LEDUC, agents, cfg)
    b = run_game_simulation(LEDUC, (GameStrategyAgent(avg), UniformGameAgent()), cfg)
    assert a["digest"] == b["digest"]
    assert a["zero_sum_violations"] == 0
    exact = expected_value_seat_averaged(LEDUC, avg, uniform_game_strategy(LEDUC))
    assert abs(a["mean_returns"][0] - exact) < 4 * a["std_errors"][0]


def test_agents_receive_only_observations() -> None:
    seen: list[object] = []

    class Spy(UniformGameAgent):
        def act(self, observation, legal_actions, rng):  # type: ignore[no-untyped-def]
            seen.append(observation)
            return super().act(observation, legal_actions, rng)

    run_game_simulation(LEDUC, (Spy(), Spy()), GameSimulationConfig(num_hands=300, seed=0))
    from pokermind.games.leduc import LeducObservation

    assert seen and all(type(o) is LeducObservation for o in seen)


# -- experiment runner and CLI ---------------------------------------------------------------------


def test_cli_train_cfr_leduc(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "leduc"
    args = [
        "train-cfr",
        "--game",
        "leduc",
        "--checkpoints",
        "1",
        "5",
        "--simulation-hands",
        "2000",
        "--output-dir",
        str(out),
    ]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "936 information sets" in text
    doc = json.loads((out / "metrics.json").read_text())
    assert doc["tree"]["decision_nodes"] == 3_780
    assert doc["training"]["decision_visits_per_iteration"] == 3_780
    assert doc["checkpoints"][0]["nash_conv"] == pytest.approx(4.747222222222222)
    strategy = load_game_strategy(
        out / doc["checkpoints"][-1]["strategy_file"], "leduc", LEDUC.action_codes
    )
    assert (
        game_strategy_digest(strategy, LEDUC.action_codes)
        == doc["checkpoints"][-1]["strategy_digest"]
    )
    assert doc["memory"]["table_array_bytes"] > 0
    # Reproducible apart from timings.
    assert main([*args[:-1], str(tmp_path / "again")]) == 0
    again = json.loads((tmp_path / "again" / "metrics.json").read_text())
    assert [c["strategy_digest"] for c in again["checkpoints"]] == [
        c["strategy_digest"] for c in doc["checkpoints"]
    ]
    assert again["simulation_check"]["digest"] == doc["simulation_check"]["digest"]


def test_cli_kuhn_generic_engine_matches_legacy(tmp_path: Path) -> None:
    common = ["train-cfr", "--checkpoints", "10", "200", "--simulation-hands", "0"]
    assert main([*common, "--output-dir", str(tmp_path / "legacy")]) == 0
    assert main([*common, "--engine", "generic", "--output-dir", str(tmp_path / "generic")]) == 0
    legacy = json.loads((tmp_path / "legacy" / "metrics.json").read_text())
    generic = json.loads((tmp_path / "generic" / "metrics.json").read_text())
    assert [c["strategy_digest"] for c in legacy["checkpoints"]] == [
        c["strategy_digest"] for c in generic["checkpoints"]
    ]
    assert [c["nash_conv"] for c in legacy["checkpoints"]] == [
        c["nash_conv"] for c in generic["checkpoints"]
    ]
    assert main(["train-cfr", "--game", "leduc", "--engine", "legacy"]) == 2
