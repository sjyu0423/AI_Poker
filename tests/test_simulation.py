"""Batch simulation: correctness at scale, reproducibility, invalid actions."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pokermind.agents import (
    AGENT_REGISTRY,
    Agent,
    AlwaysBetAgent,
    AlwaysPassAgent,
    RandomAgent,
    ThresholdAgent,
    make_agent,
)
from pokermind.games.kuhn import (
    Action,
    Card,
    InfoState,
    KuhnConfig,
    KuhnState,
    all_deals,
    deal,
    initial_state,
)
from pokermind.simulation import (
    InvalidAgentActionError,
    SimulationConfig,
    run_simulation,
)


def test_100k_hands_complete_without_engine_errors() -> None:
    # The engine raises IllegalActionError on any illegal transition and the
    # runner does not catch it, so completing the batch is the check.
    result = run_simulation(
        (RandomAgent(), RandomAgent()), SimulationConfig(num_hands=100_000, seed=2026)
    )
    assert result.num_hands == 100_000
    assert result.zero_sum_violations == 0
    assert result.invalid_actions == (0, 0)
    assert sum(result.total_returns) == 0
    assert sum(result.terminal_history_counts.values()) == 100_000
    assert set(result.terminal_history_counts) == {"xx", "bf", "bc", "xbf", "xbc"}


@pytest.mark.parametrize("pair", list(itertools.product(sorted(AGENT_REGISTRY), repeat=2)))
def test_all_agent_pairs_zero_sum(pair: tuple[str, str]) -> None:
    agents = (make_agent(pair[0]), make_agent(pair[1]))
    result = run_simulation(agents, SimulationConfig(num_hands=2_000, seed=1))
    assert result.invalid_actions == (0, 0)
    assert result.total_returns[0] == -result.total_returns[1]


def test_identical_seeds_reproduce_identical_results() -> None:
    cfg = SimulationConfig(num_hands=20_000, seed=777)
    a = run_simulation((RandomAgent(), ThresholdAgent()), cfg)
    b = run_simulation((RandomAgent(), ThresholdAgent()), cfg)
    assert a.deterministic_dict() == b.deterministic_dict()
    assert a.digest == b.digest


def test_different_seeds_differ() -> None:
    a = run_simulation((RandomAgent(), RandomAgent()), SimulationConfig(num_hands=5_000, seed=1))
    b = run_simulation((RandomAgent(), RandomAgent()), SimulationConfig(num_hands=5_000, seed=2))
    assert a.digest != b.digest


def test_deal_stream_independent_of_agent_randomness() -> None:
    """The cards dealt must not depend on how much randomness agents consume."""

    class OpeningCardRecorder:
        """Records seat 0's card at the first decision of each hand."""

        def __init__(self, inner: Agent) -> None:
            self.inner = inner
            self.name = inner.name
            self.cards: list[Card] = []

        def act(
            self, info_state: InfoState, legal: tuple[Action, ...], rng: np.random.Generator
        ) -> Action:
            if info_state.history == ():
                self.cards.append(info_state.private_card)
            return self.inner.act(info_state, legal, rng)

    n = 3_000
    cfg = SimulationConfig(num_hands=n, seed=99, alternate_seats=False)
    random_rec = OpeningCardRecorder(RandomAgent())
    run_simulation((random_rec, RandomAgent()), cfg)
    passive_rec = OpeningCardRecorder(AlwaysPassAgent())
    run_simulation((passive_rec, AlwaysBetAgent()), cfg)

    deck_rng = np.random.default_rng(np.random.SeedSequence(99).spawn(3)[0])
    expected = [deal(deck_rng).cards[0] for _ in range(n)]
    assert random_rec.cards == passive_rec.cards == expected


def test_seat_alternation() -> None:
    r = run_simulation((RandomAgent(), RandomAgent()), SimulationConfig(num_hands=1_001, seed=3))
    assert r.hands_by_seat == {
        "agent0": {"seat0": 501, "seat1": 500},
        "agent1": {"seat0": 500, "seat1": 501},
    }
    r = run_simulation(
        (RandomAgent(), RandomAgent()),
        SimulationConfig(num_hands=100, seed=3, alternate_seats=False),
    )
    assert r.hands_by_seat["agent0"] == {"seat0": 100, "seat1": 0}
    assert r.mean_returns_by_seat["agent0"]["seat1"] is None


class BadAgent:
    """Always tries to CALL, which is illegal when not facing a bet."""

    name = "bad"

    def act(
        self, info_state: InfoState, legal: tuple[Action, ...], rng: np.random.Generator
    ) -> Action:
        return Action.CALL


def test_invalid_actions_counted_and_replaced() -> None:
    cfg = SimulationConfig(num_hands=1_000, seed=4, alternate_seats=False)
    r = run_simulation((BadAgent(), AlwaysPassAgent()), cfg)
    # In seat 0 the bad agent opens every hand: illegal CALL -> substituted CHECK.
    assert r.invalid_actions == (1_000, 0)
    assert r.terminal_history_counts == {"xx": 1_000}


def test_strict_mode_raises() -> None:
    cfg = SimulationConfig(num_hands=10, seed=4, strict=True)
    with pytest.raises(InvalidAgentActionError):
        run_simulation((BadAgent(), RandomAgent()), cfg)


@pytest.mark.parametrize(("num_hands", "seed"), [(0, 0), (-5, 0), (1, -1)])
def test_invalid_simulation_config(num_hands: int, seed: int) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(num_hands=num_hands, seed=seed)


def _exact_value(state: KuhnState, policies: tuple[dict[str, dict[Action, float]], ...]) -> float:
    """Exact expected payoff for seat 0 by full tree traversal."""
    if state.is_terminal:
        return float(state.payoffs()[0])
    p = state.current_player
    dist = policies[p][state.information_state(p).key]
    return sum(prob * _exact_value(state.apply(a), policies) for a, prob in dist.items())


def _tabulate(agent: Agent) -> dict[str, dict[Action, float]]:
    """Policy table for deterministic agents, uniform for RandomAgent."""
    table: dict[str, dict[Action, float]] = {}
    rng = np.random.default_rng(0)

    def visit(state: KuhnState) -> None:
        if state.is_terminal:
            return
        p = state.current_player
        info = state.information_state(p)
        legal = state.legal_actions()
        if isinstance(agent, RandomAgent):
            table[info.key] = {a: 1 / len(legal) for a in legal}
        else:
            table[info.key] = {agent.act(info, legal, rng): 1.0}
        for a in legal:
            visit(state.apply(a))

    for cards in all_deals():
        visit(initial_state(cards))
    return table


def test_simulated_returns_match_exact_expectation() -> None:
    """Simulation must agree with an exact game-tree computation (within 4 s.e.)."""
    a0, a1 = RandomAgent(), ThresholdAgent()
    t0, t1 = _tabulate(a0), _tabulate(a1)
    exact_seat0 = sum(_exact_value(initial_state(c), (t0, t1)) for c in all_deals()) / 6
    exact_seat1 = -sum(_exact_value(initial_state(c), (t1, t0)) for c in all_deals()) / 6
    exact_agent0 = (exact_seat0 + exact_seat1) / 2

    r = run_simulation((a0, a1), SimulationConfig(num_hands=100_000, seed=11))
    assert abs(r.mean_returns[0] - exact_agent0) < 4 * r.std_errors[0]


def test_config_scaling_in_simulation() -> None:
    cfg = SimulationConfig(num_hands=500, seed=8, game=KuhnConfig(ante=2, bet_size=3))
    r = run_simulation((AlwaysBetAgent(), AlwaysBetAgent()), cfg)
    assert r.terminal_history_counts == {"bc": 500}
    # Every hand is a showdown for 5 chips.
    assert r.std_errors[0] > 0
    assert abs(r.total_returns[0]) % 5 == 0
