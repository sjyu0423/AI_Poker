"""Seeded batch simulation of Kuhn Poker between two agents.

Randomness
----------
A single integer seed is expanded with :class:`numpy.random.SeedSequence`
into three independent streams: one for dealing and one per agent. Agent
randomness therefore cannot change the cards that are dealt, and identical
seeds and configurations reproduce identical results.

Seats
-----
Kuhn Poker is asymmetric (player 0 acts first), so by default the two agents
swap seats every hand. Returns are reported per agent and per (agent, seat).

Invalid actions
---------------
If an agent returns an action that is not legal, the runner counts it against
that agent and substitutes the passive legal action (CHECK or FOLD). With
``strict=True`` it raises instead. The engine itself rejects every illegal
transition by raising :class:`~pokermind.games.kuhn.IllegalActionError`, which
the runner does not catch, so a returned result is itself evidence that no
illegal transition occurred.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from pokermind.agents.base import Agent
from pokermind.games.kuhn import KuhnConfig, deal, history_key


class InvalidAgentActionError(RuntimeError):
    """Raised in strict mode when an agent returns an illegal action."""


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    num_hands: int
    seed: int
    game: KuhnConfig = field(default_factory=KuhnConfig)
    alternate_seats: bool = True
    strict: bool = False

    def __post_init__(self) -> None:
        if self.num_hands <= 0:
            raise ValueError(f"num_hands must be positive, got {self.num_hands}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Outcome of a batch. ``deterministic_dict`` excludes timing fields."""

    num_hands: int
    seed: int
    agent_names: tuple[str, str]
    total_returns: tuple[int, int]
    mean_returns: tuple[float, float]
    std_errors: tuple[float, float]
    mean_returns_by_seat: dict[str, dict[str, float | None]]
    hands_by_seat: dict[str, dict[str, int]]
    invalid_actions: tuple[int, int]
    zero_sum_violations: int
    terminal_history_counts: dict[str, int]
    digest: str
    elapsed_seconds: float
    hands_per_second: float

    def deterministic_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        del data["elapsed_seconds"]
        del data["hands_per_second"]
        return data

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, tuple):
                data[key] = list(value)
        return data


def _std_error(total: float, total_sq: float, n: int) -> float:
    if n < 2:
        return math.nan
    mean = total / n
    variance = max(total_sq / n - mean * mean, 0.0) * n / (n - 1)
    return math.sqrt(variance / n)


def run_simulation(agents: tuple[Agent, Agent], config: SimulationConfig) -> SimulationResult:
    """Play ``config.num_hands`` hands between ``agents[0]`` and ``agents[1]``."""
    deck_ss, agent0_ss, agent1_ss = np.random.SeedSequence(config.seed).spawn(3)
    deck_rng = np.random.default_rng(deck_ss)
    agent_rngs = (np.random.default_rng(agent0_ss), np.random.default_rng(agent1_ss))

    totals = [0, 0]
    totals_sq = [0, 0]
    seat_totals = [[0, 0], [0, 0]]  # [agent][seat]
    seat_counts = [[0, 0], [0, 0]]
    invalid = [0, 0]
    zero_sum_violations = 0
    history_counts: Counter[str] = Counter()
    hasher = hashlib.sha256()

    start = time.perf_counter()
    for hand in range(config.num_hands):
        swap = config.alternate_seats and hand % 2 == 1
        seat_to_agent = (1, 0) if swap else (0, 1)

        state = deal(deck_rng, config.game)
        while not state.is_terminal:
            seat = state.current_player
            agent_idx = seat_to_agent[seat]
            legal = state.legal_actions()
            action: Any = agents[agent_idx].act(
                state.information_state(seat), legal, agent_rngs[agent_idx]
            )
            if action not in legal:
                invalid[agent_idx] += 1
                if config.strict:
                    raise InvalidAgentActionError(
                        f"agent {agents[agent_idx].name!r} returned {action!r} at "
                        f"{history_key(state.history)!r}; legal: {legal}"
                    )
                action = legal[0]  # passive action: CHECK or FOLD
            state = state.apply(action)  # raises IllegalActionError; deliberately uncaught

        payoffs = state.payoffs()
        if payoffs[0] + payoffs[1] != 0:  # pragma: no cover - defensive
            zero_sum_violations += 1

        for seat in (0, 1):
            agent_idx = seat_to_agent[seat]
            value = payoffs[seat]
            totals[agent_idx] += value
            totals_sq[agent_idx] += value * value
            seat_totals[agent_idx][seat] += value
            seat_counts[agent_idx][seat] += 1

        key = history_key(state.history)
        history_counts[key] += 1
        hasher.update(
            f"{hand},{seat_to_agent[0]},{state.cards[0].name}{state.cards[1].name},"
            f"{key},{payoffs[0]};".encode()
        )
    elapsed = time.perf_counter() - start

    n = config.num_hands
    names = (agents[0].name, agents[1].name)
    by_seat: dict[str, dict[str, float | None]] = {}
    counts_by_seat: dict[str, dict[str, int]] = {}
    for a in (0, 1):
        label = f"agent{a}"
        by_seat[label] = {
            f"seat{s}": (seat_totals[a][s] / seat_counts[a][s] if seat_counts[a][s] else None)
            for s in (0, 1)
        }
        counts_by_seat[label] = {f"seat{s}": seat_counts[a][s] for s in (0, 1)}

    return SimulationResult(
        num_hands=n,
        seed=config.seed,
        agent_names=names,
        total_returns=(totals[0], totals[1]),
        mean_returns=(totals[0] / n, totals[1] / n),
        std_errors=(
            _std_error(totals[0], totals_sq[0], n),
            _std_error(totals[1], totals_sq[1], n),
        ),
        mean_returns_by_seat=by_seat,
        hands_by_seat=counts_by_seat,
        invalid_actions=(invalid[0], invalid[1]),
        zero_sum_violations=zero_sum_violations,
        terminal_history_counts=dict(sorted(history_counts.items())),
        digest=hasher.hexdigest(),
        elapsed_seconds=elapsed,
        hands_per_second=n / elapsed if elapsed > 0 else math.inf,
    )


__all__ = [
    "InvalidAgentActionError",
    "SimulationConfig",
    "SimulationResult",
    "run_simulation",
]
