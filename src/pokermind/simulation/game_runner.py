"""Seeded simulation for any ``Game`` (M3).

The M0 runner (``simulation/runner.py``) stays unchanged for Kuhn, because its
dealing uses ``rng.permutation``. This runner instead samples each chance
node's listed outcomes with one uniform draw inverted through the CDF. That
works for any game but produces different hands from the M0 runner for the
same seed.

Streams: ``SeedSequence(seed).spawn(3)`` gives chance, agent 0 and agent 1,
as in M0, so agents' randomness can't change what is dealt. Seats alternate
each hand by default. Agents receive only ``state.observation(seat)`` and the
legal actions.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from pokermind.agents.cfr_agent import sample_action
from pokermind.games.base import Game, GameState, Observation


class GameAgent(Protocol):
    @property
    def name(self) -> str: ...

    def act(
        self, observation: Observation, legal_actions: tuple[int, ...], rng: np.random.Generator
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class GameSimulationConfig:
    num_hands: int
    seed: int
    alternate_seats: bool = True

    def __post_init__(self) -> None:
        if self.num_hands <= 0:
            raise ValueError(f"num_hands must be positive, got {self.num_hands}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


def sample_chance(state: GameState, rng: np.random.Generator) -> int:
    outcomes = state.chance_outcomes()
    return sample_action([o for o, _ in outcomes], [p for _, p in outcomes], rng, "chance")


def run_game_simulation(
    game: Game, agents: tuple[GameAgent, GameAgent], config: GameSimulationConfig
) -> dict[str, Any]:
    chance_ss, a0_ss, a1_ss = np.random.SeedSequence(config.seed).spawn(3)
    chance_rng = np.random.default_rng(chance_ss)
    rngs = (np.random.default_rng(a0_ss), np.random.default_rng(a1_ss))
    totals, totals_sq = [0.0, 0.0], [0.0, 0.0]
    zero_sum_violations = 0
    hasher = hashlib.sha256()
    start = time.perf_counter()
    for hand in range(config.num_hands):
        seat_to_agent = (1, 0) if config.alternate_seats and hand % 2 == 1 else (0, 1)
        state = game.initial_state()
        moves: list[int] = []
        while not state.is_terminal:
            if state.is_chance_node:
                move = sample_chance(state, chance_rng)
            else:
                seat = state.current_player
                idx = seat_to_agent[seat]
                legal = state.legal_actions()
                move = agents[idx].act(state.observation(seat), legal, rngs[idx])
                if move not in legal:
                    raise ValueError(f"agent {agents[idx].name!r} chose illegal action {move!r}")
            moves.append(move)
            state = state.apply(move)
        returns = state.returns()
        if abs(returns[0] + returns[1]) > 1e-12:
            zero_sum_violations += 1
        for seat in (0, 1):
            idx = seat_to_agent[seat]
            totals[idx] += returns[seat]
            totals_sq[idx] += returns[seat] ** 2
        hasher.update(f"{hand},{seat_to_agent[0]},{moves},{returns[0]!r};".encode())
    elapsed = time.perf_counter() - start
    n = config.num_hands

    def std_error(i: int) -> float:
        mean = totals[i] / n
        var = max(totals_sq[i] / n - mean * mean, 0.0) * n / (n - 1) if n > 1 else math.nan
        return math.sqrt(var / n)

    return {
        "game": game.name,
        "num_hands": n,
        "seed": config.seed,
        "agent_names": [agents[0].name, agents[1].name],
        "mean_returns": [totals[0] / n, totals[1] / n],
        "std_errors": [std_error(0), std_error(1)],
        "zero_sum_violations": zero_sum_violations,
        "digest": hasher.hexdigest(),
        "elapsed_seconds": elapsed,
        "hands_per_second": n / elapsed if elapsed > 0 else math.inf,
    }
