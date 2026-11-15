"""CFR experiments through the generic game interface (M3): Kuhn or Leduc.

Per checkpoint the runner records exact NashConv and exploitability, the
self-play value, the seat-averaged expected payoff against the uniform
policy, cumulative training time, evaluation time and the strategy file
size. Separately it measures:

* throughput (iterations and decision-node visits per second);
* memory: bytes held by the regret and strategy arrays; Python heap peak from
  ``tracemalloc`` during a few separate iterations (tracemalloc slows Python
  down, so it never wraps the timed training); the process's peak resident
  memory (``ru_maxrss``);
* a seeded simulation of the final strategy against a uniform agent, compared
  with the exact expected value.

Vanilla CFR is deterministic. The seed only drives the simulation check.
Artifacts go to a run directory (default under the git-ignored ``runs/``).
"""

from __future__ import annotations

import csv
import json
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pokermind.agents.game_agents import GameStrategyAgent, UniformGameAgent
from pokermind.evaluation.game_exact import (
    expected_value_seat_averaged,
    exploitability_report,
    information_sets,
    tree_counts,
    uniform_game_strategy,
)
from pokermind.games.base import Game
from pokermind.learning.game_cfr import GameCFRSolver
from pokermind.simulation import collect_metadata
from pokermind.simulation.game_runner import GameSimulationConfig, run_game_simulation
from pokermind.strategy import game_strategy_digest, save_game_strategy

DEFAULT_CHECKPOINTS: dict[str, tuple[int, ...]] = {
    "kuhn": (1, 10, 100, 500, 1_000, 5_000, 10_000, 50_000),
    "leduc": (1, 10, 100, 500, 1_000, 5_000, 10_000),
}


@dataclass(frozen=True, slots=True)
class GameCFRExperimentConfig:
    checkpoints: tuple[int, ...]
    seed: int = 0
    simulation_hands: int = 50_000
    memory_probe_iterations: int = 3
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoints or any(c <= 0 for c in self.checkpoints):
            raise ValueError(f"checkpoints must be positive, got {self.checkpoints}")
        if list(self.checkpoints) != sorted(set(self.checkpoints)):
            raise ValueError(f"checkpoints must be strictly increasing, got {self.checkpoints}")
        if self.seed < 0 or self.simulation_hands < 0 or self.memory_probe_iterations < 0:
            raise ValueError("seed, simulation_hands and memory_probe_iterations must be >= 0")


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # macOS: bytes; Linux: KiB


def measure_memory(game: Game, iterations: int) -> dict[str, Any]:
    """Heap peak of building a solver and running a few iterations (untimed)."""
    tracemalloc.start()
    try:
        solver = GameCFRSolver(game)
        solver.iterate(iterations)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "probe_iterations": iterations,
        "tracemalloc_peak_bytes": peak,
        "table_array_bytes": solver.table_bytes(),
        "information_sets": len(solver.tables),
    }


def run_game_cfr_experiment(
    game: Game, config: GameCFRExperimentConfig, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    infosets = information_sets(game)
    counts = tree_counts(game)
    setup_seconds = time.perf_counter() - start
    uniform = uniform_game_strategy(game, infosets)

    solver = GameCFRSolver(game)
    train_seconds = 0.0
    checkpoints: list[dict[str, Any]] = []
    for target in config.checkpoints:
        visits_before = solver.decision_visits
        start = time.perf_counter()
        solver.iterate(target - solver.iterations)
        elapsed = time.perf_counter() - start
        train_seconds += elapsed
        average = solver.average_strategy()

        start = time.perf_counter()
        report = exploitability_report(game, average, infosets=infosets)
        ev_uniform = expected_value_seat_averaged(game, average, uniform)
        eval_seconds = time.perf_counter() - start

        path = output_dir / "strategies" / f"strategy_{target:06d}.json"
        save_game_strategy(
            path,
            game.name,
            game.action_codes,
            average,
            {"algorithm": "vanilla-cfr", "iterations": target},
        )
        new_visits = solver.decision_visits - visits_before
        checkpoints.append(
            {
                "iteration": target,
                "exploitability": report.exploitability,
                "nash_conv": report.nash_conv,
                "best_response_values": list(report.best_response_values),
                "self_play_value_seat0": report.on_policy_values[0],
                "expected_payoff_vs_uniform": ev_uniform,
                "train_seconds_cumulative": train_seconds,
                "iterations_per_second_segment": (
                    (target - (checkpoints[-1]["iteration"] if checkpoints else 0)) / elapsed
                    if elapsed > 0
                    else None
                ),
                "decision_visits_segment": new_visits,
                "eval_seconds": eval_seconds,
                "strategy_file_bytes": path.stat().st_size,
                "strategy_digest": game_strategy_digest(average, game.action_codes),
                "strategy_file": str(path.relative_to(output_dir)),
            }
        )

    final = solver.average_strategy()
    simulation: dict[str, Any] | None = None
    if config.simulation_hands:
        sim = run_game_simulation(
            game,
            (GameStrategyAgent(final), UniformGameAgent()),
            GameSimulationConfig(num_hands=config.simulation_hands, seed=config.seed),
        )
        exact = expected_value_seat_averaged(game, final, uniform)
        simulation = {
            **sim,
            "exact_expected_agent0": exact,
            "z_score": (sim["mean_returns"][0] - exact) / sim["std_errors"][0],
        }

    document: dict[str, Any] = {
        "experiment": f"{game.name}-vanilla-cfr-generic",
        "game": game.name,
        "config": {
            "checkpoints": list(config.checkpoints),
            "seed": config.seed,
            "simulation_hands": config.simulation_hands,
            "algorithm": "vanilla CFR, simultaneous updates, full traversal (generic engine)",
            "exploitability_convention": "NashConv / 2, chips per hand",
            **config.extra,
        },
        "metadata": collect_metadata(),
        "tree": {
            "chance_nodes": counts.chance,
            "decision_nodes": counts.decision,
            "terminal_nodes": counts.terminal,
            "information_sets": len(infosets),
            "setup_seconds": setup_seconds,
        },
        "training": {
            "total_iterations": solver.iterations,
            "train_seconds_total": train_seconds,
            "iterations_per_second": solver.iterations / train_seconds if train_seconds else None,
            "decision_visits_per_iteration": solver.decision_visits / solver.iterations,
            "microseconds_per_decision_visit": (
                1e6 * train_seconds / solver.decision_visits if solver.decision_visits else None
            ),
            "information_sets_visited": sum(
                1 for t in solver.tables.values() if t.strategy_sum.sum() > 0
            ),
        },
        "memory": {
            **(
                measure_memory(game, config.memory_probe_iterations)
                if config.memory_probe_iterations
                else {}
            ),
            "process_peak_rss_bytes": _peak_rss_bytes(),
        },
        "checkpoints": checkpoints,
        "simulation_check": simulation,
    }
    (output_dir / "metrics.json").write_text(json.dumps(document, indent=2) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        cols = (
            "iteration",
            "exploitability",
            "nash_conv",
            "self_play_value_seat0",
            "expected_payoff_vs_uniform",
            "train_seconds_cumulative",
            "eval_seconds",
            "strategy_file_bytes",
        )
        writer.writerow(cols)
        for cp in checkpoints:
            writer.writerow([cp[c] for c in cols])
    document["plot"] = _plot(game.name, checkpoints, output_dir / "convergence.png")
    return document


def _plot(name: str, checkpoints: list[dict[str, Any]], path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    its = [c["iteration"] for c in checkpoints]
    expl = [c["exploitability"] for c in checkpoints]
    fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
    ax.plot(its, expl, color="#2a78d6", lw=2, marker="o", ms=5, label="exploitability")
    ax.plot(
        its,
        [expl[0] * (its[0] / t) ** 0.5 for t in its],
        color="#8a8984",
        ls="--",
        lw=1.2,
        label="∝ 1/√T (reference slope)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{name.capitalize()}: exploitability of the average strategy", loc="left")
    ax.set_xlabel("CFR iterations")
    ax.set_ylabel("chips per hand (NashConv / 2)")
    ax.grid(True, color="#e6e5e1", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return str(path)
