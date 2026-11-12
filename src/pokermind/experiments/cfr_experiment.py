"""Reproducible CFR training experiment with exact evaluation at checkpoints.

Outputs go to a run directory (default under ``runs/``, which is git-ignored):

* ``metrics.json``: configuration, environment metadata, every checkpoint's
  metrics and the full average strategy, plus a seeded simulation check of the
  final strategy through the M0 simulator;
* ``metrics.csv``: one row per checkpoint, for quick plotting elsewhere;
* ``strategies/strategy_<iteration>.json``: exported average strategies,
  loadable by :class:`~pokermind.agents.cfr_agent.StrategyAgent`;
* ``convergence.png``: plots (only if Matplotlib is installed).

Vanilla CFR is deterministic, so the seed is only used by the simulation
check. It is recorded either way.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pokermind.agents import RandomAgent, ThresholdAgent
from pokermind.agents.base import Agent
from pokermind.agents.cfr_agent import StrategyAgent
from pokermind.evaluation import (
    GAME_VALUE,
    estimated_alpha,
    expected_value_seat_averaged,
    exploitability_report,
    max_deviation_from_equilibrium,
    uniform_strategy,
)
from pokermind.games.kuhn import Action, KuhnConfig, KuhnState, all_deals, initial_state
from pokermind.learning import CFRSolver
from pokermind.simulation import SimulationConfig, collect_metadata, run_simulation
from pokermind.strategy import Strategy, save_strategy, strategy_digest, strategy_to_jsonable

DEFAULT_CHECKPOINTS: tuple[int, ...] = (1, 10, 100, 500, 1_000, 5_000, 10_000, 50_000)


@dataclass(frozen=True, slots=True)
class CFRExperimentConfig:
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS
    seed: int = 0
    simulation_hands: int = 100_000
    game: KuhnConfig = field(default_factory=KuhnConfig)

    def __post_init__(self) -> None:
        if not self.checkpoints or any(c <= 0 for c in self.checkpoints):
            raise ValueError(f"checkpoints must be positive, got {self.checkpoints}")
        if list(self.checkpoints) != sorted(set(self.checkpoints)):
            raise ValueError(f"checkpoints must be strictly increasing, got {self.checkpoints}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.simulation_hands < 0:
            raise ValueError(f"simulation_hands must be non-negative, got {self.simulation_hands}")


def _walk(state: KuhnState) -> Iterator[KuhnState]:
    yield state
    for action in state.legal_actions():
        yield from _walk(state.apply(action))


def deterministic_agent_strategy(agent: Agent, config: KuhnConfig | None = None) -> Strategy:
    """Tabulate a deterministic agent as a pure strategy.

    Each information set is queried with two different generators; an agent
    whose answer depends on randomness is rejected.
    """
    table: dict[str, dict[Action, float]] = {}
    for cards in all_deals():
        for state in _walk(initial_state(cards, config)):
            if state.is_terminal:
                continue
            player = state.current_player
            info = state.information_state(player)
            legal = state.legal_actions()
            picks = {agent.act(info, legal, np.random.default_rng(s)) for s in (0, 1)}
            if len(picks) != 1:
                raise ValueError(f"agent {agent.name!r} is not deterministic at {info.key!r}")
            (chosen,) = picks
            table[info.key] = {a: (1.0 if a is chosen else 0.0) for a in legal}
    return table


def evaluate_checkpoint(
    strategy: Strategy, baselines: dict[str, Strategy], config: KuhnConfig
) -> dict[str, Any]:
    report = exploitability_report(strategy, config)
    metrics: dict[str, Any] = {
        "nash_conv": report.nash_conv,
        "exploitability": report.exploitability,
        "best_response_values": list(report.best_response_values),
        "self_play_value_seat0": report.on_policy_values[0],
        "expected_payoff_vs": {
            name: expected_value_seat_averaged(strategy, base, config)
            for name, base in baselines.items()
        },
    }
    if config == KuhnConfig():  # the known equilibrium assumes ante = bet = 1
        metrics["game_value_error"] = report.on_policy_values[0] - float(GAME_VALUE)
        metrics["alpha"] = estimated_alpha(strategy)
        metrics["max_deviation_from_equilibrium"] = max_deviation_from_equilibrium(strategy)
    return metrics


def run_cfr_experiment(config: CFRExperimentConfig, output_dir: Path) -> dict[str, Any]:
    """Train to each checkpoint, evaluate exactly, export, and write the artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    game = config.game
    baselines = {
        "random": uniform_strategy(game),
        "rule": deterministic_agent_strategy(ThresholdAgent(), game),
    }

    solver = CFRSolver(game)
    train_seconds = 0.0
    checkpoints: list[dict[str, Any]] = []
    for target in config.checkpoints:
        start = time.perf_counter()
        solver.iterate(target - solver.iterations)
        train_seconds += time.perf_counter() - start

        average = solver.average_strategy()
        start = time.perf_counter()
        metrics = evaluate_checkpoint(average, baselines, game)
        eval_seconds = time.perf_counter() - start

        strategy_path = output_dir / "strategies" / f"strategy_{target:06d}.json"
        save_strategy(strategy_path, average, {"algorithm": "vanilla-cfr", "iterations": target})
        checkpoints.append(
            {
                "iteration": target,
                **metrics,
                "train_seconds_cumulative": train_seconds,
                "eval_seconds": eval_seconds,
                "strategy_digest": strategy_digest(average),
                "strategy_file": str(strategy_path.relative_to(output_dir)),
                "average_strategy": strategy_to_jsonable(average),
            }
        )

    final = solver.average_strategy()
    simulation = _simulation_check(final, baselines, config) if config.simulation_hands else []

    document: dict[str, Any] = {
        "experiment": "kuhn-vanilla-cfr",
        "config": {
            **asdict(config),
            "checkpoints": list(config.checkpoints),
            "algorithm": "vanilla CFR, simultaneous updates, full traversal",
            "exploitability_convention": "NashConv / 2, chips per hand",
        },
        "metadata": collect_metadata(),
        "total_iterations": solver.iterations,
        "train_seconds_total": train_seconds,
        "iterations_per_second": solver.iterations / train_seconds if train_seconds else None,
        "checkpoints": checkpoints,
        "simulation_check": simulation,
    }
    document["metadata"]["matplotlib_version"] = _matplotlib_version()

    (output_dir / "metrics.json").write_text(json.dumps(document, indent=2) + "\n")
    _write_csv(output_dir / "metrics.csv", checkpoints)
    document["plot"] = plot_convergence(checkpoints, output_dir / "convergence.png")
    return document


def _simulation_check(
    final: Strategy, baselines: dict[str, Strategy], config: CFRExperimentConfig
) -> list[dict[str, Any]]:
    """Play the final strategy through the M0 simulator and compare with exact EV."""
    opponents: dict[str, Agent] = {"random": RandomAgent(), "rule": ThresholdAgent()}
    results = []
    for name, opponent in opponents.items():
        sim = run_simulation(
            (StrategyAgent(final), opponent),
            SimulationConfig(num_hands=config.simulation_hands, seed=config.seed, game=config.game),
        )
        exact = expected_value_seat_averaged(final, baselines[name], config.game)
        results.append(
            {
                "opponent": name,
                "hands": sim.num_hands,
                "seed": sim.seed,
                "simulated_mean": sim.mean_returns[0],
                "std_error": sim.std_errors[0],
                "exact_expected": exact,
                "z_score": (sim.mean_returns[0] - exact) / sim.std_errors[0],
                "invalid_actions": list(sim.invalid_actions),
                "digest": sim.digest,
            }
        )
    return results


_CSV_FIELDS = (
    "iteration",
    "exploitability",
    "nash_conv",
    "self_play_value_seat0",
    "game_value_error",
    "alpha",
    "max_deviation_from_equilibrium",
    "train_seconds_cumulative",
    "eval_seconds",
)


def _write_csv(path: Path, checkpoints: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([*_CSV_FIELDS, "ev_vs_random", "ev_vs_rule"])
        for cp in checkpoints:
            writer.writerow(
                [cp.get(f, "") for f in _CSV_FIELDS]
                + [cp["expected_payoff_vs"]["random"], cp["expected_payoff_vs"]["rule"]]
            )


def _matplotlib_version() -> str | None:
    try:
        import matplotlib
    except ImportError:
        return None
    return str(matplotlib.__version__)


# Reference palette slots 1-3 (blue, orange, aqua); grey for reference lines.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
_REFERENCE = "#8a8984"
_TEXT, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def plot_convergence(checkpoints: Sequence[dict[str, Any]], path: Path) -> str | None:
    """Write convergence plots. Returns the path, or ``None`` without Matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    its = [cp["iteration"] for cp in checkpoints]

    def prob(cp: dict[str, Any], key: str, code: str) -> float:
        return float(cp["average_strategy"][key][code])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": _MUTED,
            "axes.labelcolor": _TEXT,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    ax = axes[0]
    expl = [cp["exploitability"] for cp in checkpoints]
    ax.plot(its, expl, color=_SERIES[0], lw=2, marker="o", ms=5, label="exploitability")
    ref = [expl[0] * (its[0] / t) ** 0.5 for t in its]
    ax.plot(its, ref, color=_REFERENCE, lw=1.2, ls="--", label="∝ 1/√T (reference slope)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Exploitability of the average strategy", loc="left", color=_TEXT)
    ax.set_xlabel("CFR iterations")
    ax.set_ylabel("chips per hand (NashConv / 2)")
    ax.annotate(
        f"{expl[-1]:.2e}",
        (its[-1], expl[-1]),
        textcoords="offset points",
        xytext=(-8, 8),
        ha="right",
        color=_TEXT,
    )

    ax = axes[1]
    for color, (key, code, label) in zip(
        _SERIES,
        (("Q|b", "c", "P1 calls a bet with Q"), ("J|x", "b", "P1 bluffs with J")),
        strict=False,
    ):
        ax.plot(its, [prob(cp, key, code) for cp in checkpoints], color=color, lw=2, label=label)
    ax.axhline(1 / 3, color=_REFERENCE, lw=1.2, ls="--", label="equilibrium 1/3")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_title("Player 1 (unique equilibrium)", loc="left", color=_TEXT)
    ax.set_xlabel("CFR iterations")
    ax.set_ylabel("probability")

    ax = axes[2]
    # Each line is (label, information set, action, scale, offset): α = p * scale + offset.
    estimates = (
        ("J| bet", "J|", "b", 1.0, 0.0),
        ("K| bet / 3", "K|", "b", 1 / 3, 0.0),
        ("Q|xb call − 1/3", "Q|xb", "c", 1.0, -1 / 3),
    )
    for color, (label, key, code, scale, offset) in zip(_SERIES, estimates, strict=True):
        values = [prob(cp, key, code) * scale + offset for cp in checkpoints]
        ax.plot(its, values, color=color, lw=2, label=label)
    ax.axhspan(0, 1 / 3, color=_GRID, alpha=0.5, lw=0, label="valid α ∈ [0, 1/3]")
    ax.set_xscale("log")
    ax.set_title("Player 0: three estimates of α should agree", loc="left", color=_TEXT)
    ax.set_xlabel("CFR iterations")
    ax.set_ylabel("implied α")

    for ax in axes:
        ax.grid(True, color=_GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, labelcolor=_TEXT)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return str(path)
