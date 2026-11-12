"""Command-line interface.

Examples::

    pokermind simulate --agents random rule --hands 100000 --seed 42
    pokermind simulate --agents rule always-bet --hands 1000 --seed 7 --output run.json
    pokermind agents
    pokermind train-cfr --checkpoints 100 1000 10000 --output-dir runs/cfr
    pokermind simulate --agents cfr random --strategy runs/cfr/strategies/strategy_010000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pokermind.agents import AGENT_REGISTRY, make_agent
from pokermind.agents.base import Agent
from pokermind.agents.cfr_agent import StrategyAgent
from pokermind.games.kuhn import KuhnConfig
from pokermind.simulation import (
    SimulationConfig,
    SimulationResult,
    collect_metadata,
    run_simulation,
)
from pokermind.strategy import StrategyFormatError

STRATEGY_AGENT = "cfr"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pokermind", description="PokerMind research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run a seeded batch of Kuhn Poker hands")
    sim.add_argument(
        "--agents",
        nargs=2,
        metavar=("AGENT0", "AGENT1"),
        default=["random", "rule"],
        choices=sorted([*AGENT_REGISTRY, STRATEGY_AGENT]),
        help="two registered agent names (default: random rule); 'cfr' needs --strategy",
    )
    sim.add_argument("--hands", type=int, default=10_000, help="number of hands")
    sim.add_argument("--seed", type=int, default=0, help="non-negative integer seed")
    sim.add_argument("--ante", type=int, default=1)
    sim.add_argument("--bet-size", type=int, default=1)
    sim.add_argument(
        "--no-alternate-seats",
        action="store_true",
        help="keep agent0 in seat 0 for every hand",
    )
    sim.add_argument("--strict", action="store_true", help="raise on invalid agent actions")
    sim.add_argument("--output", type=Path, help="write full JSON results to this path")
    sim.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    sim.add_argument("--strategy", type=Path, help="strategy file played by any agent named 'cfr'")

    train = sub.add_parser("train-cfr", help="train vanilla CFR with exact evaluation")
    train.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=None,
        help="increasing iteration counts to evaluate and export "
        "(default: 1 10 100 500 1000 5000 10000 50000)",
    )
    train.add_argument("--seed", type=int, default=0, help="seed for the simulation check")
    train.add_argument(
        "--simulation-hands",
        type=int,
        default=100_000,
        help="hands per simulated match of the final strategy (0 to skip)",
    )
    train.add_argument("--output-dir", type=Path, default=Path("runs/cfr"))

    sub.add_parser("agents", help="list registered agents")
    return parser


def format_summary(result: SimulationResult) -> str:
    lines = [
        f"Kuhn Poker: {result.num_hands:,} hands, seed={result.seed}",
        "",
        f"{'agent':<34}{'mean/hand':>11}{'± s.e.':>9}{'total':>9}{'invalid':>9}",
    ]
    for i in (0, 1):
        lines.append(
            f"{i}: {result.agent_names[i]:<31}{result.mean_returns[i]:>+11.4f}"
            f"{result.std_errors[i]:>9.4f}{result.total_returns[i]:>+9d}"
            f"{result.invalid_actions[i]:>9d}"
        )
    lines.append("")
    for i in (0, 1):
        seats = result.mean_returns_by_seat[f"agent{i}"]
        parts = [
            f"seat{s}: {'n/a' if v is None else f'{v:+.4f}'}"
            for s, v in ((0, seats["seat0"]), (1, seats["seat1"]))
        ]
        lines.append(f"agent{i} by seat  " + "   ".join(parts))
    lines += [
        "",
        "terminal histories: "
        + ", ".join(f"{k}={v}" for k, v in result.terminal_history_counts.items()),
        f"zero-sum violations: {result.zero_sum_violations}",
        f"elapsed: {result.elapsed_seconds:.3f}s ({result.hands_per_second:,.0f} hands/s)",
        f"result digest: {result.digest}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "agents":
        for name in sorted(AGENT_REGISTRY):
            print(f"{name:<14}{make_agent(name).name}")
        print(f"{STRATEGY_AGENT:<14}tabular strategy from --strategy FILE (e.g. a CFR export)")
        return 0
    if args.command == "train-cfr":
        return _train_cfr(args)

    try:
        game = KuhnConfig(ante=args.ante, bet_size=args.bet_size)
        config = SimulationConfig(
            num_hands=args.hands,
            seed=args.seed,
            game=game,
            alternate_seats=not args.no_alternate_seats,
            strict=args.strict,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        agents = (_build_agent(args.agents[0], args), _build_agent(args.agents[1], args))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = run_simulation(agents, config)

    report = {
        "config": {
            "game": "kuhn",
            "agents": list(args.agents),
            "num_hands": config.num_hands,
            "seed": config.seed,
            "ante": game.ante,
            "bet_size": game.bet_size,
            "alternate_seats": config.alternate_seats,
            "strict": config.strict,
            "strategy": None if args.strategy is None else str(args.strategy),
        },
        "result": result.to_dict(),
        "metadata": collect_metadata(["pokermind", *(argv if argv is not None else sys.argv[1:])]),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2) if args.json else format_summary(result))
    return 0


def _build_agent(name: str, args: argparse.Namespace) -> Agent:
    if name != STRATEGY_AGENT:
        return make_agent(name)
    if args.strategy is None:
        raise ValueError("agent 'cfr' requires --strategy FILE")
    from pokermind.evaluation import validate_strategy

    agent = StrategyAgent.from_file(args.strategy)
    try:
        validate_strategy(agent.strategy, (0, 1), config=KuhnConfig(args.ante, args.bet_size))
    except ValueError as exc:
        raise StrategyFormatError(f"{args.strategy}: {exc}") from None
    return agent


def _train_cfr(args: argparse.Namespace) -> int:
    from pokermind.experiments import DEFAULT_CHECKPOINTS, CFRExperimentConfig, run_cfr_experiment

    try:
        config = CFRExperimentConfig(
            checkpoints=tuple(args.checkpoints) if args.checkpoints else DEFAULT_CHECKPOINTS,
            seed=args.seed,
            simulation_hands=args.simulation_hands,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    doc = run_cfr_experiment(config, args.output_dir)
    print(f"vanilla CFR on Kuhn Poker -> {args.output_dir}")
    print(
        f"{'iteration':>10}{'exploitability':>16}{'NashConv':>12}{'seat0 value':>13}{'train s':>9}"
    )
    for cp in doc["checkpoints"]:
        print(
            f"{cp['iteration']:>10,}{cp['exploitability']:>16.6f}{cp['nash_conv']:>12.6f}"
            f"{cp['self_play_value_seat0']:>+13.6f}{cp['train_seconds_cumulative']:>9.2f}"
        )
    for sim in doc["simulation_check"]:
        print(
            f"simulated vs {sim['opponent']}: {sim['simulated_mean']:+.4f} ± {sim['std_error']:.4f}"
            f" (exact {sim['exact_expected']:+.4f}, z={sim['z_score']:+.2f})"
        )
    print(f"plot: {doc['plot'] or 'skipped (matplotlib not installed)'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
