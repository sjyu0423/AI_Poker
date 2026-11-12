"""Command-line interface.

Examples::

    pokermind simulate --agents random rule --hands 100000 --seed 42
    pokermind simulate --agents rule always-bet --hands 1000 --seed 7 --output run.json
    pokermind agents
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pokermind.agents import AGENT_REGISTRY, make_agent
from pokermind.games.kuhn import KuhnConfig
from pokermind.simulation import (
    SimulationConfig,
    SimulationResult,
    collect_metadata,
    run_simulation,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pokermind", description="PokerMind research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run a seeded batch of Kuhn Poker hands")
    sim.add_argument(
        "--agents",
        nargs=2,
        metavar=("AGENT0", "AGENT1"),
        default=["random", "rule"],
        choices=sorted(AGENT_REGISTRY),
        help="two registered agent names (default: random rule)",
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
        return 0

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

    agents = (make_agent(args.agents[0]), make_agent(args.agents[1]))
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
        },
        "result": result.to_dict(),
        "metadata": collect_metadata(["pokermind", *(argv if argv is not None else sys.argv[1:])]),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2) if args.json else format_summary(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
