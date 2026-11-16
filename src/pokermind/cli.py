"""Command-line interface.

Examples::

    pokermind simulate --agents random rule --hands 100000 --seed 42
    pokermind simulate --agents rule always-bet --hands 1000 --seed 7 --output run.json
    pokermind agents
    pokermind train-cfr --checkpoints 100 1000 10000 --output-dir runs/cfr
    pokermind leduc-neural --teacher runs/m4-leduc-teacher/strategies/strategy_100000.json
    pokermind opponent-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json
    pokermind adaptive-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json
    pokermind robust-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json
    pokermind reliability convergence --hero runs/m4-leduc-teacher/strategies/strategy_100000.json
    pokermind research-config --check
    pokermind simulate --agents cfr random --strategy runs/cfr/strategies/strategy_010000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
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
NEURAL_AGENT = "neural"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pokermind", description="PokerMind research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run a seeded batch of Kuhn Poker hands")
    sim.add_argument(
        "--agents",
        nargs=2,
        metavar=("AGENT0", "AGENT1"),
        default=["random", "rule"],
        choices=sorted([*AGENT_REGISTRY, STRATEGY_AGENT, NEURAL_AGENT]),
        help="two registered agent names (default: random rule); "
        "'cfr' needs --strategy, 'neural' needs --model",
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

    sim.add_argument(
        "--model", type=Path, help="neural checkpoint played by any agent named 'neural'"
    )

    neural = sub.add_parser(
        "train-neural", help="distill the CFR strategy into tiny neural policies (PyTorch)"
    )
    neural.add_argument("--epochs", type=int, default=3_000)
    neural.add_argument("--learning-rates", type=float, nargs="+", default=[0.01, 0.03])
    neural.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    neural.add_argument("--architectures", nargs="+", default=["linear", "mlp16", "mlp16x16"])
    neural.add_argument(
        "--teacher", type=Path, help="teacher strategy file (default: re-run M1 CFR)"
    )
    neural.add_argument("--teacher-iterations", type=int, default=50_000)
    neural.add_argument("--simulation-hands", type=int, default=50_000)
    neural.add_argument("--no-benchmark", action="store_true", help="skip latency benchmarks")
    neural.add_argument("--output-dir", type=Path, default=Path("runs/m2a-neural"))

    study = sub.add_parser(
        "neural-study", help="M2B: training-stability, capacity and latency studies"
    )
    study.add_argument(
        "--studies",
        nargs="+",
        choices=["stability", "capacity", "benchmark"],
        default=["stability", "capacity", "benchmark"],
    )
    study.add_argument("--seeds", type=int, default=10, help="seeds 0..N-1 per configuration")
    study.add_argument("--epochs", type=int, default=3_000)
    study.add_argument("--learning-rates", type=float, nargs="+", default=[0.01, 0.03])
    study.add_argument("--teacher-iterations", type=int, default=50_000)
    study.add_argument("--benchmark-number", type=int, default=20_000)
    study.add_argument("--benchmark-repeats", type=int, default=15)
    study.add_argument("--output-dir", type=Path, default=Path("runs/m2b"))

    train = sub.add_parser("train-cfr", help="train vanilla CFR with exact evaluation")
    train.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=None,
        help="increasing iteration counts to evaluate and export (default: kuhn 1 10 100 "
        "500 1000 5000 10000 50000; leduc 1 10 100 500 1000 5000 10000)",
    )
    train.add_argument("--game", choices=["kuhn", "leduc"], default="kuhn")
    train.add_argument(
        "--engine",
        choices=["legacy", "generic"],
        default=None,
        help="kuhn: 'legacy' (M1 code path, default) or 'generic' (M3 game interface); "
        "leduc always uses 'generic'",
    )
    train.add_argument("--seed", type=int, default=0, help="seed for the simulation check")
    train.add_argument(
        "--simulation-hands",
        type=int,
        default=100_000,
        help="hands per simulated match of the final strategy (0 to skip)",
    )
    train.add_argument(
        "--output-dir", type=Path, default=None, help="default: runs/cfr (kuhn), runs/<game>-cfr"
    )

    leduc = sub.add_parser(
        "leduc-neural", help="M4: neural distillation and generalization on Leduc"
    )
    leduc.add_argument(
        "--teacher", type=Path, required=True, help="Leduc CFR strategy file (the teacher)"
    )
    leduc.add_argument(
        "--comparison-teacher", type=Path, default=None, help="an older teacher to compare"
    )
    leduc.add_argument("--seeds", type=int, default=10, help="seeds 0..N-1 per configuration")
    leduc.add_argument("--epochs", type=int, default=5_000)
    leduc.add_argument("--learning-rate", type=float, default=0.01)
    leduc.add_argument(
        "--splits", nargs="+", choices=["full", "random", "public", "history"], default=None
    )
    leduc.add_argument("--simulation-hands", type=int, default=50_000)
    leduc.add_argument("--benchmark-number", type=int, default=5_000)
    leduc.add_argument("--output-dir", type=Path, default=Path("runs/m4-leduc-neural"))

    opp = sub.add_parser(
        "opponent-study", help="M5: Bayesian opponent modelling and sample efficiency on Leduc"
    )
    opp.add_argument("--hero", type=Path, required=True, help="hero's Leduc strategy file")
    opp.add_argument("--seeds", type=int, default=30, help="seeds per held-out opponent")
    opp.add_argument("--tuning-seeds", type=int, default=10, help="seeds per development opponent")
    opp.add_argument("--per-family", type=int, default=4, help="opponents per family and split")
    opp.add_argument("--switching", type=int, default=8, help="switching opponents per split")
    opp.add_argument("--output-dir", type=Path, default=Path("runs/m5-opponent-modeling"))

    adapt = sub.add_parser(
        "adaptive-study", help="M6: opponent-adaptive strategies with exploitability budgets"
    )
    adapt.add_argument("--hero", type=Path, required=True, help="hero's Leduc CFR strategy file")
    adapt.add_argument("--seeds", type=int, default=6, help="seeds per held-out opponent")
    adapt.add_argument("--development-seeds", type=int, default=3)
    adapt.add_argument("--change-seeds", type=int, default=6)
    adapt.add_argument("--workers", type=int, default=0, help="processes (0 = one per CPU)")
    adapt.add_argument("--output-dir", type=Path, default=Path("runs/m6-adaptive-strategy"))

    robust = sub.add_parser(
        "robust-study", help="M7A: RNR and DBR versus the M6 mixture at equal exploitability"
    )
    robust.add_argument("--hero", type=Path, required=True, help="hero's Leduc CFR strategy file")
    robust.add_argument(
        "--m6-metrics",
        type=Path,
        default=Path("runs/m6-adaptive-strategy/metrics.json"),
        help="M6 results, to check that the M6 baselines reproduce",
    )
    robust.add_argument("--seeds", type=int, default=3)
    robust.add_argument("--development-seeds", type=int, default=2)
    robust.add_argument("--change-seeds", type=int, default=3)
    robust.add_argument("--iterations", type=int, default=500, help="CFR+ iterations per solve")
    robust.add_argument("--workers", type=int, default=0, help="processes (0 = one per CPU)")
    robust.add_argument("--output-dir", type=Path, default=Path("runs/m7a-robust-response"))

    rel = sub.add_parser(
        "reliability", help="M7B: convergence, calibration, frozen evaluation and benchmark"
    )
    rel.add_argument(
        "stage",
        choices=("convergence", "calibrate", "evaluate", "benchmark"),
        help="calibrate uses development opponents only; evaluate needs the frozen config",
    )
    rel.add_argument("--hero", type=Path, required=True, help="hero's Leduc CFR strategy file")
    rel.add_argument(
        "--m7a-metrics",
        type=Path,
        default=Path("runs/m7a-robust-response/metrics.json"),
        help="M7A results, for the reproduction checks (optional)",
    )
    rel.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pokermind-research-v1.json"),
        help="frozen research configuration (evaluate, benchmark)",
    )
    rel.add_argument("--iterations", type=int, default=500)
    rel.add_argument("--seeds", type=int, default=3)
    rel.add_argument("--development-seeds", type=int, default=2)
    rel.add_argument("--change-seeds", type=int, default=2)
    rel.add_argument("--bait-count", type=int, default=24)
    rel.add_argument("--workers", type=int, default=0, help="processes (0 = one per CPU)")
    rel.add_argument("--output-dir", type=Path, default=Path("runs/m7b-adaptive-reliability"))

    frz = sub.add_parser(
        "research-config", help="M7B: write or validate the versioned research configuration"
    )
    frz.add_argument("--output", type=Path, default=Path("configs/pokermind-research-v1.json"))
    frz.add_argument(
        "--calibration",
        type=Path,
        help="calibration.json whose frozen proposal sets the selection fields",
    )
    frz.add_argument("--check", action="store_true", help="validate --output and print it")

    serve = sub.add_parser(
        "serve", help="M8A: local API to play Leduc against the adaptive AI (needs [app])"
    )
    serve.add_argument("--host", default="127.0.0.1", help="bind address (local only by default)")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--config", type=Path, default=Path("configs/pokermind-research-v1.json"))
    serve.add_argument("--hero", type=Path, help="CFR hero file (default: the config's path)")
    serve.add_argument("--update-workers", type=int, default=1, help="solver processes")
    serve.add_argument(
        "--recompute-every",
        type=int,
        help="default hands between strategy recomputations for new sessions (default 50)",
    )

    launch = sub.add_parser(
        "launch", help="M9: check the setup, then start the backend and web app together"
    )
    launch.add_argument("--host", default="127.0.0.1", help="bind address (local only by default)")
    launch.add_argument("--api-port", type=int, default=8000)
    launch.add_argument("--web-port", type=int, default=3000)
    launch.add_argument("--config", type=Path, default=Path("configs/pokermind-research-v1.json"))
    launch.add_argument("--hero", type=Path, help="CFR hero file (default: the config's path)")
    launch.add_argument(
        "--recompute-every", type=int, help="hands between strategy updates (default 50)"
    )
    launch.add_argument(
        "--demo",
        action="store_true",
        help="demo pacing: strategy updates every 10 hands (overrides --recompute-every)",
    )
    launch.add_argument("--rebuild", action="store_true", help="force a fresh web build")
    launch.add_argument("--open", action="store_true", help="open the browser when ready")
    launch.add_argument(
        "--train-hero",
        action="store_true",
        help="one-time setup: train the frozen CFR hero (~30 min), then exit",
    )

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
        print(f"{NEURAL_AGENT:<14}neural policy from --model FILE (needs PyTorch)")
        return 0
    if args.command == "train-neural":
        return _train_neural(args)
    if args.command == "neural-study":
        return _neural_study(args)
    if args.command == "train-cfr":
        return _train_cfr(args)
    if args.command == "leduc-neural":
        return _leduc_neural(args)
    if args.command == "opponent-study":
        return _opponent_study(args)
    if args.command == "adaptive-study":
        return _adaptive_study(args)
    if args.command == "robust-study":
        return _robust_study(args)
    if args.command == "reliability":
        return _reliability(args)
    if args.command == "research-config":
        return _research_config(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "launch":
        return _launch(args)

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
    except (OSError, ValueError, RuntimeError) as exc:
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
            "model": None if args.model is None else str(args.model),
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
    if name == NEURAL_AGENT:
        if args.model is None:
            raise ValueError("agent 'neural' requires --model FILE")
        if (args.ante, args.bet_size) != (1, 1):
            raise ValueError("neural policies are trained for ante 1, bet 1 only")
        from pokermind.neural import NeuralPolicyAgent

        return NeuralPolicyAgent.from_checkpoint(args.model)
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


def _train_neural(args: argparse.Namespace) -> int:
    from pokermind.experiments.neural_experiment import (
        NeuralExperimentConfig,
        run_neural_experiment,
    )

    try:
        config = NeuralExperimentConfig(
            architectures=tuple(args.architectures),
            learning_rates=tuple(args.learning_rates),
            seeds=tuple(args.seeds),
            epochs=args.epochs,
            teacher_iterations=args.teacher_iterations,
            teacher_path=None if args.teacher is None else str(args.teacher),
            simulation_hands=args.simulation_hands,
            benchmark=not args.no_benchmark,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    doc = run_neural_experiment(config, args.output_dir)
    t = doc["teacher"]
    print(f"teacher {t['digest'][:12]}: exploitability {t['exploitability']:.6f}")
    print(
        f"{'run':<24}{'params':>7}{'max err':>10}{'mean err':>10}{'mean KL':>11}"
        f"{'exploit.':>11}{'train s':>8}"
    )
    for r in doc["runs"]:
        print(
            f"{r['run_id']:<24}{r['parameter_count']:>7}{r['max_abs_error']:>10.5f}"
            f"{r['mean_abs_error']:>10.5f}{r['mean_kl']:>11.2e}{r['exploitability']:>11.6f}"
            f"{r['train_seconds']:>8.2f}"
        )
    for arch, sel in doc["selected"].items():
        print(f"selected {arch}: {sel['run_id']}")
    print(f"plot: {doc['plot'] or 'skipped (matplotlib not installed)'}")
    return 0


def _neural_study(args: argparse.Namespace) -> int:
    from pokermind.experiments.stability_study import StudyConfig, run_studies

    try:
        if args.seeds <= 0:
            raise ValueError("--seeds must be positive")
        config = StudyConfig(
            seeds=tuple(range(args.seeds)),
            learning_rates=tuple(args.learning_rates),
            epochs=args.epochs,
            teacher_iterations=args.teacher_iterations,
            benchmark_number=args.benchmark_number,
            benchmark_repeats=args.benchmark_repeats,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = run_studies(config, args.output_dir, args.studies)
    if "stability" in out:
        print("stability (pooled over architectures, learning rates and seeds):")
        print(f"  {'procedure':<17}{'median expl.':>14}{'worst expl.':>13}{'degraded':>10}")
        for name, g in out["stability"]["pooled"].items():
            e = g["exploitability"]
            print(
                f"  {name:<17}{e['median']:>14.6f}{e['max']:>13.6f}"
                f"{g['degraded_runs']:>6}/{g['runs']}"
            )
    if "capacity" in out:
        print("capacity:")
        for name, g in out["capacity"]["summary"].items():
            print(
                f"  {name:<14}{g['parameter_count']:>5} params  median expl "
                f"{g['exploitability']['median']:.6f}  median max err "
                f"{g['max_abs_error']['median']:.5f}"
            )
    for name, path in out["plots"].items():
        print(f"plot {name}: {path or 'skipped (matplotlib not installed)'}")
    print(f"results: {args.output_dir}")
    return 0


def _train_cfr(args: argparse.Namespace) -> int:
    engine = args.engine or ("legacy" if args.game == "kuhn" else "generic")
    if args.game == "leduc" and engine == "legacy":
        print("error: leduc requires --engine generic", file=sys.stderr)
        return 2
    if engine == "generic":
        return _train_game_cfr(args)
    if args.output_dir is None:
        args.output_dir = Path("runs/cfr")
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


def _train_game_cfr(args: argparse.Namespace) -> int:
    from pokermind.experiments.game_cfr_experiment import (
        DEFAULT_CHECKPOINTS,
        GameCFRExperimentConfig,
        run_game_cfr_experiment,
    )
    from pokermind.games.base import Game
    from pokermind.games.kuhn_game import KuhnGame
    from pokermind.games.leduc import LeducGame

    game: Game = LeducGame() if args.game == "leduc" else KuhnGame()
    output_dir = args.output_dir or Path(f"runs/{args.game}-cfr")
    try:
        config = GameCFRExperimentConfig(
            checkpoints=tuple(args.checkpoints or DEFAULT_CHECKPOINTS[args.game]),
            seed=args.seed,
            simulation_hands=args.simulation_hands,
            extra={"engine": "generic"},
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    doc = run_game_cfr_experiment(game, config, output_dir)
    tree, tr, mem = doc["tree"], doc["training"], doc["memory"]
    print(
        f"vanilla CFR on {args.game} (generic engine) -> {output_dir}\n"
        f"tree: {tree['chance_nodes']} chance, {tree['decision_nodes']} decision, "
        f"{tree['terminal_nodes']} terminal, {tree['information_sets']} information sets"
    )
    print(
        f"{'iteration':>10}{'exploitability':>16}{'NashConv':>12}{'seat0 value':>13}"
        f"{'EV vs unif':>12}{'train s':>9}{'eval s':>8}"
    )
    for cp in doc["checkpoints"]:
        print(
            f"{cp['iteration']:>10,}{cp['exploitability']:>16.6f}{cp['nash_conv']:>12.6f}"
            f"{cp['self_play_value_seat0']:>+13.6f}{cp['expected_payoff_vs_uniform']:>+12.4f}"
            f"{cp['train_seconds_cumulative']:>9.2f}{cp['eval_seconds']:>8.2f}"
        )
    print(
        f"throughput: {tr['iterations_per_second']:.1f} it/s, "
        f"{tr['microseconds_per_decision_visit']:.2f} us per decision visit"
    )
    print(
        f"memory: tables {mem.get('table_array_bytes', 0):,} B, "
        f"heap peak {mem.get('tracemalloc_peak_bytes', 0):,} B, "
        f"process peak RSS {mem['process_peak_rss_bytes'] / 2**20:.0f} MiB"
    )
    sim = doc["simulation_check"]
    if sim:
        print(
            f"simulated vs uniform: {sim['mean_returns'][0]:+.4f} ± {sim['std_errors'][0]:.4f} "
            f"(exact {sim['exact_expected_agent0']:+.4f}, z={sim['z_score']:+.2f})"
        )
    print(f"plot: {doc['plot'] or 'skipped (matplotlib not installed)'}")
    return 0


def _leduc_neural(args: argparse.Namespace) -> int:
    from pokermind.experiments.leduc_neural_study import (
        LeducNeuralStudyConfig,
        run_leduc_neural_study,
    )
    from pokermind.neural.leduc_dataset import SPLIT_NAMES

    try:
        if args.seeds <= 0:
            raise ValueError("--seeds must be positive")
        if not args.teacher.is_file():
            raise ValueError(f"teacher file not found: {args.teacher}")
        config = LeducNeuralStudyConfig(
            teacher_path=args.teacher,
            comparison_teacher_path=args.comparison_teacher,
            seeds=tuple(range(args.seeds)),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            simulation_hands=args.simulation_hands,
            benchmark_number=args.benchmark_number,
            splits=tuple(args.splits) if args.splits else SPLIT_NAMES,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    doc = run_leduc_neural_study(config, args.output_dir)
    print(f"teacher exploitability {doc['teacher']['policy']['exploitability']:.6f}")
    print(
        f"  {'split':<9}{'variant':<15}{'selection':<11}{'test KL':>9}{'full expl':>11}"
        f"{'hybrid expl':>13}"
    )
    for row in doc["summary"]:
        cells = [
            f"{row[n]['mean']:.4f}" if n in row else "-"
            for n in ("test_kl", "full_exploitability", "hybrid_exploitability")
        ]
        print(
            f"  {row['split']:<9}{row['variant']:<15}{row['selection']:<11}"
            f"{cells[0]:>9}{cells[1]:>11}{cells[2]:>13}"
        )
    print(f"plot: {doc['plot'] or 'skipped (matplotlib not installed)'}")
    print(f"results: {args.output_dir}")
    return 0


def _opponent_study(args: argparse.Namespace) -> int:
    from pokermind.experiments.opponent_study import OpponentStudyConfig, run_opponent_study

    try:
        if not args.hero.is_file():
            raise ValueError(f"hero strategy file not found: {args.hero}")
        config = OpponentStudyConfig(
            hero_path=args.hero,
            seeds=args.seeds,
            tuning_seeds=args.tuning_seeds,
            per_family=args.per_family,
            switching_per_split=args.switching,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    doc = run_opponent_study(config, args.output_dir)
    selected = doc["tuning"]["selected"]["main"]
    print(
        f"selected model: {selected['candidate']} "
        f"(development excess {selected['mean_excess']:.4f})"
    )
    main_rows = doc["sample_efficiency"]["heldout"]["main"]
    print(f"  {'hands':>6}{'excess log loss (95% CI)':>30}{'accuracy':>10}")
    for budget, row in main_rows.items():
        e = row["excess_log_loss"]
        print(
            f"  {budget:>6}{e['mean']:>12.4f} [{e['ci_low']:.4f}, {e['ci_high']:.4f}]"
            f"{row['accuracy']['mean']:>10.3f}"
        )
    det = doc["change"]["detection"]
    print(
        f"change detection: {det['detected']}/{det['runs']} detected, median delay "
        f"{det['delay_median']} hands"
    )
    print(f"plot: {doc['plot'] or 'skipped (matplotlib not installed)'}")
    print(f"results: {args.output_dir}")
    return 0


def _adaptive_study(args: argparse.Namespace) -> int:
    from pokermind.experiments.adaptive_study import AdaptiveStudyConfig, run_adaptive_study

    try:
        if not args.hero.is_file():
            raise ValueError(f"hero strategy file not found: {args.hero}")
        config = AdaptiveStudyConfig(
            hero_path=args.hero,
            seeds=args.seeds,
            development_seeds=args.development_seeds,
            change_seeds=args.change_seeds,
            workers=args.workers,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    doc = run_adaptive_study(config, args.output_dir)
    print(f"tuned on development opponents: {doc['tuning']['chosen']}")
    held = doc["heldout"]["methods"]
    print(f"  {'hands':>6}  {'method':<18}{'gain (95% CI)':>28}{'max expl.':>11}")
    for budget in [str(b) for b in doc["config"]["budgets"]]:
        for method in ("budget_max", "gated", "br_map"):
            row = held[method][budget]
            g = row["gain"]
            print(
                f"  {budget:>6}  {method:<18}{g['mean']:>+10.4f} [{g['ci_low']:+.4f}, "
                f"{g['ci_high']:+.4f}]{row['max_exploitability']:>11.3f}"
            )
    print(f"plot: {doc['plot'] or 'skipped (matplotlib not installed)'}")
    print(f"results: {args.output_dir}")
    return 0


def _robust_study(args: argparse.Namespace) -> int:
    from pokermind.experiments.robust_study import RobustStudyConfig, run_robust_study

    try:
        if not args.hero.is_file():
            raise ValueError(f"hero strategy file not found: {args.hero}")
        config = RobustStudyConfig(
            hero_path=args.hero,
            m6_metrics=args.m6_metrics,
            seeds=args.seeds,
            development_seeds=args.development_seeds,
            change_seeds=args.change_seeds,
            iterations=args.iterations,
            workers=args.workers,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    doc = run_robust_study(config, args.output_dir)
    print(f"DBR threshold tuned on development opponents: {doc['dbr_tuning']['chosen']:g}")
    print(f"M6 reproduction: {doc['m6_reproduction']}")
    gain = doc["heldout"]["gain"]
    print(
        f"  held-out gain over CFR after 100 hands, by budget above CFR ({'/'.join(gain['mix'])}):"
    )
    for family in ("mix", "rnr", "dbr"):
        cells = " ".join(f"{gain[family][d]['100']['mean']:+.4f}" for d in gain[family])
        print(f"    {family:<5}{cells}")
    print(f"plot: {doc['plot'] or 'skipped (matplotlib not installed)'}")
    print(f"results: {args.output_dir}")
    return 0


def _reliability(args: argparse.Namespace) -> int:
    from pokermind.experiments import reliability_study as rs
    from pokermind.research import load_research_config

    try:
        if not args.hero.is_file():
            raise ValueError(f"hero strategy file not found: {args.hero}")
        config = rs.ReliabilityConfig(
            hero_path=args.hero,
            m7a_metrics=args.m7a_metrics,
            iterations=args.iterations,
            seeds=args.seeds,
            development_seeds=args.development_seeds,
            change_seeds=args.change_seeds,
            bait_count=args.bait_count,
            workers=args.workers,
        )
        research = (
            load_research_config(args.config) if args.stage in ("evaluate", "benchmark") else None
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.stage == "convergence":
        doc = rs.run_convergence(config, args.output_dir)
        for cp, gaps in doc["summary"]["gap"].items():
            cells = " ".join(f"p={p}:{g['mean']:.5f}" for p, g in gaps.items())
            print(f"  {cp:>5} iterations, mean RNR-game gap: {cells}")
    elif args.stage == "calibrate":
        doc = rs.run_calibration(config, args.output_dir)
        print(f"calibration: {doc['calibration_choice']}")
        print(f"frozen proposal: {doc['frozen_proposal']}")
    elif args.stage == "evaluate":
        assert research is not None
        doc = rs.run_evaluation(config, research, args.output_dir)
        print(f"M7A reproduction: {doc['m7a_reproduction']}")
        gain = doc["heldout"]["gain"]["frozen"][str(rs.PRIMARY_DELTA)]
        print(
            "  held-out gain (frozen rule): "
            + " ".join(f"{n}:{v['mean']:+.4f}" for n, v in gain.items())
        )
    else:
        assert research is not None
        doc = rs.run_benchmark(config, research, args.output_dir)
        for v, s in doc["summary"].items():
            if "mean_seconds" in s:
                print(f"  {v:<18}{s['mean_seconds']:6.2f} s  x{s['speedup_vs_m7a']:.1f}")
    print(f"results: {args.output_dir} ({doc['elapsed_seconds']:.0f} s)")
    return 0


def _research_config(args: argparse.Namespace) -> int:

    from pokermind.research import (
        ResearchConfig,
        SelectionConfig,
        load_research_config,
        save_research_config,
    )

    try:
        if args.check:
            config = load_research_config(args.output)
            print(json.dumps(config.to_dict(), indent=2))
            print(f"digest: {config.digest()} (valid)")
            return 0
        base = ResearchConfig()
        if args.calibration is not None:
            cal = json.loads(args.calibration.read_text())
            prop = cal["frozen_proposal"]
            base = ResearchConfig(
                selection=SelectionConfig(
                    rule=prop["rule"], z=float(prop["z"]), inflation=float(prop["inflation"])
                ),
                reconstruction=replace(base.reconstruction, temperature=float(prop["temperature"])),
                provenance={
                    "selection": prop,
                    "calibration_choice": cal["calibration_choice"],
                    "source": str(args.calibration),
                    "evidence": "development opponents only",
                },
            )
        save_research_config(base, args.output)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output} (digest {base.digest()})")
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from pokermind.app.api import create_app
    except ImportError:
        print("error: serving needs the app extra: pip install -e '.[app]'", file=sys.stderr)
        return 2
    from pokermind.app.context import ContextError, load_app_context
    from pokermind.app.manager import SessionManager
    from pokermind.app.session import SessionSettings
    from pokermind.app.updates import ProcessBackend

    if args.update_workers <= 0:
        print("error: --update-workers must be positive", file=sys.stderr)
        return 2
    try:
        defaults = (
            SessionSettings()
            if args.recompute_every is None
            else SessionSettings(recompute_every=args.recompute_every)
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        ctx = load_app_context(args.config, args.hero)
    except ContextError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    backend = ProcessBackend(args.config, args.hero, args.update_workers)
    app = create_app(SessionManager(ctx, backend, defaults=defaults))
    print(
        f"PokerMind research {ctx.research.version} ({ctx.config_digest[:12]}), "
        f"exploitability limit {ctx.budget:.4f}, recompute every "
        f"{defaults.recompute_every} hands; http://{args.host}:{args.port}/docs"
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


DEMO_RECOMPUTE_EVERY = 10


def _launch(args: argparse.Namespace) -> int:
    from pokermind.launcher import LaunchConfig, LaunchError, launch, train_hero

    root = Path.cwd()
    if not (root / "pyproject.toml").exists() or not (root / "web").is_dir():
        print("error: run `pokermind launch` from the repository root", file=sys.stderr)
        return 2
    cfg = LaunchConfig(
        root=root,
        host=args.host,
        api_port=args.api_port,
        web_port=args.web_port,
        config=args.config,
        hero=args.hero,
        recompute_every=DEMO_RECOMPUTE_EVERY if args.demo else args.recompute_every,
        rebuild=args.rebuild,
        open_browser=args.open,
    )
    try:
        return train_hero(cfg) if args.train_hero else launch(cfg)
    except LaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
