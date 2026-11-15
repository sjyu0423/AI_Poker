"""M4: can small networks represent and generalize a Leduc CFR policy?

For every split (``neural/leduc_dataset.py``), architecture and seed, a
network is trained on the training states with the M2B procedure (Adam,
cosine decay, best checkpoint) and evaluated on:

* **distribution metrics** per part (train / validation / test): masked
  cross-entropy, KL(p ‖ q) with ``q`` floored at ``1e-12`` (and a count of the
  floored entries), total variation, mean and max absolute probability error,
  and argmax agreement;
* **exact policy metrics** of the *complete* predicted policy (the network on
  all 936 information sets): exploitability and seat-averaged expected payoff
  against reference opponents (uniform, always-call, always-raise, the teacher);
* a **hybrid** policy for held-out splits: the teacher on the states the
  model was fitted on, the model on the test states. Its exploitability
  isolates the cost of the held-out predictions.

Two checkpoint rules are compared on held-out splits: ``validation`` (lowest
validation KL; never looks at test data) and ``final`` (last epoch). The
``full`` split has no held-out data and uses ``train_best`` (lowest training
KL, as in M2B).

Non-neural baselines fill the test states from the training states only:
``uniform``; ``nearest`` (teacher distribution of the L1-nearest fitted state
with the same legal actions, ties averaged); ``history_mean`` (mean teacher
distribution of fitted states with the same betting history and legal
actions, else uniform).

Everything except timings is deterministic. Artifacts go to a run directory
(default ``runs/m4-leduc-neural``, git-ignored).
"""

from __future__ import annotations

import csv
import json
import math
import resource
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

from pokermind.agents.game_agents import GameStrategyAgent, UniformGameAgent
from pokermind.evaluation.game_exact import (
    GameInfoSet,
    expected_value,
    exploitability_report,
    information_sets,
)
from pokermind.experiments.latency import time_statement
from pokermind.games.leduc import CALL, RAISE, LeducGame
from pokermind.neural.game_model import (
    GamePolicyConfig,
    GamePolicyNetwork,
    checkpoint_bytes,
    legal_probabilities,
    save_checkpoint,
)
from pokermind.neural.game_training import Arrays, train_game_policy
from pokermind.neural.leduc_agent import LeducNeuralAgent
from pokermind.neural.leduc_dataset import (
    SPLIT_NAMES,
    LeducDataset,
    Split,
    build_leduc_dataset,
    make_split,
    split_coverage,
)
from pokermind.neural.leduc_encoding import NUM_ACTIONS, NUM_FEATURES, encode_observation
from pokermind.neural.training import TrainingConfig, state_dict_digest
from pokermind.simulation import collect_metadata
from pokermind.simulation.game_runner import GameSimulationConfig, run_game_simulation
from pokermind.strategy import GameStrategy, game_strategy_digest, load_game_strategy

ARCHITECTURES: Final[dict[str, tuple[int, ...]]] = {
    "linear": (),
    "mlp16": (16,),
    "mlp32": (32,),
    "mlp64": (64,),
}
PROBABILITY_FLOOR: Final = 1e-12
Row = npt.NDArray[np.float64]
Table = dict[str, dict[int, float]]


@dataclass(frozen=True, slots=True)
class LeducNeuralStudyConfig:
    teacher_path: Path
    comparison_teacher_path: Path | None = None
    """An older teacher (M3's 10k) to compare against; optional."""
    architectures: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: dict(ARCHITECTURES)
    )
    seeds: tuple[int, ...] = tuple(range(10))
    splits: tuple[str, ...] = SPLIT_NAMES
    split_seed: int = 0
    epochs: int = 5_000
    learning_rate: float = 0.01
    min_lr_factor: float = 0.01
    ablation_architecture: str = "mlp32"
    ablation_weight_decays: tuple[float, ...] = (1e-3,)
    comparison_seeds: tuple[int, ...] = tuple(range(5))
    benchmark_number: int = 5_000
    benchmark_repeats: int = 9
    simulation_hands: int = 50_000
    simulation_seed: int = 2026

    def __post_init__(self) -> None:
        if not self.seeds or not self.architectures:
            raise ValueError("seeds and architectures must be non-empty")
        unknown = set(self.splits) - set(SPLIT_NAMES)
        if unknown:
            raise ValueError(f"unknown splits {sorted(unknown)}")
        if self.ablation_weight_decays and self.ablation_architecture not in self.architectures:
            raise ValueError(f"ablation architecture {self.ablation_architecture!r} not trained")

    def training(self, keep_best: bool) -> TrainingConfig:
        return TrainingConfig(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            lr_schedule="cosine",
            min_lr_factor=self.min_lr_factor,
            keep_best=keep_best,
        )


# -- metrics ----------------------------------------------------------------------------------


def distribution_metrics(
    predicted: Row, target: Row, masks: npt.NDArray[np.bool_]
) -> dict[str, float]:
    """Imitation metrics averaged over states (rows). Illegal entries are 0 in both."""
    if predicted.shape != target.shape or len(predicted) == 0:
        raise ValueError("predicted and target need the same non-empty shape")
    q = np.maximum(predicted, PROBABILITY_FLOOR)
    positive = target > 0
    log_t = np.log(np.where(positive, target, 1.0))
    kl = np.where(positive, target * (log_t - np.log(q)), 0.0).sum(axis=1)
    ce = -np.where(positive, target * np.log(q), 0.0).sum(axis=1)
    abs_err = np.abs(predicted - target)
    legal_counts = masks.sum(axis=1)
    return {
        "cross_entropy": float(ce.mean()),
        "kl": float(kl.mean()),
        "kl_max": float(kl.max()),
        "total_variation": float(0.5 * abs_err.sum(axis=1).mean()),
        "mean_abs_error": float((abs_err.sum(axis=1) / legal_counts).mean()),
        "max_abs_error": float(abs_err.max()),
        "argmax_agreement": float((predicted.argmax(axis=1) == target.argmax(axis=1)).mean()),
        "floored_entries": int(((predicted < PROBABILITY_FLOOR) & positive).sum()),
    }


def table_from_rows(dataset: LeducDataset, rows: Row) -> Table:
    return {
        e.key: {a: float(rows[i, a]) for a in e.legal_actions}
        for i, e in enumerate(dataset.examples)
    }


def model_rows(model: GamePolicyNetwork, dataset: LeducDataset) -> Row:
    return legal_probabilities(model, dataset.features(), dataset.masks())


def reference_opponents(
    specs: Mapping[str, GameInfoSet], teacher: GameStrategy
) -> dict[str, Table]:
    def pure(choose: Any) -> Table:
        return {k: {a: float(a == choose(s.actions)) for a in s.actions} for k, s in specs.items()}

    return {
        "uniform": {k: {a: 1.0 / len(s.actions) for a in s.actions} for k, s in specs.items()},
        "always_call": pure(lambda actions: CALL),
        "always_raise": pure(lambda actions: RAISE if RAISE in actions else CALL),
        "teacher": {k: dict(v) for k, v in teacher.items()},
    }


def policy_metrics(
    game: LeducGame,
    specs: dict[str, GameInfoSet],
    strategy: GameStrategy,
    opponents: Mapping[str, GameStrategy],
) -> dict[str, Any]:
    report = exploitability_report(game, strategy, infosets=specs)
    ev = {}
    for name, opp in opponents.items():
        as0 = expected_value(game, (strategy, opp), infosets=specs)
        as1 = -expected_value(game, (opp, strategy), infosets=specs)
        ev[name] = (as0 + as1) / 2
    return {
        "exploitability": report.exploitability,
        "nash_conv": report.nash_conv,
        "best_response_values": list(report.best_response_values),
        "seat0_self_play_value": report.on_policy_values[0],
        "ev_vs": ev,
    }


# -- non-neural baselines ---------------------------------------------------------------------


def baseline_rows(dataset: LeducDataset, split: Split) -> dict[str, tuple[Row, dict[str, int]]]:
    """Each baseline's rows for all states: the teacher where fitted, the baseline on test."""
    teacher = dataset.targets()
    feats = dataset.features().astype(np.float64)
    masks = dataset.masks()
    fitted = list(split.train) + list(split.validation)
    out: dict[str, tuple[Row, dict[str, int]]] = {}

    uniform = teacher.copy()
    for i in split.test:
        uniform[i] = masks[i] / masks[i].sum()
    out["uniform"] = (uniform, {})

    nearest = teacher.copy()
    for i in split.test:
        same = [j for j in fitted if (masks[j] == masks[i]).all()]
        dist = np.abs(feats[same] - feats[i]).sum(axis=1)
        ties = [same[k] for k in np.flatnonzero(dist == dist.min())]
        nearest[i] = teacher[ties].mean(axis=0)
    out["nearest"] = (nearest, {})

    history = teacher.copy()
    fallbacks = 0
    for i in split.test:
        h = dataset.examples[i].histories
        same = [
            j for j in fitted if dataset.examples[j].histories == h and (masks[j] == masks[i]).all()
        ]
        if same:
            history[i] = teacher[same].mean(axis=0)
        else:
            history[i] = masks[i] / masks[i].sum()
            fallbacks += 1
    out["history_mean"] = (history, {"uniform_fallbacks": fallbacks})
    return out


# -- the study --------------------------------------------------------------------------------


def _arrays(dataset: LeducDataset, idx: Sequence[int]) -> Arrays:
    return Arrays(dataset.features(idx), dataset.masks(idx), dataset.targets(idx))


def _parts(split: Split) -> dict[str, tuple[int, ...]]:
    parts = {"train": split.train, "validation": split.validation, "test": split.test}
    return {k: v for k, v in parts.items() if v}


def _selections(split: Split) -> tuple[str, ...]:
    return ("validation", "final") if split.validation else ("train_best",)


def _evaluate_rows(
    rows: Row,
    dataset: LeducDataset,
    split: Split,
    game: LeducGame,
    specs: dict[str, GameInfoSet],
    opponents: Mapping[str, GameStrategy],
) -> dict[str, Any]:
    targets, masks = dataset.targets(), dataset.masks()
    result: dict[str, Any] = {
        "distribution": {
            part: distribution_metrics(rows[list(idx)], targets[list(idx)], masks[list(idx)])
            for part, idx in _parts(split).items()
        },
        "full_policy": policy_metrics(
            game, specs, dataset.expand(table_from_rows(dataset, rows)), opponents
        ),
    }
    if split.test:
        hybrid = targets.copy()
        hybrid[list(split.test)] = rows[list(split.test)]
        result["hybrid_policy"] = policy_metrics(
            game, specs, dataset.expand(table_from_rows(dataset, hybrid)), opponents
        )
    return result


def _train_and_evaluate(
    config: LeducNeuralStudyConfig,
    dataset: LeducDataset,
    split: Split,
    variant: str,
    hidden: tuple[int, ...],
    seed: int,
    selection: str,
    weight_decay: float,
    game: LeducGame,
    specs: dict[str, GameInfoSet],
    opponents: Mapping[str, GameStrategy],
    checkpoint_dir: Path | None,
) -> tuple[dict[str, Any], GamePolicyNetwork]:
    policy = GamePolicyConfig(NUM_FEATURES, NUM_ACTIONS, hidden, seed)
    validation = _arrays(dataset, split.validation) if selection == "validation" else None
    model, history = train_game_policy(
        _arrays(dataset, split.train),
        policy,
        config.training(keep_best=selection != "final"),
        validation,
        weight_decay=weight_decay,
    )
    rows = model_rows(model, dataset)
    record: dict[str, Any] = {
        "split": split.name,
        "variant": variant,
        "architecture": policy.name,
        "weight_decay": weight_decay,
        "seed": seed,
        "selection": selection,
        "parameters": sum(p.numel() for p in model.parameters()),
        "selected_epoch": history.selected_epoch,
        "training_seconds": history.seconds,
        "state_dict_digest": state_dict_digest(model),  # type: ignore[arg-type]
        "strategy_digest": game_strategy_digest(
            dataset.expand(table_from_rows(dataset, rows)), game.action_codes
        ),
        **_evaluate_rows(rows, dataset, split, game, specs, opponents),
        "curve": history.to_dict(every=50),
    }
    if checkpoint_dir is not None and seed == config.seeds[0]:
        name = f"{split.name}_{variant}_{selection}_seed{seed}.pt"
        record["checkpoint"] = str(checkpoint_dir / name)
        save_checkpoint(
            checkpoint_dir / name,
            model,
            {"split": split.name, "selection": selection, "teacher": dict(dataset.teacher)},
        )
    return record, model


SUMMARY_FIELDS: Final = (
    ("train_kl", ("distribution", "train", "kl")),
    ("validation_kl", ("distribution", "validation", "kl")),
    ("test_kl", ("distribution", "test", "kl")),
    ("test_cross_entropy", ("distribution", "test", "cross_entropy")),
    ("test_total_variation", ("distribution", "test", "total_variation")),
    ("test_mean_abs_error", ("distribution", "test", "mean_abs_error")),
    ("test_max_abs_error", ("distribution", "test", "max_abs_error")),
    ("test_argmax_agreement", ("distribution", "test", "argmax_agreement")),
    ("full_exploitability", ("full_policy", "exploitability")),
    ("hybrid_exploitability", ("hybrid_policy", "exploitability")),
    ("full_ev_vs_uniform", ("full_policy", "ev_vs", "uniform")),
    ("full_ev_vs_always_call", ("full_policy", "ev_vs", "always_call")),
    ("full_ev_vs_always_raise", ("full_policy", "ev_vs", "always_raise")),
    ("full_ev_vs_teacher", ("full_policy", "ev_vs", "teacher")),
    ("selected_epoch", ("selected_epoch",)),
)


def _get(record: Mapping[str, Any], path: Sequence[str]) -> float | None:
    node: Any = record
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return float(node)


def summarize(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean, standard deviation, min, median and max over seeds for each group."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["split"], r["variant"], r["selection"]), []).append(r)
    out = []
    for (split, variant, selection), rs in groups.items():
        row: dict[str, Any] = {
            "split": split,
            "variant": variant,
            "selection": selection,
            "seeds": len(rs),
            "parameters": rs[0]["parameters"],
        }
        for name, path in SUMMARY_FIELDS:
            values = [v for v in (_get(r, path) for r in rs) if v is not None]
            if not values:
                continue
            row[name] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
        out.append(row)
    return out


def _teacher(path: Path, game: LeducGame) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    strategy = load_game_strategy(path, game.name, game.action_codes)
    document = json.loads(path.read_text())
    return strategy, {
        "path": str(path),
        "digest": game_strategy_digest(strategy, game.action_codes),
        "metadata": document.get("metadata", {}),
    }


def compare_teachers(
    config: LeducNeuralStudyConfig,
    game: LeducGame,
    specs: dict[str, GameInfoSet],
    main: LeducDataset,
    opponents: Mapping[str, GameStrategy],
) -> dict[str, Any] | None:
    """The older teacher vs the main one, and full-fit students of each."""
    if config.comparison_teacher_path is None:
        return None
    other_strategy, other_meta = _teacher(config.comparison_teacher_path, game)
    other = build_leduc_dataset(other_strategy, other_meta, game)
    diff = np.abs(other.targets() - main.targets())
    full = make_split(main, "full")
    arch = max(config.architectures, key=lambda a: sum(config.architectures[a]) or 0)
    students: dict[str, list[float]] = {"main": [], "comparison": []}
    for label, ds in (("main", main), ("comparison", other)):
        for seed in config.comparison_seeds:
            model, _ = train_game_policy(
                _arrays(ds, full.train),
                GamePolicyConfig(NUM_FEATURES, NUM_ACTIONS, config.architectures[arch], seed),
                config.training(keep_best=True),
            )
            table = ds.expand(table_from_rows(ds, model_rows(model, ds)))
            students[label].append(
                exploitability_report(game, table, infosets=specs).exploitability
            )
    return {
        "comparison_teacher": other_meta,
        "comparison_policy": policy_metrics(game, specs, other_strategy, opponents),
        "teacher_difference": {
            "mean_total_variation": float(0.5 * diff.sum(axis=1).mean()),
            "max_abs_difference": float(diff.max()),
            "states_differing_by_more_than_0.05": int((diff.max(axis=1) > 0.05).sum()),
        },
        "student_architecture": arch,
        "student_exploitability": students,
    }


def benchmark_performance(
    models: Mapping[str, GamePolicyNetwork],
    dataset: LeducDataset,
    teacher: GameStrategy,
    teacher_path: Path,
    number: int,
    repeats: int,
) -> dict[str, Any]:
    """Latency of table lookup vs neural inference, sizes and memory."""
    example = next(e for e in dataset.examples if e.key == "K|J|rc|r")
    obs, legal = example.observation, example.legal_actions
    key = obs.key
    table = {k: dict(v) for k, v in teacher.items()}
    table_agent = GameStrategyAgent(table)
    rng_check = np.random.default_rng(0)
    assert table_agent.act(obs, legal, rng_check) in legal
    all_x = torch.from_numpy(dataset.features())
    big_x = all_x.repeat(math.ceil(4096 / len(all_x)), 1)[:4096].contiguous()
    namespace: dict[str, Any] = {
        "table": table,
        "key": key,
        "table_agent": table_agent,
        "obs": obs,
        "legal": legal,
        "rng": np.random.default_rng(1),
        "encode": encode_observation,
    }
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    results: dict[str, Any] = {}
    try:
        with torch.inference_mode():
            empty = time_statement(
                "empty", "pass", "pass", namespace, number=number, repeats=repeats
            )
            overhead = empty.median_ns
            shared = [
                time_statement(
                    n, d, s, namespace, number=number, repeats=repeats, overhead_ns=overhead
                )
                for n, d, s in (
                    ("dict_lookup", "bare table[key]", "_ = table[key]"),
                    (
                        "table_agent_decision",
                        "GameStrategyAgent.act",
                        "_ = table_agent.act(obs, legal, rng)",
                    ),
                    ("encode", "encode_observation", "_ = encode(obs)"),
                )
            ]
            results["table_and_encoding"] = {t.name: t.to_dict() for t in [empty, *shared]}
            for name, model in models.items():
                agent = LeducNeuralAgent(model)
                assert agent.act(obs, legal, rng_check) in legal
                ns = {
                    **namespace,
                    "model": model,
                    "agent": agent,
                    "x1": all_x[:1].clone(),
                    "all_x": all_x,
                    "big_x": big_x,
                }
                timings = [
                    time_statement(n, d, s, ns, number=num, repeats=repeats, overhead_ns=overhead)
                    for n, d, s, num in (
                        ("forward_1", "model(x) on a 1x30 tensor", "_ = model(x1)", number),
                        (
                            "agent_decision",
                            "LeducNeuralAgent.act (encode, forward, "
                            "float64 masked softmax, sample)",
                            "_ = agent.act(obs, legal, rng)",
                            number,
                        ),
                        (
                            "forward_288",
                            "model(x) on all 288 canonical states",
                            "_ = model(all_x)",
                            max(number // 10, 100),
                        ),
                        (
                            "forward_4096",
                            "model(x) on a 4096x30 batch",
                            "_ = model(big_x)",
                            max(number // 50, 50),
                        ),
                    )
                ]
                entry: dict[str, Any] = {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "parameter_bytes": sum(
                        p.numel() * p.element_size() for p in model.parameters()
                    ),
                    "checkpoint_bytes": len(checkpoint_bytes(model)),
                    "timings": {t.name: t.to_dict() for t in timings},
                    "per_state_ns": {
                        "batch_288": timings[2].net_median_ns / 288,
                        "batch_4096": timings[3].net_median_ns / 4096,
                    },
                }
                entry["mps"] = _benchmark_mps(model, big_x, repeats)
                results[name] = entry
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "conditions": {
            "torch_threads": 1,
            "torch_mode": "inference_mode",
            "number": number,
            "repeats": repeats,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "caveat": "single machine and process; compare operations measured together",
        },
        "timings": results,
        "table": {
            "information_sets": len(table),
            "json_file_bytes": teacher_path.stat().st_size,
            "float64_values": sum(len(v) for v in table.values()),
            "approx_python_bytes": _deep_size(table),
        },
        "process_max_rss_bytes": _max_rss_bytes(),
    }


def _benchmark_mps(model: GamePolicyNetwork, big_x: torch.Tensor, repeats: int) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"available": False}
    device = torch.device("mps")
    gpu = GamePolicyNetwork(model.config).to(device).eval()
    gpu.load_state_dict(model.state_dict())
    gx = big_x.to(device)
    diff = float((gpu(gx).cpu() - model(big_x)).abs().max())
    t = time_statement(
        "forward_4096_mps_with_copy",
        "model(x) on a 4096x30 batch on the Apple GPU + copy to the CPU",
        "_ = gpu(gx).cpu()",
        {"gpu": gpu, "gx": gx},
        number=200,
        repeats=repeats,
        sync=torch.mps.synchronize,
    )
    return {
        "available": True,
        "max_abs_logit_diff_vs_cpu": diff,
        "timing": t.to_dict(),
        "per_state_ns": t.median_ns / 4096,
    }


def _deep_size(table: Mapping[str, Mapping[int, float]]) -> int:
    size = sys.getsizeof(table)
    for key, dist in table.items():
        size += sys.getsizeof(key) + sys.getsizeof(dist)
        size += sum(sys.getsizeof(a) + sys.getsizeof(p) for a, p in dist.items())
    return size


def _max_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def run_leduc_neural_study(config: LeducNeuralStudyConfig, output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    game = LeducGame()
    specs = information_sets(game)
    teacher, teacher_meta = _teacher(config.teacher_path, game)
    dataset = build_leduc_dataset(teacher, teacher_meta, game)
    opponents = reference_opponents(specs, teacher)
    teacher_policy = policy_metrics(game, specs, teacher, opponents)
    checkpoint_dir = output_dir / "checkpoints"

    variants: list[tuple[str, tuple[int, ...], float]] = [
        (name, hidden, 0.0) for name, hidden in config.architectures.items()
    ]
    variants += [
        (
            f"{config.ablation_architecture}_wd{wd:g}",
            config.architectures[config.ablation_architecture],
            wd,
        )
        for wd in config.ablation_weight_decays
    ]

    coverage: dict[str, Any] = {}
    baselines: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    full_models: dict[str, GamePolicyNetwork] = {}
    for split_name in config.splits:
        split = make_split(dataset, split_name, config.split_seed)
        coverage[split_name] = split_coverage(dataset, split)
        if split.test:
            baselines[split_name] = {
                name: {
                    **_evaluate_rows(rows, dataset, split, game, specs, opponents),
                    "notes": notes,
                }
                for name, (rows, notes) in baseline_rows(dataset, split).items()
            }
        for variant, hidden, wd in variants:
            for seed in config.seeds:
                for selection in _selections(split):
                    record, model = _train_and_evaluate(
                        config,
                        dataset,
                        split,
                        variant,
                        hidden,
                        seed,
                        selection,
                        wd,
                        game,
                        specs,
                        opponents,
                        checkpoint_dir,
                    )
                    records.append(record)
                    if split_name == "full" and seed == config.seeds[0] and wd == 0.0:
                        full_models[variant] = model
                    print(
                        f"{split_name:8s} {variant:14s} seed {seed} {selection:10s} "
                        f"full expl {record['full_policy']['exploitability']:.4f}",
                        flush=True,
                    )

    comparison = compare_teachers(config, game, specs, dataset, opponents)
    performance = (
        benchmark_performance(
            full_models,
            dataset,
            teacher,
            config.teacher_path,
            config.benchmark_number,
            config.benchmark_repeats,
        )
        if full_models
        else None
    )
    simulation = None
    if config.simulation_hands > 0 and full_models:
        best = min(
            (
                r
                for r in records
                if r["split"] == "full"
                and r["seed"] == config.seeds[0]
                and r["weight_decay"] == 0.0
            ),
            key=lambda r: r["full_policy"]["exploitability"],
        )
        agent = LeducNeuralAgent(full_models[best["variant"]])
        sim = run_game_simulation(
            game,
            (agent, UniformGameAgent()),
            GameSimulationConfig(config.simulation_hands, config.simulation_seed),
        )
        exact = best["full_policy"]["ev_vs"]["uniform"]
        simulation = {
            **sim,
            "variant": best["variant"],
            "exact_ev_vs_uniform": exact,
            "z_score": (sim["mean_returns"][0] - exact) / sim["std_errors"][0],
        }

    document: dict[str, Any] = {
        "experiment": "m4-leduc-neural-generalization",
        "config": {
            "teacher_path": str(config.teacher_path),
            "comparison_teacher_path": (
                str(config.comparison_teacher_path) if config.comparison_teacher_path else None
            ),
            "architectures": {k: list(v) for k, v in config.architectures.items()},
            "seeds": list(config.seeds),
            "splits": list(config.splits),
            "split_seed": config.split_seed,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "lr_schedule": "cosine",
            "min_lr_factor": config.min_lr_factor,
            "ablation": {
                "architecture": config.ablation_architecture,
                "weight_decays": list(config.ablation_weight_decays),
            },
            "probability_floor": PROBABILITY_FLOOR,
        },
        "metadata": collect_metadata(),
        "teacher": {**teacher_meta, "policy": teacher_policy},
        "dataset": {
            "canonical_states": len(dataset),
            "information_sets": sum(len(e.members) for e in dataset.examples),
            "max_suit_symmetry_deviation": dataset.max_symmetry_deviation,
            "num_features": NUM_FEATURES,
        },
        "coverage": coverage,
        "baselines": baselines,
        "summary": summarize(records),
        "teacher_comparison": comparison,
        "performance": performance,
        "simulation_check": simulation,
        "runs": records,
        "elapsed_seconds": time.perf_counter() - start,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(document, indent=2) + "\n")
    _write_csv(output_dir / "runs.csv", records)
    document["plot"] = plot_study(document, output_dir / "generalization.png")
    return document


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    columns = ["split", "variant", "selection", "seed", "parameters"] + [
        n for n, _ in SUMMARY_FIELDS
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for r in records:
            values = [_get(r, p) for _, p in SUMMARY_FIELDS]
            writer.writerow(
                [r["split"], r["variant"], r["selection"], r["seed"], r["parameters"]]
                + ["" if v is None else f"{v:.10g}" for v in values]
            )


def plot_study(document: Mapping[str, Any], path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    summary = document["summary"]
    variants = list(document["config"]["architectures"])
    held_out = [s for s in document["config"]["splits"] if s != "full"]
    teacher_expl = document["teacher"]["policy"]["exploitability"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    def stat(split: str, variant: str, selection: str, name: str) -> tuple[float, float]:
        for row in summary:
            same = (row["split"], row["variant"], row["selection"]) == (split, variant, selection)
            if same and name in row:
                return row[name]["mean"], row[name]["std"]
        return math.nan, math.nan

    x = np.arange(len(variants))
    if "full" in document["config"]["splits"]:
        m, s = zip(
            *(stat("full", v, "train_best", "full_exploitability") for v in variants), strict=True
        )
        axes[0].bar(x, m, yerr=s, color="#4c72b0")
        axes[0].axhline(teacher_expl, color="black", ls="--", lw=1, label="teacher")
        axes[0].set_yscale("log")
        axes[0].set_xticks(x, variants)
        axes[0].set_title("Fit on all 288 states: exploitability")
        axes[0].legend()
    width = 0.8 / max(len(held_out), 1)
    for k, split in enumerate(held_out):
        m, s = zip(*(stat(split, v, "validation", "test_kl") for v in variants), strict=True)
        axes[1].bar(x + k * width, m, width, yerr=s, label=split)
        m, s = zip(
            *(stat(split, v, "validation", "hybrid_exploitability") for v in variants), strict=True
        )
        axes[2].bar(x + k * width, m, width, yerr=s, label=split)
        nearest = document["baselines"].get(split, {}).get("nearest")
        if nearest:
            axes[2].hlines(
                nearest["hybrid_policy"]["exploitability"],
                x[0] - 0.4 + k * width,
                x[-1] + 0.4 + k * width,
                colors=f"C{k}",
                linestyles=":",
                lw=1,
            )
    for ax, title in (
        (axes[1], "Held-out KL (validation-selected)"),
        (axes[2], "Hybrid exploitability (teacher + model on test)\ndotted: nearest-neighbour"),
    ):
        ax.set_xticks(x + width * (len(held_out) - 1) / 2, variants)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
