"""M2A experiment: distill the M1 CFR strategy into tiny networks and evaluate them.

For each (architecture, learning rate, seed), the runner trains a policy and
measures its imitation error against the teacher. Every policy is also
evaluated with M1's exact evaluator: exploitability, and expected payoff
against the baseline agents. On top of that the runner:

* attributes each selected model's extra exploitability to information sets,
  by swapping one teacher entry at a time for the student's;
* benchmarks lookup latency against the 12-entry table under the same
  conditions (one CPU thread, warm-up, median of repeats), and on the Apple
  Silicon GPU (MPS) when it is available;
* plays the selected models through the M0 simulator.

Artifacts go to a run directory (default under the git-ignored ``runs/``).
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pokermind.agents import RandomAgent, StrategyAgent, ThresholdAgent
from pokermind.agents.base import Agent
from pokermind.evaluation import (
    expected_value_seat_averaged,
    exploitability,
    exploitability_report,
    uniform_strategy,
)
from pokermind.experiments.cfr_experiment import deterministic_agent_strategy
from pokermind.games.kuhn import Action, Card, InfoState
from pokermind.neural.agent import NeuralPolicyAgent
from pokermind.neural.dataset import TeacherDataset, load_teacher
from pokermind.neural.encoding import ACTIONS, encode_info_state, legal_action_mask
from pokermind.neural.model import (
    PolicyConfig,
    PolicyNetwork,
    model_strategy,
    parameter_count,
    save_checkpoint,
)
from pokermind.neural.training import TrainingConfig, state_dict_digest, train_policy
from pokermind.simulation import SimulationConfig, collect_metadata, run_simulation
from pokermind.strategy import Strategy, save_strategy

ARCHITECTURES: dict[str, tuple[int, ...]] = {"linear": (), "mlp16": (16,), "mlp16x16": (16, 16)}


@dataclass(frozen=True, slots=True)
class NeuralExperimentConfig:
    architectures: tuple[str, ...] = ("linear", "mlp16", "mlp16x16")
    learning_rates: tuple[float, ...] = (0.01, 0.03)
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 3_000
    teacher_iterations: int = 50_000
    teacher_path: str | None = None
    simulation_hands: int = 50_000
    simulation_seed: int = 0
    benchmark: bool = True
    benchmark_calls: int = 2_000
    benchmark_repeats: int = 7

    def __post_init__(self) -> None:
        unknown = set(self.architectures) - set(ARCHITECTURES)
        if unknown:
            raise ValueError(
                f"unknown architectures {sorted(unknown)}; known: {list(ARCHITECTURES)}"
            )
        if not self.architectures or not self.learning_rates or not self.seeds:
            raise ValueError("architectures, learning_rates and seeds must be non-empty")
        if self.simulation_hands < 0:
            raise ValueError("simulation_hands must be non-negative")


# -- teacher/student comparison ---------------------------------------------------------


def compare_to_teacher(student: Strategy, teacher: Strategy) -> dict[str, Any]:
    """Probability errors over every (information set, legal action), and KL(p ‖ q)."""
    abs_errors: list[float] = []
    per_state: dict[str, dict[str, float]] = {}
    for key, dist in teacher.items():
        errors = [abs(student[key][a] - p) for a, p in dist.items()]
        kl = sum(p * (math.log(p) - math.log(student[key][a])) for a, p in dist.items() if p > 0)
        per_state[key] = {"max_abs_error": max(errors), "kl": kl}
        abs_errors.extend(errors)
    return {
        "max_abs_error": max(abs_errors),
        "mean_abs_error": sum(abs_errors) / len(abs_errors),
        "mean_kl": sum(s["kl"] for s in per_state.values()) / len(per_state),
        "max_kl": max(s["kl"] for s in per_state.values()),
        "per_state": per_state,
    }


def exploitability_attribution(student: Strategy, teacher: Strategy) -> dict[str, float]:
    """Exploitability change when only one information set uses the student's policy.

    Exploitability isn't additive across information sets, so these numbers
    are diagnostic only: they show where the student's errors cost the most.
    """
    base = exploitability(teacher)
    result = {}
    for key in teacher:
        mixed = {k: dict(v) for k, v in teacher.items()}
        mixed[key] = dict(student[key])
        result[key] = exploitability(mixed) - base
    return dict(sorted(result.items(), key=lambda kv: -abs(kv[1])))


def table_from_dataset(dataset: TeacherDataset) -> dict[str, dict[Action, float]]:
    return {
        e.key: {a: float(e.target[i]) for i, a in enumerate(ACTIONS) if e.mask[i]}
        for e in dataset.examples
    }


# -- latency benchmark --------------------------------------------------------------------


def _median_ns_per_call(fn: Callable[[], object], calls: int, repeats: int) -> float:
    for _ in range(max(calls // 10, 10)):  # warm-up
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(calls):
            fn()
        samples.append((time.perf_counter_ns() - start) / calls)
    return statistics.median(samples)


def benchmark_latency(
    model: PolicyNetwork, teacher: Strategy, calls: int, repeats: int
) -> dict[str, Any]:
    """Median nanoseconds per call under identical conditions (1 CPU thread)."""
    info = InfoState(player=1, private_card=Card.Q, history=(Action.BET,))
    legal = (Action.FOLD, Action.CALL)
    rng = np.random.default_rng(0)
    table_agent = StrategyAgent(teacher)
    neural_agent = NeuralPolicyAgent(model)
    x1 = torch.from_numpy(encode_info_state(info)).unsqueeze(0)
    from pokermind.neural.dataset import decision_info_states

    states = list(decision_info_states().values())
    x12 = torch.from_numpy(np.stack([encode_info_state(i) for i, _ in states]))
    masks12 = np.stack([legal_action_mask(lg) for _, lg in states])
    key = info.key

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            results: dict[str, Any] = {
                "conditions": {
                    "torch_threads": 1,
                    "calls_per_repeat": calls,
                    "repeats": repeats,
                    "statistic": "median ns/call",
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                },
                "table_lookup_ns": _median_ns_per_call(lambda: teacher[key], calls, repeats),
                "table_agent_act_ns": _median_ns_per_call(
                    lambda: table_agent.act(info, legal, rng), calls, repeats
                ),
                "neural_forward_cpu_batch1_ns": _median_ns_per_call(
                    lambda: model(x1), calls, repeats
                ),
                "neural_forward_cpu_batch12_ns": _median_ns_per_call(
                    lambda: model(x12), calls, repeats
                ),
                "neural_policy_cpu_batch12_float64_ns": _median_ns_per_call(
                    lambda: _probabilities(model, x12, masks12), calls, repeats
                ),
                "neural_agent_act_cpu_ns": _median_ns_per_call(
                    lambda: neural_agent.act(info, legal, rng), calls, repeats
                ),
            }
            results["mps"] = _benchmark_mps(model, x1, x12, calls, repeats)
    finally:
        torch.set_num_threads(previous_threads)
    return results


def _probabilities(model: PolicyNetwork, x: torch.Tensor, masks: np.ndarray) -> np.ndarray:
    logits = model(x).double().numpy()
    z = np.where(masks, logits, -np.inf)
    e = np.where(masks, np.exp(z - z.max(axis=-1, keepdims=True)), 0.0)
    return e / e.sum(axis=-1, keepdims=True)


def _benchmark_mps(
    model: PolicyNetwork, x1: torch.Tensor, x12: torch.Tensor, calls: int, repeats: int
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"available": False}
    device = torch.device("mps")
    gpu_model = PolicyNetwork(model.config).to(device)
    gpu_model.load_state_dict(model.state_dict())
    gpu_model.eval()
    g1, g12 = x1.to(device), x12.to(device)

    def run(x: torch.Tensor) -> Callable[[], object]:
        def call() -> object:
            out = gpu_model(x).cpu()  # include the device -> host copy, which synchronizes
            return out

        return call

    # Same calls/repeats as the CPU benchmark; input already resident on the GPU.
    out = {
        "available": True,
        "includes": "forward pass + copy of the logits back to the CPU",
        "neural_forward_mps_batch1_ns": _median_ns_per_call(run(g1), calls, repeats),
        "neural_forward_mps_batch12_ns": _median_ns_per_call(run(g12), calls, repeats),
    }
    with torch.no_grad():
        cpu_logits = model(x12)
        out["max_abs_logit_diff_vs_cpu"] = float((gpu_model(g12).cpu() - cpu_logits).abs().max())
    return out


# -- runner -------------------------------------------------------------------------------


def run_neural_experiment(config: NeuralExperimentConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_teacher(
        Path(config.teacher_path) if config.teacher_path else None, config.teacher_iterations
    )
    teacher = table_from_dataset(dataset)
    teacher_report = exploitability_report(teacher)
    baselines = {
        "random": uniform_strategy(),
        "rule": deterministic_agent_strategy(ThresholdAgent()),
    }
    teacher_file = output_dir / "teacher_strategy.json"
    save_strategy(teacher_file, teacher, dict(dataset.teacher))

    runs: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    models: dict[str, PolicyNetwork] = {}
    for arch in config.architectures:
        for lr in config.learning_rates:
            for seed in config.seeds:
                policy = PolicyConfig(hidden_sizes=ARCHITECTURES[arch], seed=seed)
                model, history = train_policy(
                    dataset, policy, TrainingConfig(epochs=config.epochs, learning_rate=lr)
                )
                run_id = f"{arch}_lr{lr:g}_seed{seed}"
                student = model_strategy(model)
                report = exploitability_report(student)
                path = output_dir / "models" / f"{run_id}.pt"
                size = save_checkpoint(path, model, {"run_id": run_id, "teacher": dataset.teacher})
                runs.append(
                    {
                        "run_id": run_id,
                        "architecture": arch,
                        "hidden_sizes": list(ARCHITECTURES[arch]),
                        "learning_rate": lr,
                        "seed": seed,
                        "epochs": config.epochs,
                        "parameter_count": parameter_count(model),
                        "parameter_bytes_float32": parameter_count(model) * 4,
                        "checkpoint_bytes": size,
                        "train_seconds": history.seconds,
                        "final_cross_entropy": history.cross_entropy[-1],
                        "final_kl_training": history.kl[-1],
                        **{
                            k: v
                            for k, v in compare_to_teacher(student, teacher).items()
                            if k != "per_state"
                        },
                        "exploitability": report.exploitability,
                        "nash_conv": report.nash_conv,
                        "exploitability_minus_teacher": report.exploitability
                        - teacher_report.exploitability,
                        "self_play_value_seat0": report.on_policy_values[0],
                        "expected_payoff_vs": {
                            n: expected_value_seat_averaged(student, b)
                            for n, b in baselines.items()
                        },
                        "weights_digest": state_dict_digest(model),
                        "checkpoint": str(path.relative_to(output_dir)),
                    }
                )
                histories[run_id] = history.to_dict()
                models[run_id] = model

    selected = _select(runs, config.architectures)
    analysis = {}
    for arch, run in selected.items():
        model = models[run["run_id"]]
        student = model_strategy(model)
        entry: dict[str, Any] = {
            "run_id": run["run_id"],
            "per_state": compare_to_teacher(student, teacher)["per_state"],
            "exploitability_attribution": exploitability_attribution(student, teacher),
            "student_strategy": {k: {a.code: p for a, p in d.items()} for k, d in student.items()},
            "close_imitation_but_worse_exploitability": bool(
                run["max_abs_error"] < 0.01
                and run["exploitability"] > 1.5 * teacher_report.exploitability
            ),
        }
        if config.simulation_hands:
            entry["simulation_check"] = _simulation_check(model, student, baselines, config)
        if config.benchmark:
            entry["latency"] = benchmark_latency(
                model, teacher, config.benchmark_calls, config.benchmark_repeats
            )
        analysis[arch] = entry

    document: dict[str, Any] = {
        "experiment": "kuhn-m2a-neural-distillation",
        "config": {
            **{k: getattr(config, k) for k in config.__dataclass_fields__},
            "loss": "soft-target cross-entropy (gradient-equivalent to KL(teacher || student))",
            "optimizer": "Adam (beta1=0.9, beta2=0.999, eps=1e-8), full batch of 12",
            "selection": "per architecture: lowest final training KL (train set = all 12 states)",
        },
        "metadata": {**collect_metadata(), "torch_version": torch.__version__},
        "teacher": {
            **dict(dataset.teacher),
            "exploitability": teacher_report.exploitability,
            "nash_conv": teacher_report.nash_conv,
            "expected_payoff_vs": {
                n: expected_value_seat_averaged(teacher, b) for n, b in baselines.items()
            },
            "table_entries": len(teacher),
            "table_probabilities": sum(len(d) for d in teacher.values()),
            "table_bytes_float64": 8 * sum(len(d) for d in teacher.values()),
            "table_file_bytes": teacher_file.stat().st_size,
        },
        "dataset": dataset.describe(),
        "runs": runs,
        "selected": analysis,
    }
    (output_dir / "metrics.json").write_text(json.dumps(document, indent=2) + "\n")
    (output_dir / "histories.json").write_text(json.dumps(histories) + "\n")
    document["plot"] = plot_training(runs, histories, selected, output_dir / "training_curves.png")
    return document


def _select(runs: Sequence[dict[str, Any]], archs: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {
        arch: min(
            (r for r in runs if r["architecture"] == arch), key=lambda r: r["final_kl_training"]
        )
        for arch in archs
    }


def _simulation_check(
    model: PolicyNetwork,
    student: Strategy,
    baselines: dict[str, Strategy],
    config: NeuralExperimentConfig,
) -> list[dict[str, Any]]:
    opponents: dict[str, Agent] = {"random": RandomAgent(), "rule": ThresholdAgent()}
    out = []
    for name, opponent in opponents.items():
        sim = run_simulation(
            (NeuralPolicyAgent(model), opponent),
            SimulationConfig(num_hands=config.simulation_hands, seed=config.simulation_seed),
        )
        exact = expected_value_seat_averaged(student, baselines[name])
        out.append(
            {
                "opponent": name,
                "hands": sim.num_hands,
                "simulated_mean": sim.mean_returns[0],
                "std_error": sim.std_errors[0],
                "exact_expected": exact,
                "z_score": (sim.mean_returns[0] - exact) / sim.std_errors[0],
                "invalid_actions": list(sim.invalid_actions),
                "digest": sim.digest,
                "hands_per_second": sim.hands_per_second,
            }
        )
    return out


# -- plots ---------------------------------------------------------------------------------

_SERIES = {"linear": "#2a78d6", "mlp16": "#eb6834", "mlp16x16": "#1baf7a"}
_TEXT, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def plot_training(
    runs: Sequence[dict[str, Any]],
    histories: dict[str, dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    path: Path,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

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
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    panels = (
        ("kl", "Training loss: KL(teacher ‖ student)", "mean KL (nats)"),
        ("max_abs_error", "Largest probability error vs teacher", "max |q − p|"),
    )
    for ax, (metric, title, ylabel) in zip(axes, panels, strict=True):
        for arch, best in selected.items():
            color = _SERIES.get(arch, _MUTED)
            # All seeds at the selected learning rate: thin lines; selected run: thick.
            for run in runs:
                if run["architecture"] == arch and run["learning_rate"] == best["learning_rate"]:
                    hist = histories[run["run_id"]]
                    values = [max(v, 1e-12) for v in hist[metric]]
                    is_best = run["run_id"] == best["run_id"]
                    ax.plot(
                        [e + 1 for e in hist["epoch"]],
                        values,
                        color=color,
                        lw=2 if is_best else 0.8,
                        alpha=1.0 if is_best else 0.45,
                        label=f"{arch} (lr {best['learning_rate']:g})" if is_best else None,
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", color=_TEXT)
        ax.set_xlabel("epoch (full-batch Adam step)")
        ax.set_ylabel(ylabel)
        ax.grid(True, color=_GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(
            frameon=False,
            fontsize=9,
            labelcolor=_TEXT,
            title="thick: selected seed",
            title_fontsize=8,
        )
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return str(path)


__all__ = [
    "ARCHITECTURES",
    "NeuralExperimentConfig",
    "benchmark_latency",
    "compare_to_teacher",
    "exploitability_attribution",
    "run_neural_experiment",
    "table_from_dataset",
]
