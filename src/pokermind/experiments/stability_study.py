"""M2B studies: training stability, model capacity, benchmark reliability.

* **Stability:** the same architectures, learning rates and seeds trained
  under five procedures:
  * ``baseline``: M2A, constant learning rate, last epoch;
  * ``best_checkpoint``: constant learning rate, lowest-KL epoch;
  * ``cosine``: cosine learning-rate decay, last epoch;
  * ``cosine_best``: cosine decay, lowest-KL epoch;
  * ``early_stop_best``: constant learning rate, early stopping, lowest-KL epoch.
  Each run's final policy is evaluated exactly.
* **Capacity:** linear and one hidden layer of width 2, 4, 8 and 16, trained
  with a fixed procedure over several seeds.
* **Benchmarks:** :func:`pokermind.experiments.latency.benchmark_policies` for
  one model per width.

Everything except wall-clock timings is deterministic. Artifacts go to a run
directory (default ``runs/m2b``, git-ignored).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from pokermind.evaluation import exploitability
from pokermind.experiments.latency import benchmark_policies
from pokermind.experiments.neural_experiment import compare_to_teacher, table_from_dataset
from pokermind.neural.dataset import TeacherDataset, load_teacher
from pokermind.neural.model import PolicyConfig, checkpoint_bytes, model_strategy, parameter_count
from pokermind.neural.training import TrainingConfig, state_dict_digest, train_policy
from pokermind.simulation import collect_metadata

PROCEDURES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "best_checkpoint": {"keep_best": True},
    "cosine": {"lr_schedule": "cosine"},
    "cosine_best": {"lr_schedule": "cosine", "keep_best": True},
    "early_stop_best": {
        "keep_best": True,
        "early_stopping_patience": 200,
        "min_relative_improvement": 0.05,
    },
}

STABILITY_ARCHITECTURES: dict[str, tuple[int, ...]] = {"mlp16": (16,), "mlp16x16": (16, 16)}
CAPACITY_ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "linear": (),
    "mlp2": (2,),
    "mlp4": (4,),
    "mlp8": (8,),
    "mlp16": (16,),
}


@dataclass(frozen=True, slots=True)
class StudyConfig:
    seeds: tuple[int, ...] = tuple(range(10))
    learning_rates: tuple[float, ...] = (0.01, 0.03)
    epochs: int = 3_000
    teacher_iterations: int = 50_000
    capacity_procedure: str = "cosine_best"
    spike_factor: float = 10.0
    benchmark_number: int = 20_000
    benchmark_repeats: int = 15

    def __post_init__(self) -> None:
        if not self.seeds or not self.learning_rates:
            raise ValueError("seeds and learning_rates must be non-empty")
        if self.capacity_procedure not in PROCEDURES:
            raise ValueError(f"unknown procedure {self.capacity_procedure!r}")


def training_config(procedure: str, epochs: int, lr: float) -> TrainingConfig:
    return replace(TrainingConfig(epochs=epochs, learning_rate=lr), **PROCEDURES[procedure])


def count_spikes(kl: Sequence[float], factor: float = 10.0) -> int:
    """Number of spike onsets: epochs where the KL first exceeds ``factor`` times
    its running minimum, after having been below that level."""
    spikes, best, inside = 0, float("inf"), False
    for value in kl:
        above = value > factor * best
        if above and not inside:
            spikes += 1
        inside = above
        best = min(best, value)
    return spikes


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if len(ordered) >= 2:
        q1, median, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    else:
        q1 = median = q3 = ordered[0]
    return {"min": ordered[0], "q1": q1, "median": median, "q3": q3, "max": ordered[-1]}


def _train_and_evaluate(
    dataset: TeacherDataset,
    teacher: dict[str, Any],
    teacher_exploitability: float,
    hidden: tuple[int, ...],
    seed: int,
    training: TrainingConfig,
    spike_factor: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model, history = train_policy(dataset, PolicyConfig(hidden, seed), training)
    student = model_strategy(model)
    errors = compare_to_teacher(student, teacher)
    expl = exploitability(student)
    row = {
        "seed": seed,
        "learning_rate": training.learning_rate,
        "parameter_count": parameter_count(model),
        "checkpoint_bytes": len(checkpoint_bytes(model)),
        "max_abs_error": errors["max_abs_error"],
        "mean_abs_error": errors["mean_abs_error"],
        "mean_kl": errors["mean_kl"],
        "exploitability": expl,
        "excess_exploitability": expl - teacher_exploitability,
        "selected_epoch": history.selected_epoch,
        "epochs_run": history.epoch[-1],
        "stopped_early": history.stopped_early,
        "spikes": count_spikes(history.kl, spike_factor),
        "final_kl": history.kl[-1],
        "selected_kl": history.selected_kl,
        "train_seconds": history.seconds,
        "weights_digest": state_dict_digest(model),
    }
    return row, {"kl": history.kl, "learning_rate": history.learning_rate}


def _group_summary(rows: Sequence[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    return {
        "runs": len(rows),
        "exploitability": _summary([r["exploitability"] for r in rows]),
        "max_abs_error": _summary([r["max_abs_error"] for r in rows]),
        "mean_kl": _summary([r["mean_kl"] for r in rows]),
        "spikes": _summary([float(r["spikes"]) for r in rows]),
        "epochs_run": _summary([float(r["epochs_run"]) for r in rows]),
        "degraded_runs": sum(r["excess_exploitability"] > tolerance for r in rows),
        "degraded_definition": f"exploitability > teacher + {tolerance:g}",
    }


def run_stability_study(
    config: StudyConfig, dataset: TeacherDataset, teacher: dict[str, Any], teacher_expl: float
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}
    for arch, hidden in STABILITY_ARCHITECTURES.items():
        for lr in config.learning_rates:
            for procedure in PROCEDURES:
                training = training_config(procedure, config.epochs, lr)
                for seed in config.seeds:
                    row, curve = _train_and_evaluate(
                        dataset, teacher, teacher_expl, hidden, seed, training, config.spike_factor
                    )
                    row.update({"architecture": arch, "procedure": procedure})
                    rows.append(row)
                    if seed == config.seeds[0]:
                        curves[f"{arch}_lr{lr:g}_{procedure}_seed{seed}"] = curve
    tolerance = 1e-4
    groups = {}
    for arch in STABILITY_ARCHITECTURES:
        for lr in config.learning_rates:
            for procedure in PROCEDURES:
                selected = [
                    r
                    for r in rows
                    if r["architecture"] == arch
                    and r["learning_rate"] == lr
                    and r["procedure"] == procedure
                ]
                groups[f"{arch}|lr{lr:g}|{procedure}"] = _group_summary(selected, tolerance)
    pooled = {
        procedure: _group_summary([r for r in rows if r["procedure"] == procedure], tolerance)
        for procedure in PROCEDURES
    }
    return {
        "procedures": PROCEDURES,
        "rows": rows,
        "groups": groups,
        "pooled": pooled,
        "curves": curves,
    }


def run_capacity_study(
    config: StudyConfig, dataset: TeacherDataset, teacher: dict[str, Any], teacher_expl: float
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for arch, hidden in CAPACITY_ARCHITECTURES.items():
        for lr in config.learning_rates:
            training = training_config(config.capacity_procedure, config.epochs, lr)
            for seed in config.seeds:
                row, _ = _train_and_evaluate(
                    dataset, teacher, teacher_expl, hidden, seed, training, config.spike_factor
                )
                row.update({"architecture": arch, "procedure": config.capacity_procedure})
                rows.append(row)
    summary = {}
    for arch in CAPACITY_ARCHITECTURES:
        for lr in config.learning_rates:
            sel = [r for r in rows if r["architecture"] == arch and r["learning_rate"] == lr]
            summary[f"{arch}|lr{lr:g}"] = {
                "parameter_count": sel[0]["parameter_count"],
                "checkpoint_bytes": sel[0]["checkpoint_bytes"],
                **_group_summary(sel, 1e-4),
            }
    return {"procedure": config.capacity_procedure, "rows": rows, "summary": summary}


def run_benchmarks(config: StudyConfig, dataset: TeacherDataset) -> dict[str, Any]:
    teacher = table_from_dataset(dataset)
    training = training_config(config.capacity_procedure, config.epochs, config.learning_rates[-1])
    results = {}
    for arch, hidden in CAPACITY_ARCHITECTURES.items():
        model, _ = train_policy(dataset, PolicyConfig(hidden, config.seeds[0]), training)
        results[arch] = {
            "parameter_count": parameter_count(model),
            "checkpoint_bytes": len(checkpoint_bytes(model)),
            **benchmark_policies(
                model, teacher, number=config.benchmark_number, repeats=config.benchmark_repeats
            ),
        }
    return results


def run_studies(
    config: StudyConfig,
    output_dir: Path,
    studies: Sequence[str] = ("stability", "capacity", "benchmark"),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_teacher(iterations=config.teacher_iterations)
    teacher = table_from_dataset(dataset)
    teacher_expl = exploitability(teacher)
    header = {
        "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
        "metadata": {**collect_metadata(), "torch_version": torch.__version__},
        "teacher": {**dict(dataset.teacher), "exploitability": teacher_expl},
    }
    out: dict[str, Any] = {"plots": {}}
    if "stability" in studies:
        result = {**header, **run_stability_study(config, dataset, teacher, teacher_expl)}
        (output_dir / "stability.json").write_text(json.dumps(result, indent=1) + "\n")
        out["stability"] = result
        out["plots"]["stability"] = plot_stability(result, output_dir / "stability.png")
    if "capacity" in studies:
        result = {**header, **run_capacity_study(config, dataset, teacher, teacher_expl)}
        (output_dir / "capacity.json").write_text(json.dumps(result, indent=1) + "\n")
        out["capacity"] = result
        out["plots"]["capacity"] = plot_capacity(result, output_dir / "capacity.png")
    if "benchmark" in studies:
        result = {**header, "benchmarks": run_benchmarks(config, dataset)}
        (output_dir / "benchmarks.json").write_text(json.dumps(result, indent=1) + "\n")
        out["benchmark"] = result
    return out


# -- plots ------------------------------------------------------------------------------------

_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
_TEXT, _MUTED, _GRID, _REF = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8984"


def _pyplot() -> Any:
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
    return plt


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=_TEXT)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=_GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_stability(result: dict[str, Any], path: Path) -> str | None:
    plt = _pyplot()
    if plt is None:
        return None
    procedures = list(PROCEDURES)
    teacher = result["teacher"]["exploitability"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)

    ax = axes[0]
    combos = [(a, lr) for a in STABILITY_ARCHITECTURES for lr in result["config"]["learning_rates"]]
    width = 0.8 / len(combos)
    for c, (arch, lr) in enumerate(combos):
        for p, proc in enumerate(procedures):
            ys = [
                r["exploitability"]
                for r in result["rows"]
                if r["architecture"] == arch and r["learning_rate"] == lr and r["procedure"] == proc
            ]
            xs = [p - 0.4 + width * (c + 0.5)] * len(ys)
            ax.scatter(
                xs,
                ys,
                s=18,
                color=_SERIES[c],
                alpha=0.8,
                label=f"{arch}, lr {lr:g}" if p == 0 else None,
                linewidths=0,
            )
    ax.axhline(teacher, color=_REF, ls="--", lw=1.2, label=f"teacher {teacher:.6f}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(procedures)), procedures, rotation=15)
    _style(ax, "Final exploitability per seed", "training procedure", "chips per hand")
    ax.legend(frameon=False, fontsize=8, labelcolor=_TEXT)

    ax = axes[1]
    key_base = next(k for k in result["curves"] if k.startswith("mlp16x16_lr0.03_baseline"))
    key_cos = key_base.replace("_baseline_", "_cosine_")
    for color, key, label in (
        (_SERIES[0], key_base, "constant lr (M2A)"),
        (_SERIES[1], key_cos, "cosine decay"),
    ):
        kl = [max(v, 1e-12) for v in result["curves"][key]["kl"]]
        ax.plot(range(1, len(kl) + 1), kl, color=color, lw=1.6, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, f"Training KL, {key_base.split('_baseline')[0]} seed 0", "epoch", "mean KL (nats)")
    ax.legend(frameon=False, fontsize=9, labelcolor=_TEXT)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return str(path)


def plot_capacity(result: dict[str, Any], path: Path) -> str | None:
    plt = _pyplot()
    if plt is None:
        return None
    teacher = result["teacher"]["exploitability"]
    lrs = sorted({r["learning_rate"] for r in result["rows"]})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, metric, title, ylabel in (
        (axes[0], "max_abs_error", "Largest probability error vs teacher", "max |q − p|"),
        (axes[1], "exploitability", "Exact exploitability", "chips per hand"),
    ):
        for i, lr in enumerate(lrs):
            for arch in CAPACITY_ARCHITECTURES:
                rows = [
                    r
                    for r in result["rows"]
                    if r["architecture"] == arch and r["learning_rate"] == lr
                ]
                n = rows[0]["parameter_count"] * (1.0 + 0.06 * (i - 0.5))
                ys = [r[metric] for r in rows]
                ax.scatter(
                    [n] * len(ys),
                    ys,
                    s=16,
                    color=_SERIES[i],
                    alpha=0.6,
                    linewidths=0,
                    label=f"lr {lr:g}" if arch == "linear" else None,
                )
                ax.plot([n], [statistics.median(ys)], marker="_", ms=16, mew=2.5, color=_SERIES[i])
        for k, arch in enumerate(CAPACITY_ARCHITECTURES):
            n = next(r["parameter_count"] for r in result["rows"] if r["architecture"] == arch)
            ax.annotate(
                arch,
                (n, 1.0 if k % 2 else 0.955),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=8,
                color=_MUTED,
            )
        if metric == "exploitability":
            ax.axhline(teacher, color=_REF, ls="--", lw=1.2, label="teacher")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(ax, title, "parameters (log)", ylabel)
        ax.legend(
            frameon=False,
            fontsize=9,
            labelcolor=_TEXT,
            title="dots: seeds, bar: median",
            title_fontsize=8,
        )
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return str(path)
