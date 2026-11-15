"""M5: how much can we learn about an unfamiliar Leduc opponent from N hands?

Protocol (``docs/design/opponent_modeling.md``):

1. **Population.** Development and held-out opponents (``opponents/population.py``).
   The population prior is built from the development stationary opponents.
2. **Tuning (development only).** For each development opponent and seed,
   the model observes 500 hands against the hero. Every candidate predictor
   (prior × concentrations × back-off) is scored exactly at each budget; the
   configuration with the lowest mean excess log loss is selected.
3. **Sample efficiency (held-out).** For each held-out stationary opponent and
   seed: an *estimation* stream of 500 hands and an independent *evaluation*
   stream of 500 future hands. Snapshots after 0, 10, 20, 50, 100, 200, 500
   hands are scored on the future hands (sampled) and in exact expectation.
4. **Changing opponents.** Switching opponents play 500 hands; cumulative
   counts, exponential forgetting and CUSUM reset (each tuned on development
   opponents) are compared after the switch, and on stationary opponents for
   false alarms and the cost of forgetting.

Everything is deterministic given the seeds. Artifacts go to a run directory
(default ``runs/m5-opponent-modeling``, git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from pokermind.games.leduc import LeducGame
from pokermind.opponents.bayes import (
    ChangeDetectingModel,
    DetectorConfig,
    ModelConfig,
    OpponentModel,
    PopulationPrior,
    raw_frequency,
)
from pokermind.opponents.context import PublicHandRecord, context_class
from pokermind.opponents.exact import (
    TRAITS,
    ContextStats,
    Prediction,
    context_stats,
    expected_scores,
    weighted_trait,
)
from pokermind.opponents.play import play_records
from pokermind.opponents.population import Opponent, build_population
from pokermind.simulation import collect_metadata
from pokermind.strategy import GameStrategy, game_strategy_digest, load_game_strategy

BUDGETS: Final = (0, 10, 20, 50, 100, 200, 500)
TUNING_BUDGETS: Final = (10, 20, 50, 100, 200, 500)
CHANGE_OFFSETS: Final = (0, 10, 20, 50, 100, 200)
"""Hands after the switch at which the change study is scored (0 = just before)."""
LOG_FLOOR: Final = 1e-12
CALIBRATION_BINS: Final = 10
Predictor = Callable[[str, tuple[int, ...]], Prediction]
Table = dict[str, Prediction]


@dataclass(frozen=True, slots=True)
class OpponentStudyConfig:
    hero_path: Path
    seeds: int = 30
    tuning_seeds: int = 10
    per_family: int = 4
    switching_per_split: int = 8
    estimation_hands: int = 500
    evaluation_hands: int = 500
    budgets: tuple[int, ...] = BUDGETS
    alpha_fine_grid: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    alpha_coarse_grid: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
    decay_grid: tuple[float, ...] = (0.995, 0.99, 0.98, 0.95)
    detector_grid: tuple[DetectorConfig, ...] = tuple(
        DetectorConfig(s, h, d)
        for s in (0.8, 0.9)
        for h in (3.0, 5.0, 8.0, 12.0)
        for d in (0.0, 0.02)
    )

    def __post_init__(self) -> None:
        if self.seeds <= 0 or self.tuning_seeds <= 0:
            raise ValueError("seeds must be positive")
        if max(self.budgets) > self.estimation_hands:
            raise ValueError("budgets cannot exceed estimation_hands")


def stream_id(*parts: str | int) -> int:
    digest = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:7], "little")


def uniform_predict(context: str, legal: tuple[int, ...]) -> Prediction:
    return np.full(len(legal), 1.0 / len(legal))


def laplace(model: OpponentModel) -> Predictor:
    """Add-one smoothing per context (no back-off, no prior information)."""

    def predict(context: str, legal: tuple[int, ...]) -> Prediction:
        vec = model.counts[0].get(context)
        n = np.zeros(len(legal)) if vec is None else vec
        out: Prediction = (n + 1.0) / (n.sum() + len(legal))
        return out

    return predict


def tabulate(stats: ContextStats, predict: Predictor) -> Table:
    return {ctx: predict(ctx, stats.legal[ctx]) for ctx in stats.weight}


def oracle_predictor(stats: ContextStats) -> Predictor:
    """The exact public-context marginal (the best public-information prediction)."""
    return lambda ctx, legal: stats.marginal[ctx]


def table_predictor(table: Table) -> Predictor:
    return lambda ctx, legal: table[ctx]


# -- statistics helpers -------------------------------------------------------------------


def mean_ci(values: Sequence[float]) -> dict[str, float]:
    """Mean, a normal-approximation 95 % CI, sd and the 10th/90th percentiles."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": float(arr.mean()),
        "ci_low": float(arr.mean() - half),
        "ci_high": float(arr.mean() + half),
        "sd": sd,
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n": n,
    }


def clustered(values: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Mean over clusters (opponents) of per-cluster means, CI over clusters,
    plus the run-level 10th/90th percentiles."""
    per = [statistics.fmean(v) for v in values.values()]
    flat = [x for v in values.values() for x in v]
    out = mean_ci(per)
    out["p10"] = float(np.percentile(flat, 10))
    out["p90"] = float(np.percentile(flat, 90))
    out["runs"] = len(flat)
    return out


# -- the study ----------------------------------------------------------------------------


@dataclass(slots=True)
class Context:
    config: OpponentStudyConfig
    game: LeducGame
    hero: GameStrategy
    population: dict[str, list[Opponent]]
    stats: dict[str, tuple[ContextStats, ...]]
    population_prior: PopulationPrior
    uniform_prior: PopulationPrior


def _setup(config: OpponentStudyConfig) -> Context:
    game = LeducGame()
    hero = load_game_strategy(config.hero_path, game.name, game.action_codes)
    population = build_population(game, hero, config.per_family, config.switching_per_split)
    stats = {
        o.name: tuple(context_stats(game, hero, p) for p in o.policies)
        for split in population.values()
        for o in split
    }
    dev = [stats[o.name][0] for o in population["development"] if not o.is_switching]
    return Context(
        config,
        game,
        hero,
        population,
        stats,
        PopulationPrior.from_opponents(dev, "development stationary opponents"),
        PopulationPrior.uniform(),
    )


def _estimation_snapshots(
    ctx: Context, opponent: Opponent, seed: int, budgets: Sequence[int], purpose: str
) -> tuple[dict[int, OpponentModel], list[PublicHandRecord]]:
    records = play_records(
        ctx.game,
        ctx.hero,
        opponent.policy_at,
        max(budgets),
        stream_id(opponent.name, seed, purpose),
    )
    model = OpponentModel(ModelConfig())
    snaps = {0: model.snapshot()} if 0 in budgets else {}
    for n, record in enumerate(records, start=1):
        model.observe_hand(record)
        if n in budgets:
            snaps[n] = model.snapshot()
    return snaps, records


def _candidates(config: OpponentStudyConfig) -> list[tuple[str, ModelConfig, str]]:
    out = []
    for prior in ("population", "uniform"):
        for a in config.alpha_fine_grid:
            out.append((f"flat_{prior}_a{a:g}", ModelConfig(a, 1.0, hierarchical=False), prior))
            for c in config.alpha_coarse_grid:
                out.append((f"hier_{prior}_a{a:g}_c{c:g}", ModelConfig(a, c), prior))
    return out


def tune(ctx: Context) -> dict[str, Any]:
    """Score every candidate on development opponents; pick the best per family of rules."""
    cfg = ctx.config
    candidates = _candidates(cfg)
    scores: dict[str, list[float]] = {name: [] for name, _, _ in candidates}
    by_budget: dict[str, dict[int, list[float]]] = {
        name: {b: [] for b in TUNING_BUDGETS} for name, _, _ in candidates
    }
    dev = [o for o in ctx.population["development"] if not o.is_switching]
    for opponent in dev:
        stats = ctx.stats[opponent.name][0]
        for seed in range(cfg.tuning_seeds):
            snaps, _ = _estimation_snapshots(ctx, opponent, seed, TUNING_BUDGETS, "tuning")
            for name, model_cfg, prior_name in candidates:
                prior = ctx.population_prior if prior_name == "population" else ctx.uniform_prior
                for b in TUNING_BUDGETS:
                    model = snaps[b].with_config(model_cfg, prior)
                    excess = expected_scores(stats, model.predict).excess_log_loss
                    scores[name].append(excess)
                    by_budget[name][b].append(excess)
    mean = {name: statistics.fmean(v) for name, v in scores.items()}

    def best(prefix: str) -> str:
        return min((n for n in mean if n.startswith(prefix)), key=lambda n: mean[n])

    chosen = {
        "main": best(""),
        "hier_population": best("hier_population"),
        "hier_uniform": best("hier_uniform"),
        "flat_population": best("flat_population"),
        "flat_uniform": best("flat_uniform"),
    }
    lookup = {name: (model_cfg, prior) for name, model_cfg, prior in candidates}
    return {
        "objective": "mean exact excess log loss over budgets "
        f"{list(TUNING_BUDGETS)} on development stationary opponents",
        "development_runs": len(dev) * cfg.tuning_seeds,
        "selected": {
            role: {
                "candidate": name,
                "alpha_fine": lookup[name][0].alpha_fine,
                "alpha_coarse": lookup[name][0].alpha_coarse,
                "hierarchical": lookup[name][0].hierarchical,
                "prior": lookup[name][1],
                "mean_excess": mean[name],
            }
            for role, name in chosen.items()
        },
        "ranking": sorted(
            (
                {
                    "candidate": n,
                    "mean_excess": mean[n],
                    "by_budget": {str(b): statistics.fmean(v) for b, v in by_budget[n].items()},
                }
                for n in mean
            ),
            key=lambda r: r["mean_excess"],
        ),
        "_configs": {role: lookup[name] for role, name in chosen.items()},
    }


def _predictors(
    ctx: Context, snap: OpponentModel, chosen: Mapping[str, tuple[ModelConfig, str]]
) -> dict[str, Predictor]:
    def model_for(role: str) -> OpponentModel:
        model_cfg, prior_name = chosen[role]
        prior = ctx.population_prior if prior_name == "population" else ctx.uniform_prior
        return snap.with_config(model_cfg, prior)

    main_cfg, _ = chosen["main"]
    prior_only = OpponentModel(main_cfg, ctx.population_prior)
    return {
        "uniform": uniform_predict,
        "population_prior": prior_only.predict,
        "raw_frequency": lambda c, legal: raw_frequency(snap, c, legal),
        "laplace": laplace(snap),
        "flat_population": model_for("flat_population").predict,
        "hier_population": model_for("hier_population").predict,
        "hier_uniform": model_for("hier_uniform").predict,
        "main": model_for("main").predict,
    }


def _sampled_scores(
    table: Table, future: Sequence[PublicHandRecord]
) -> tuple[float, float, float, int]:
    ll = brier = acc = 0.0
    n = 0
    for record in future:
        for d in record.decisions:
            q = table[d.context]
            i = d.legal_actions.index(d.action)
            ll -= math.log(max(float(q[i]), LOG_FLOOR))
            target = np.zeros(len(q))
            target[i] = 1.0
            brier += float(((q - target) ** 2).sum())
            acc += float(int(np.argmax(q)) == i)
            n += 1
    return ll / n, brier / n, acc / n, n


def sample_efficiency(
    ctx: Context, chosen: Mapping[str, tuple[ModelConfig, str]]
) -> dict[str, Any]:
    cfg = ctx.config
    budgets = cfg.budgets
    metrics = ("excess_log_loss", "log_loss", "brier", "accuracy",
               "sampled_log_loss", "sampled_brier", "sampled_accuracy")  # fmt: skip
    # results[split][predictor][budget][metric][opponent] -> list over seeds
    results: dict[str, Any] = {}
    calibration: dict[int, list[tuple[float, float]]] = {b: [] for b in budgets}
    coverage: dict[int, dict[str, list[float]]] = {b: {} for b in budgets}
    posterior_sd: dict[int, list[float]] = {b: [] for b in budgets}
    classes: dict[int, dict[str, dict[str, list[float]]]] = {b: {} for b in budgets}
    traits: dict[str, dict[int, list[float]]] = {t: {b: [] for b in budgets} for t in TRAITS}
    oracle: dict[str, dict[str, list[float]]] = {"oracle_log_loss": {}, "card_aware_log_loss": {}}
    families: dict[str, dict[int, dict[str, list[float]]]] = {}
    seeds_by_split = {"heldout": cfg.seeds, "development": cfg.tuning_seeds}

    for split in ("heldout", "development"):
        split_res: dict[str, dict[int, dict[str, dict[str, list[float]]]]] = {}
        for opponent in (o for o in ctx.population[split] if not o.is_switching):
            stats = ctx.stats[opponent.name][0]
            for seed in range(seeds_by_split[split]):
                snaps, _ = _estimation_snapshots(ctx, opponent, seed, budgets, "estimate")
                future = play_records(
                    ctx.game, ctx.hero, opponent.policy_at, cfg.evaluation_hands,
                    stream_id(opponent.name, seed, "evaluate"),
                )  # fmt: skip
                for b in budgets:
                    for name, predict in _predictors(ctx, snaps[b], chosen).items():
                        table = tabulate(stats, predict)
                        exact = expected_scores(stats, table_predictor(table))
                        sampled = _sampled_scores(table, future)
                        values = (
                            exact.excess_log_loss, exact.log_loss, exact.brier, exact.accuracy,
                            sampled[0], sampled[1], sampled[2],
                        )  # fmt: skip
                        slot = split_res.setdefault(name, {}).setdefault(b, {})
                        for metric, value in zip(metrics, values, strict=True):
                            slot.setdefault(metric, {}).setdefault(opponent.name, []).append(value)
                        if split != "heldout" or name != "main":
                            continue
                        fam = families.setdefault(opponent.family, {}).setdefault(b, {})
                        fam.setdefault(opponent.name, []).append(exact.excess_log_loss)
                        for record in future:
                            for d in record.decisions:
                                q = table[d.context]
                                for i, prob in enumerate(q):
                                    calibration[b].append(
                                        (float(prob), float(i == d.legal_actions.index(d.action)))
                                    )
                        main_cfg, _ = chosen["main"]
                        model = snaps[b].with_config(main_cfg, ctx.population_prior)
                        cov = sd_sum = w_sum = 0.0
                        per_class: dict[str, float] = {}
                        per_class_w: dict[str, float] = {}
                        for c, w in stats.weight.items():
                            legal = stats.legal[c]
                            truth = stats.marginal[c]
                            inside = model.interval_contains(c, legal, truth)
                            cov += w * float(inside.mean())
                            sd_sum += w * float(model.posterior_std(c, legal).mean())
                            w_sum += w
                            q = table[c]
                            kl = float(
                                (
                                    truth
                                    * (
                                        np.log(np.maximum(truth, LOG_FLOOR))
                                        - np.log(np.maximum(q, LOG_FLOOR))
                                    )
                                ).sum()
                            )
                            k = context_class(c, legal)
                            per_class[k] = per_class.get(k, 0.0) + w * kl
                            per_class_w[k] = per_class_w.get(k, 0.0) + w
                        coverage[b].setdefault(opponent.name, []).append(cov / w_sum)
                        posterior_sd[b].append(sd_sum / w_sum)
                        for k, v in per_class.items():
                            entry = classes[b].setdefault(
                                k, {"excess": [], "opportunities": [], "weight": []}
                            )
                            entry["excess"].append(v / per_class_w[k])
                            entry["weight"].append(per_class_w[k])
                            entry["opportunities"].append(
                                sum(snaps[b].opportunities(c) for c in stats.weight
                                    if context_class(c, stats.legal[c]) == k)
                            )  # fmt: skip
                        for trait in TRAITS:
                            true = weighted_trait(stats, oracle_predictor(stats), trait)
                            est = weighted_trait(stats, table_predictor(table), trait)
                            traits[trait][b].append(abs(est - true))
            if split == "heldout":
                e = expected_scores(stats, oracle_predictor(stats))
                oracle["oracle_log_loss"][opponent.name] = [e.oracle_log_loss]
                oracle["card_aware_log_loss"][opponent.name] = [e.card_aware_log_loss]
        results[split] = split_res

    def summarize(split: str) -> dict[str, Any]:
        return {
            name: {
                str(b): {m: clustered(v) for m, v in per_metric.items()}
                for b, per_metric in per_budget.items()
            }
            for name, per_budget in results[split].items()
        }

    def calib(b: int) -> dict[str, Any]:
        pairs = np.array(calibration[b])
        edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
        idx = np.clip(np.digitize(pairs[:, 0], edges) - 1, 0, CALIBRATION_BINS - 1)
        bins, ece = [], 0.0
        for k in range(CALIBRATION_BINS):
            sel = idx == k
            if not sel.any():
                continue
            pred, obs = float(pairs[sel, 0].mean()), float(pairs[sel, 1].mean())
            bins.append(
                {"bin": k, "count": int(sel.sum()), "mean_predicted": pred, "observed": obs}
            )
            ece += sel.sum() / len(pairs) * abs(pred - obs)
        return {"ece": float(ece), "pairs": len(pairs), "bins": bins}

    return {
        "heldout": summarize("heldout"),
        "development": summarize("development"),
        "oracle": {k: clustered(v) for k, v in oracle.items()},
        "calibration": {str(b): calib(b) for b in budgets},
        "credible_interval_coverage_90": {str(b): clustered(v) for b, v in coverage.items()},
        "posterior_sd": {str(b): mean_ci(v) for b, v in posterior_sd.items()},
        "by_context_class": {
            str(b): {
                k: {
                    "excess_log_loss": mean_ci(e["excess"]),
                    "opportunities": mean_ci(e["opportunities"]),
                    "decisions_per_hand": mean_ci(e["weight"]),
                }
                for k, e in sorted(per.items())
            }
            for b, per in classes.items()
        },
        "trait_abs_error": {
            t: {str(b): mean_ci(v) for b, v in per.items()} for t, per in traits.items()
        },
        "by_family_main_excess": {
            fam: {str(b): clustered(v) for b, v in per.items()} for fam, per in families.items()
        },
    }


# -- changing opponents -------------------------------------------------------------------


Strategy = tuple[str, Callable[[], Any]]


def _change_strategies(
    ctx: Context, main: ModelConfig, decays: Sequence[float], detectors: Sequence[DetectorConfig]
) -> list[Strategy]:
    prior = ctx.population_prior

    def decayed(g: float) -> Callable[[], Any]:
        return lambda: OpponentModel(replace(main, decay=g), prior)

    def detecting(d: DetectorConfig) -> Callable[[], Any]:
        return lambda: ChangeDetectingModel(main, d, prior)

    out: list[Strategy] = [("cumulative", decayed(1.0))]
    out += [(f"decay{g:g}", decayed(g)) for g in decays]
    out += [(d.name, detecting(d)) for d in detectors]
    return out


def _run_change(
    ctx: Context, opponent: Opponent, seed: int, strategies: Sequence[Strategy], purpose: str
) -> dict[str, dict[str, Any]]:
    horizon = ctx.config.estimation_hands
    records = play_records(
        ctx.game, ctx.hero, opponent.policy_at, horizon, stream_id(opponent.name, seed, purpose)
    )
    switch = opponent.switch_hand
    checkpoints = (
        sorted({switch + o for o in CHANGE_OFFSETS if switch + o <= horizon})
        if switch is not None
        else [round(horizon * k / 5) for k in range(1, 6)]
    )
    out: dict[str, dict[str, Any]] = {}
    for name, factory in strategies:
        model = factory()
        excess: dict[int, float] = {}
        prequential: list[tuple[int, float]] = []
        for n, record in enumerate(records, start=1):
            for d in record.decisions:
                q = model.predict(d.context, d.legal_actions)
                i = d.legal_actions.index(d.action)
                prequential.append((record.hand_index, -math.log(max(float(q[i]), LOG_FLOOR))))
            model.observe_hand(record)
            if n in checkpoints:
                phase = 1 if switch is not None and n >= switch else 0
                excess[n] = expected_scores(
                    ctx.stats[opponent.name][phase], model.predict
                ).excess_log_loss
        alarms = list(getattr(model, "alarms", []))
        out[name] = {"excess": excess, "prequential": prequential, "alarms": alarms}
    return out


def change_study(ctx: Context, main: ModelConfig) -> dict[str, Any]:
    cfg = ctx.config
    # 1. Tune decay and detector on development opponents (switching + stationary).
    dev_switch = [o for o in ctx.population["development"] if o.is_switching]
    dev_stat = [o for o in ctx.population["development"] if not o.is_switching]
    tuning = _change_strategies(ctx, main, cfg.decay_grid, cfg.detector_grid)
    objective: dict[str, list[float]] = {name: [] for name, _ in tuning}
    for opponent in dev_switch + dev_stat:
        for seed in range(cfg.tuning_seeds):
            runs = _run_change(ctx, opponent, seed, tuning, "change-tuning")
            for name, r in runs.items():
                if r["excess"]:
                    objective[name].append(statistics.fmean(r["excess"].values()))
    mean_obj = {n: statistics.fmean(v) for n, v in objective.items()}
    best_decay = min((n for n in mean_obj if n.startswith("decay")), key=lambda n: mean_obj[n])
    best_cusum = min((n for n in mean_obj if n.startswith("cusum")), key=lambda n: mean_obj[n])
    decay_value = float(best_decay.removeprefix("decay"))
    detector = next(d for d in cfg.detector_grid if d.name == best_cusum)
    selected = _change_strategies(ctx, main, (decay_value,), (detector,))

    # 2. Held-out switching and stationary opponents.
    switching = [o for o in ctx.population["heldout"] if o.is_switching]
    stationary = [o for o in ctx.population["heldout"] if not o.is_switching]
    after: dict[str, dict[int, dict[str, list[float]]]] = {}
    windows: dict[str, dict[str, dict[str, list[float]]]] = {}
    delays: list[float] = []
    missed = pre_switch_alarms = 0
    switch_runs = 0
    window_spec = ((-50, 0), (0, 20), (20, 50), (50, 100), (100, 200))
    for opponent in switching:
        assert opponent.switch_hand is not None
        t = opponent.switch_hand
        for seed in range(cfg.seeds):
            runs = _run_change(ctx, opponent, seed, selected, "change")
            switch_runs += 1
            for name, r in runs.items():
                for n, v in r["excess"].items():
                    after.setdefault(name, {}).setdefault(n - t, {}).setdefault(
                        opponent.name, []
                    ).append(v)
                for lo, hi in window_spec:
                    vals = [x for h, x in r["prequential"] if t + lo <= h < t + hi]
                    if vals:
                        windows.setdefault(name, {}).setdefault(f"{lo}..{hi}", {}).setdefault(
                            opponent.name, []
                        ).append(statistics.fmean(vals))
                if name.startswith("cusum"):
                    pre_switch_alarms += sum(1 for a in r["alarms"] if a < t)
                    post = [a for a in r["alarms"] if a >= t]
                    if post:
                        delays.append(post[0] - t + 1)
                    else:
                        missed += 1
    stationary_alarms: list[float] = []
    stationary_cost: dict[str, dict[str, list[float]]] = {}
    for opponent in stationary:
        for seed in range(cfg.seeds):
            runs = _run_change(ctx, opponent, seed, selected, "change")
            for name, r in runs.items():
                stationary_cost.setdefault(name, {}).setdefault(opponent.name, []).append(
                    r["excess"][cfg.estimation_hands]
                )
                if name.startswith("cusum"):
                    stationary_alarms.append(len(r["alarms"]))
    hands_before_switch = sum((o.switch_hand or 0) for o in switching) * cfg.seeds
    return {
        "tuning": {
            "objective": "mean exact excess log loss over checkpoints, development "
            "switching and stationary opponents",
            "ranking": sorted(
                ({"strategy": n, "objective": v} for n, v in mean_obj.items()),
                key=lambda r: r["objective"],
            ),
            "selected_decay": decay_value,
            "selected_detector": {
                "short_decay": detector.short_decay,
                "threshold": detector.threshold,
                "drift": detector.drift,
            },
        },
        "heldout_switching_opponents": [o.name for o in switching],
        "excess_after_switch": {
            name: {str(k): clustered(v) for k, v in sorted(per.items())}
            for name, per in after.items()
        },
        "prequential_log_loss_windows": {
            name: {w: clustered(v) for w, v in per.items()} for name, per in windows.items()
        },
        "detection": {
            "runs": switch_runs,
            "detected": len(delays),
            "missed": missed,
            "delay_hands": mean_ci(delays) if delays else None,
            "delay_median": float(np.median(delays)) if delays else None,
            "pre_switch_false_alarms": pre_switch_alarms,
            "pre_switch_false_alarms_per_1000_hands": 1000
            * pre_switch_alarms
            / max(hands_before_switch, 1),
            "stationary_runs": len(stationary_alarms),
            "stationary_alarms_per_500_hands": mean_ci(stationary_alarms),
            "stationary_runs_with_any_alarm": sum(1 for a in stationary_alarms if a > 0),
        },
        "stationary_excess_at_500": {name: clustered(per) for name, per in stationary_cost.items()},
    }


def run_opponent_study(config: OpponentStudyConfig, output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    ctx = _setup(config)
    tuning = tune(ctx)
    chosen = tuning.pop("_configs")
    efficiency = sample_efficiency(ctx, chosen)
    change = change_study(ctx, chosen["main"][0])
    population = {
        split: [
            {
                "name": o.name,
                "family": o.family,
                "parameters": o.parameters,
                "switch_hand": o.switch_hand,
                "decisions_per_hand": [s.decisions_per_hand for s in ctx.stats[o.name]],
                "contexts_reached": [len(s.weight) for s in ctx.stats[o.name]],
                "policy_digests": [
                    game_strategy_digest(p, ctx.game.action_codes)[:16] for p in o.policies
                ],
            }
            for o in opponents
        ]
        for split, opponents in ctx.population.items()
    }
    document = {
        "experiment": "m5-opponent-modeling",
        "config": {
            "hero_path": str(config.hero_path),
            "hero_digest": game_strategy_digest(ctx.hero, ctx.game.action_codes),
            "seeds": config.seeds,
            "tuning_seeds": config.tuning_seeds,
            "per_family": config.per_family,
            "switching_per_split": config.switching_per_split,
            "estimation_hands": config.estimation_hands,
            "evaluation_hands": config.evaluation_hands,
            "budgets": list(config.budgets),
        },
        "metadata": collect_metadata(),
        "population": population,
        "tuning": tuning,
        "sample_efficiency": efficiency,
        "change": change,
        "elapsed_seconds": time.perf_counter() - start,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(document, indent=2) + "\n")
    document["plot"] = plot_study(document, output_dir / "opponent_modeling.png")
    return document


def plot_study(document: Mapping[str, Any], path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    eff = document["sample_efficiency"]
    budgets = document["config"]["budgets"]
    xs = [max(b, 1) for b in budgets]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for name in (
        "uniform",
        "population_prior",
        "laplace",
        "flat_population",
        "hier_uniform",
        "main",
    ):
        series = eff["heldout"][name]
        m = [series[str(b)]["excess_log_loss"]["mean"] for b in budgets]
        lo = [series[str(b)]["excess_log_loss"]["ci_low"] for b in budgets]
        hi = [series[str(b)]["excess_log_loss"]["ci_high"] for b in budgets]
        axes[0].plot(xs, m, marker="o", label=name)
        axes[0].fill_between(xs, lo, hi, alpha=0.2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("hands observed (0 plotted at 1)")
    axes[0].set_title("Held-out: excess log loss over the public oracle")
    axes[0].legend(fontsize=8)
    for fam, per in eff["by_family_main_excess"].items():
        axes[1].plot(xs, [per[str(b)]["mean"] for b in budgets], marker="o", label=fam)
    axes[1].set_xscale("log")
    axes[1].set_title("Main model by opponent family")
    axes[1].legend(fontsize=8)
    for b in (0, 20, 100, 500):
        if str(b) not in eff["calibration"]:
            continue
        bins = eff["calibration"][str(b)]["bins"]
        axes[2].plot(
            [x["mean_predicted"] for x in bins], [x["observed"] for x in bins], marker="o",
            label=f"{b} hands (ECE {eff['calibration'][str(b)]['ece']:.3f})",
        )  # fmt: skip
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[2].set_title("Calibration on future hands (main)")
    axes[2].legend(fontsize=8)
    change = document["change"]["excess_after_switch"]
    for name, per in change.items():
        ks = sorted(int(k) for k in per)
        axes[3].plot(ks, [per[str(k)]["mean"] for k in ks], marker="o", label=name)
    axes[3].set_title("Switching opponents: excess log loss after the switch")
    axes[3].set_xlabel("hands since switch")
    if change:
        axes[3].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
