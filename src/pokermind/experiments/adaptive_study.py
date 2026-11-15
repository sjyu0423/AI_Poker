"""M6: does opponent adaptation beat fixed CFR with controlled exploitability?

Protocol (``docs/design/adaptive_strategy.md`` §5):

1. **Development pass.** Development opponents (M5 population), several
   seeds, every budget. Adapter hyperparameters (fixed α, evidence k,
   confidence k, gate z) are chosen by the mean exact payoff gain over CFR,
   with the exploitability budget enforced.
2. **Frozen held-out pass.** The tuned settings are frozen. Held-out
   opponents (same families, unseen parameters) and out-of-population
   opponents (families never seen in development) are evaluated.
3. **Changing opponents.** M5's held-out switching opponents and two
   bait-and-switch opponents, with and without change detection.

Hero observes hands from one stream and is evaluated **exactly** against the
true opponent policy, which only the evaluation and oracle code read. All
work is deterministic given the seeds; tasks run in parallel processes.
Artifacts go to ``runs/m6-adaptive-strategy`` (git-ignored).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from pokermind.adaptive.adapter import (
    AdaptationSettings,
    Adapter,
    ChangeAwareObserver,
    Priors,
    Snapshot,
    build_priors,
)
from pokermind.adaptive.reconstruction import (
    ObserverHandRecord,
    expand,
    infoset_contexts,
    observer_record,
    opponent_infoset_weights,
    reconstruction_error,
)
from pokermind.adaptive.response import (
    Mixer,
    hero_best_response,
    hero_value,
)
from pokermind.adaptive.robust_opponents import (
    OUT_OF_POPULATION,
    make_bait_switch,
    make_out_of_population,
)
from pokermind.agents.game_agents import GameStrategyAgent
from pokermind.evaluation.game_exact import GameInfoSet, information_sets
from pokermind.experiments.latency import time_statement
from pokermind.experiments.opponent_study import clustered, mean_ci, stream_id
from pokermind.games.leduc import LeducGame
from pokermind.opponents.bayes import PopulationPrior
from pokermind.opponents.exact import context_stats, expected_scores
from pokermind.opponents.play import play_hand
from pokermind.opponents.population import Opponent, build_population
from pokermind.simulation import collect_metadata
from pokermind.strategy import GameStrategy, game_strategy_digest, load_game_strategy

BUDGETS: Final = (0, 10, 20, 50, 100, 200, 500)
CHANGE_OFFSETS: Final = (0, 10, 20, 50, 100, 200)
DEVELOPMENT_SETTINGS: Final = AdaptationSettings(confidence_k=(10.0, 3.0, 30.0))
"""Settings for the development pass; the first confidence k is the default."""
METHODS: Final = (
    "cfr", "br_naive", "br_map", "budget_max", "fixed_mix", "evidence_mix",
    "confidence_budget", "gated",
)  # fmt: skip
RESPONSE_OF: Final = {
    "br_naive": "br_naive",
    "br_map": "br_map",
    "budget_max": "br_map",
    "fixed_mix": "br_map",
    "evidence_mix": "br_map",
    "confidence_budget": "br_confidence",
}


@dataclass(frozen=True, slots=True)
class AdaptiveStudyConfig:
    hero_path: Path
    development_seeds: int = 3
    seeds: int = 6
    change_seeds: int = 6
    budgets: tuple[int, ...] = BUDGETS
    settings: AdaptationSettings = DEVELOPMENT_SETTINGS
    fixed_alpha_grid: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    evidence_k_grid: tuple[float, ...] = (10.0, 30.0, 100.0, 300.0)
    gate_z_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    per_family: int = 4
    out_of_population_per_family: int = 2
    switching: int = 8
    workers: int = 0
    """0 = one per CPU."""

    def __post_init__(self) -> None:
        if min(self.seeds, self.development_seeds, self.change_seeds) <= 0:
            raise ValueError("seed counts must be positive")


# -- worker context ---------------------------------------------------------------------------


@dataclass(slots=True)
class World:
    game: LeducGame
    specs: dict[str, GameInfoSet]
    cfr: GameStrategy
    priors: Priors
    opponents: dict[str, Opponent]
    groups: dict[str, list[str]]
    oracle: dict[tuple[str, int], dict[str, Any]]


_WORLD: World | None = None


def build_world(hero_path: Path, per_family: int, oop_per_family: int, switching: int) -> World:
    game = LeducGame()
    specs = information_sets(game)
    cfr = load_game_strategy(hero_path, game.name, game.action_codes)
    population = build_population(game, cfr, per_family, switching)
    dev = [o for o in population["development"] if not o.is_switching]
    public_prior = PopulationPrior.from_opponents(
        [context_stats(game, cfr, o.policies[0]) for o in dev], "development stationary"
    )
    priors = build_priors(
        game, specs, cfr, [o.policies[0] for o in dev], public_prior, infoset_contexts(game)
    )
    opponents: dict[str, Opponent] = {}
    groups: dict[str, list[str]] = {}

    def add(group: str, opponent: Opponent) -> None:
        opponents[opponent.name] = opponent
        groups.setdefault(group, []).append(opponent.name)

    for o in population["development"]:
        add("development_switching" if o.is_switching else "development", o)
    for o in population["heldout"]:
        add("heldout_switching" if o.is_switching else "heldout", o)
    for family in OUT_OF_POPULATION:
        for i in range(oop_per_family):
            add("out_of_population", make_out_of_population(family, i, specs))
    for i in range(2):
        add("bait_switch", make_bait_switch(i, specs))
    return World(game, specs, cfr, priors, opponents, groups, {})


def _init_worker(hero_path: Path, per_family: int, oop: int, switching: int) -> None:
    global _WORLD
    _WORLD = build_world(hero_path, per_family, oop, switching)


def _world() -> World:
    if _WORLD is None:
        raise RuntimeError("worker not initialized")
    return _WORLD


def oracle_for(world: World, name: str, phase: int, budget: float) -> dict[str, Any]:
    """Evaluation-only benchmarks from the true policy (cached per worker)."""
    key = (name, phase)
    if key not in world.oracle:
        truth = world.opponents[name].policies[phase]
        g, specs = world.game, world.specs
        br = hero_best_response(g, truth, specs)
        mixer = Mixer(g, specs, world.cfr, br, world.priors.cfr_exploitability)
        search = mixer.largest_alpha(world.priors.cfr_exploitability + budget)
        cfr_value = hero_value(g, world.cfr, truth, specs)
        world.oracle[key] = {
            "cfr_value": cfr_value,
            "oracle_br_gain": hero_value(g, br, truth, specs) - cfr_value,
            "oracle_br_exploitability": mixer.exploitability(1.0),
            "oracle_budget_alpha": search.alpha,
            "oracle_budget_gain": hero_value(g, mixer.policy(search.alpha), truth, specs)
            - cfr_value,
            "oracle_budget_exploitability": search.exploitability,
        }
    return world.oracle[key]


def observe(world: World, opponent: Opponent, seed: int, hands: int) -> list[ObserverHandRecord]:
    stream = stream_id(opponent.name, seed, "observe")
    return [
        observer_record(
            world.game, play_hand(world.game, world.cfr, opponent.policy_at(h), h, stream), h % 2, h
        )
        for h in range(hands)
    ]


def _method_exploitability(snap: Snapshot, method: str, cfr_expl: float) -> float:
    if method in ("cfr",) or (method == "gated" and snap.alphas["gated"] == 0.0):
        return cfr_expl
    if method == "gated":
        name = "br_map" if snap.gate["chosen"] == "budget_max" else "br_confidence"
        return snap.responses[name].mixer.exploitability(snap.alphas["gated"])
    if method == "br_naive":
        return snap.responses["br_naive"].budget.exploitability
    return snap.responses[RESPONSE_OF[method]].mixer.exploitability(snap.alphas[method])


def evaluate_snapshot(
    world: World, snap: Snapshot, truth: GameStrategy, cfr_value: float, extended: bool
) -> dict[str, Any]:
    g, specs = world.game, world.specs
    cfr_expl = world.priors.cfr_exploitability
    row: dict[str, Any] = {"hands": snap.hands, "prior": list(snap.prior), "alphas": snap.alphas}
    methods = {}
    for m in METHODS:
        value = hero_value(g, snap.policies[m], truth, specs)
        methods[m] = {
            "gain": value - cfr_value,
            "exploitability": _method_exploitability(snap, m, cfr_expl),
        }
    row["methods"] = methods
    gate = snap.gate
    row["gate"] = {
        "open": bool(gate["open"]),
        "chosen": gate["chosen"],
        "candidates": {
            k: {"mean": v["mean"], "sd": v["sd"], "lcb": v["lcb"]}
            for k, v in gate["candidates"].items()
        },
    }
    if not extended:
        return row
    frontier: dict[str, dict[str, Any]] = {}
    for name in ("br_map", "br_confidence"):
        res = snap.responses[name]
        frontier[name] = {
            str(b): {
                "alpha": s.alpha,
                "exploitability": s.exploitability,
                "gain": hero_value(g, res.mixer.policy(s.alpha), truth, specs) - cfr_value,
            }
            for b, s in res.frontier.items()
        }
    row["frontier"] = frontier
    # true gains of every gate candidate (for tuning z and checking calibration)
    true_gain = {}
    for name in gate["candidates"]:
        resp = (
            "br_map"
            if name == "budget_max"
            else "br_confidence" + name.removeprefix("confidence_budget")
        )
        res = snap.responses[resp]
        true_gain[name] = (
            hero_value(g, res.mixer.policy(res.budget.alpha), truth, specs) - cfr_value
        )
        row["gate"]["candidates"][name]["alpha"] = res.budget.alpha
    row["gate"]["true_gain"] = true_gain
    # map response, unconstrained and its value (for linear tuning of α rules)
    row["br_map_gain"] = methods["br_map"]["gain"]
    row["budget_max_alpha"] = snap.alphas["budget_max"]
    weights = opponent_infoset_weights(g, world.cfr, truth)
    pri = world.priors
    row["reconstruction"] = {
        "estimate": reconstruction_error(snap.estimate, truth, weights),
        "population_prior": reconstruction_error(
            expand(pri.population_theta, specs), truth, weights
        ),
        "cfr_prior": reconstruction_error(expand(pri.cfr_theta, specs), truth, weights),
    }
    true_stats = context_stats(g, world.cfr, truth)
    implied = context_stats(g, world.cfr, snap.estimate)
    row["public_excess_log_loss_em"] = expected_scores(
        true_stats, lambda c, legal: implied.marginal[c]
    ).excess_log_loss
    row["timings"] = snap.timings
    return row


def snapshot_task(
    args: tuple[str, int, AdaptationSettings, tuple[int, ...], bool],
) -> dict[str, Any]:
    name, seed, settings, budgets, extra = args
    world = _world()
    opponent = world.opponents[name]
    truth = opponent.policies[0]
    oracle = oracle_for(world, name, 0, settings.budget)
    adapter = Adapter(world.game, world.specs, world.priors, settings)
    records = observe(world, opponent, seed, max(budgets))
    rows = []
    for b in budgets:
        snap = adapter.snapshot(records[:b], extra_confidence=extra)
        row = evaluate_snapshot(world, snap, truth, oracle["cfr_value"], extended=True)
        rows.append(row)
    return {
        "opponent": name,
        "family": opponent.family,
        "seed": seed,
        "oracle": oracle,
        "rows": rows,
    }


def change_task(args: tuple[str, int, AdaptationSettings, bool]) -> dict[str, Any]:
    name, seed, settings, detect = args
    world = _world()
    opponent = world.opponents[name]
    assert opponent.switch_hand is not None
    switch = opponent.switch_hand
    horizon = switch + max(CHANGE_OFFSETS)
    checkpoints = sorted({switch + o for o in CHANGE_OFFSETS if switch + o <= horizon})
    records = observe(world, opponent, seed, max(checkpoints))
    observer = ChangeAwareObserver(world.priors.public_prior, detect=detect)
    adapter = Adapter(world.game, world.specs, world.priors, settings)
    out: dict[str, Any] = {"opponent": name, "seed": seed, "detect": detect, "checkpoints": {}}
    for n, record in enumerate(records, start=1):
        observer.observe(record)
        if n in checkpoints:
            phase = 1 if n >= switch else 0
            truth = opponent.policies[phase]
            oracle = oracle_for(world, name, phase, settings.budget)
            snap = adapter.snapshot(observer.retained(), public_model=observer.public_model())
            row = evaluate_snapshot(world, snap, truth, oracle["cfr_value"], extended=False)
            row["retained_hands"] = len(observer.retained())
            row["oracle_budget_gain"] = oracle["oracle_budget_gain"]
            out["checkpoints"][str(n - switch)] = row
    out["resets"] = observer.resets
    return out


# -- tuning -------------------------------------------------------------------------------------


def tune(results: Sequence[Mapping[str, Any]], config: AdaptiveStudyConfig) -> dict[str, Any]:
    rows = [(r, row) for r in results for row in r["rows"] if row["hands"] > 0]

    def mean_gain(fn: Any) -> float:
        return statistics.fmean(fn(row) for _, row in rows)

    def linear(alpha: float, row: Mapping[str, Any]) -> float:
        return alpha * float(row["br_map_gain"])

    fixed = {
        a: mean_gain(lambda row, a=a: linear(min(a, row["budget_max_alpha"]), row))
        for a in config.fixed_alpha_grid
    }
    evidence = {
        k: mean_gain(
            lambda row, k=k: linear(
                min(row["hands"] / (row["hands"] + k), row["budget_max_alpha"]), row
            )
        )
        for k in config.evidence_k_grid
    }

    def conf_name(k: float) -> str:
        return (
            "confidence_budget"
            if k == config.settings.confidence_k[0]
            else f"confidence_budget_k{k:g}"
        )

    confidence = {
        k: mean_gain(lambda row, k=k: row["gate"]["true_gain"][conf_name(k)])
        for k in config.settings.confidence_k
    }
    best_k = max(confidence, key=lambda k: confidence[k])

    def gated(row: Mapping[str, Any], z: float) -> float:
        cands = {}
        for label, cname in (("budget_max", "budget_max"), ("confidence", conf_name(best_k))):
            c = row["gate"]["candidates"][cname]
            cands[label] = (c["mean"] - z * c["sd"], row["gate"]["true_gain"][cname])
        lcb, gain = max(cands.values(), key=lambda v: v[0])
        return gain if lcb > 0 else 0.0

    gate = {z: mean_gain(lambda row, z=z: gated(row, z)) for z in config.gate_z_grid}
    chosen = {
        "fixed_alpha": max(fixed, key=lambda a: fixed[a]),
        "evidence_k": max(evidence, key=lambda k: evidence[k]),
        "confidence_k": best_k,
        "gate_z": max(gate, key=lambda z: gate[z]),
    }
    return {
        "objective": "mean exact payoff gain over CFR on development opponents, "
        "hands 10..500, exploitability budget enforced",
        "fixed_alpha": {str(k): v for k, v in fixed.items()},
        "evidence_k": {str(k): v for k, v in evidence.items()},
        "confidence_k": {str(k): v for k, v in confidence.items()},
        "gate_z": {str(k): v for k, v in gate.items()},
        "chosen": chosen,
    }


# -- aggregation -------------------------------------------------------------------------------


def summarize_snapshots(
    results: Sequence[Mapping[str, Any]], budgets: Sequence[int]
) -> dict[str, Any]:
    by_opp: dict[str, list[Mapping[str, Any]]] = {}
    for r in results:
        by_opp.setdefault(r["opponent"], []).append(r)

    def cluster(fn: Any, b: int) -> dict[str, float]:
        values = {
            name: [fn(row) for r in rs for row in r["rows"] if row["hands"] == b]
            for name, rs in by_opp.items()
        }
        return clustered(values)

    out: dict[str, Any] = {"methods": {}, "paired": {}, "oracle": {}, "harm_rate": {}}
    for m in METHODS:
        out["methods"][m] = {
            str(b): {
                "gain": cluster(lambda row, m=m: row["methods"][m]["gain"], b),
                "exploitability": cluster(lambda row, m=m: row["methods"][m]["exploitability"], b),
                "max_exploitability": max(
                    row["methods"][m]["exploitability"]
                    for r in results
                    for row in r["rows"]
                    if row["hands"] == b
                ),
            }
            for b in budgets
        }
        out["harm_rate"][m] = {
            str(b): float(
                np.mean(
                    [
                        row["methods"][m]["gain"] < -1e-9
                        for r in results
                        for row in r["rows"]
                        if row["hands"] == b
                    ]
                )
            )
            for b in budgets
        }
    for a, b_ in (
        ("gated", "budget_max"),
        ("budget_max", "confidence_budget"),
        ("budget_max", "fixed_mix"),
        ("budget_max", "evidence_mix"),
    ):
        out["paired"][f"{a}-{b_}"] = {
            str(b): cluster(
                lambda row, a=a, b_=b_: row["methods"][a]["gain"] - row["methods"][b_]["gain"], b
            )
            for b in budgets
        }
    for key in (
        "oracle_br_gain",
        "oracle_br_exploitability",
        "oracle_budget_gain",
        "oracle_budget_alpha",
    ):
        out["oracle"][key] = clustered({n: [rs[0]["oracle"][key]] for n, rs in by_opp.items()})
    fams: dict[str, dict[str, Any]] = {}
    for r in results:
        fams.setdefault(r["family"], {}).setdefault(r["opponent"], []).append(r)
    out["by_family"] = {
        fam: {
            m: {
                str(b): clustered(
                    {
                        n: [
                            row["methods"][m]["gain"]
                            for r in rs
                            for row in r["rows"]
                            if row["hands"] == b
                        ]
                        for n, rs in opps.items()
                    }
                )["mean"]
                for b in budgets
            }
            for m in ("budget_max", "gated", "br_map", "br_naive")
        }
        | {
            "oracle_budget_gain": statistics.fmean(
                rs[0]["oracle"]["oracle_budget_gain"] for rs in opps.values()
            )
        }
        | {
            "oracle_br_gain": statistics.fmean(
                rs[0]["oracle"]["oracle_br_gain"] for rs in opps.values()
            )
        }
        for fam, opps in fams.items()
    }
    out["frontier"] = {
        resp: {
            str(b): {
                delta: {
                    "gain": cluster(
                        lambda row, resp=resp, d=delta: row["frontier"][resp][d]["gain"], b
                    )["mean"],
                    "alpha": cluster(
                        lambda row, resp=resp, d=delta: row["frontier"][resp][d]["alpha"], b
                    )["mean"],
                    "exploitability_max": max(
                        row["frontier"][resp][delta]["exploitability"]
                        for r in results
                        for row in r["rows"]
                        if row["hands"] == b
                    ),
                }
                for delta in results[0]["rows"][0]["frontier"][resp]
            }
            for b in budgets
        }
        for resp in ("br_map", "br_confidence")
    }
    out["reconstruction"] = {
        k: {str(b): cluster(lambda row, k=k: row["reconstruction"][k], b)["mean"] for b in budgets}
        for k in ("estimate", "population_prior", "cfr_prior")
    }
    out["public_excess_log_loss_em"] = {
        str(b): cluster(lambda row: row["public_excess_log_loss_em"], b)["mean"] for b in budgets
    }
    gate_rows = [row for r in results for row in r["rows"]]
    out["gate"] = {}
    for b in budgets:
        rs = [row for row in gate_rows if row["hands"] == b]
        opened = [row for row in rs if row["gate"]["open"]]
        chosen_true = [row["gate"]["true_gain"][row["gate"]["chosen"]] for row in rs]
        chosen_lcb = [row["gate"]["candidates"][row["gate"]["chosen"]]["lcb"] for row in rs]
        out["gate"][str(b)] = {
            "open_rate": len(opened) / len(rs),
            "open_with_negative_true_gain": (
                sum(row["gate"]["true_gain"][row["gate"]["chosen"]] < 0 for row in opened)
                / len(opened)
                if opened
                else None
            ),
            "true_gain_at_or_above_lcb": float(
                np.mean([t >= lcb for t, lcb in zip(chosen_true, chosen_lcb, strict=True)])
            ),
            "mean_posterior_sd": float(
                np.mean([row["gate"]["candidates"][row["gate"]["chosen"]]["sd"] for row in rs])
            ),
        }
    out["prior_selection"] = {
        str(b): {
            k: float(
                np.mean(
                    [
                        f"{row['prior'][0]}:{row['prior'][1]:g}" == k
                        for r in results
                        for row in r["rows"]
                        if row["hands"] == b
                    ]
                )
            )
            for k in sorted(
                {f"{row['prior'][0]}:{row['prior'][1]:g}" for r in results for row in r["rows"]}
            )
        }
        for b in budgets
    }
    out["timings"] = {
        str(b): {
            k: statistics.fmean(
                row["timings"][k] for r in results for row in r["rows"] if row["hands"] == b
            )
            for k in results[0]["rows"][0]["timings"]
        }
        for b in budgets
    }
    return out


def summarize_change(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group_name, bait in (("switching", False), ("bait_switch", True)):
        sel = [r for r in results if ("bait_switch" in r["opponent"]) == bait]
        group: dict[str, Any] = {}
        for detect in (False, True):
            label = "detection" if detect else "cumulative"
            rs = [r for r in sel if r["detect"] == detect]
            offsets = sorted({int(k) for r in rs for k in r["checkpoints"]})
            group[label] = {
                m: {
                    str(o): clustered(
                        {
                            name: [
                                r["checkpoints"][str(o)]["methods"][m]["gain"]
                                for r in rs
                                if r["opponent"] == name and str(o) in r["checkpoints"]
                            ]
                            for name in {r["opponent"] for r in rs}
                            if any(str(o) in r["checkpoints"] for r in rs if r["opponent"] == name)
                        }
                    )
                    for o in offsets
                }
                for m in ("budget_max", "gated")
            }
            group[label]["oracle_budget_gain"] = {
                str(o): statistics.fmean(
                    r["checkpoints"][str(o)]["oracle_budget_gain"]
                    for r in rs
                    if str(o) in r["checkpoints"]
                )
                for o in offsets
            }
            group[label]["resets_per_run"] = statistics.fmean(len(r["resets"]) for r in rs)
            if detect:
                delays = []
                for r in rs:
                    switch = int(r["opponent"].rsplit("@", 1)[1])
                    post = [h for h in r["resets"] if h >= switch]
                    if post:
                        delays.append(post[0] - switch + 1)
                group[label]["detection_delay"] = mean_ci(delays) if delays else None
                group[label]["detected_runs"] = len(delays)
                group[label]["runs"] = len(rs)
                group[label]["max_exploitability"] = max(
                    cp["methods"][m]["exploitability"]
                    for r in rs
                    for cp in r["checkpoints"].values()
                    for m in ("budget_max", "gated")
                )
        out[group_name] = group
    return out


def measure_runtime(world: World, settings: AdaptationSettings) -> dict[str, Any]:
    opponent = world.opponents[world.groups["heldout"][0]]
    records = observe(world, opponent, 0, 200)
    adapter = Adapter(world.game, world.specs, world.priors, settings)
    start = time.perf_counter()
    snap = adapter.snapshot(records)
    elapsed = time.perf_counter() - start
    tracemalloc.start()  # separate pass: tracing slows Python several-fold
    adapter.snapshot(records)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    agent = GameStrategyAgent(snap.policies["gated"])
    state = world.game.state_from_moves((5, 0, 2))
    obs, legal = state.observation(1), state.legal_actions()
    ns = {"agent": agent, "obs": obs, "legal": legal, "rng": np.random.default_rng(0)}
    empty = time_statement("empty", "pass", "pass", ns, number=20_000, repeats=9)
    act = time_statement(
        "adaptive_decision", "GameStrategyAgent.act on the adaptive policy",
        "_ = agent.act(obs, legal, rng)", ns, number=20_000, repeats=9,
        overhead_ns=empty.median_ns,
    )  # fmt: skip
    return {
        "snapshot_seconds_200_hands": elapsed,
        "snapshot_timings": snap.timings,
        "snapshot_tracemalloc_peak_bytes": peak,
        "decision_latency": act.to_dict(),
        "policy_entries": sum(len(v) for v in snap.policies["gated"].values()),
    }


def _pool(config: AdaptiveStudyConfig) -> ProcessPoolExecutor:
    workers = config.workers or (os.cpu_count() or 1)
    return ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            config.hero_path,
            config.per_family,
            config.out_of_population_per_family,
            config.switching,
        ),
    )


def run_adaptive_study(config: AdaptiveStudyConfig, output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    world = build_world(
        config.hero_path, config.per_family, config.out_of_population_per_family, config.switching
    )
    budgets = config.budgets
    with _pool(config) as pool:
        dev_tasks = [
            (n, s, config.settings, budgets, True)
            for n in world.groups["development"]
            for s in range(config.development_seeds)
        ]
        dev = list(pool.map(snapshot_task, dev_tasks))
        tuning = tune(dev, config)
        c = tuning["chosen"]
        frozen = replace(
            config.settings,
            confidence_k=(c["confidence_k"],),
            fixed_alpha=c["fixed_alpha"],
            evidence_k=c["evidence_k"],
            gate_z=c["gate_z"],
        )
        held_tasks = [
            (n, s, frozen, budgets, False)
            for group in ("heldout", "out_of_population")
            for n in world.groups[group]
            for s in range(config.seeds)
        ]
        held = list(pool.map(snapshot_task, held_tasks))
        change_tasks = [
            (n, s, frozen, d)
            for group in ("heldout_switching", "bait_switch")
            for n in world.groups[group]
            for s in range(config.change_seeds)
            for d in (False, True)
        ]
        change = list(pool.map(change_task, change_tasks))
    heldout = [r for r in held if r["opponent"] in set(world.groups["heldout"])]
    oop = [r for r in held if r["opponent"] in set(world.groups["out_of_population"])]
    document: dict[str, Any] = {
        "experiment": "m6-adaptive-strategy",
        "config": {
            "hero_path": str(config.hero_path),
            "hero_digest": game_strategy_digest(world.cfr, world.game.action_codes),
            "cfr_exploitability": world.priors.cfr_exploitability,
            "budgets": list(budgets),
            "development_seeds": config.development_seeds,
            "seeds": config.seeds,
            "change_seeds": config.change_seeds,
            "development_settings": asdict(config.settings),
            "frozen_settings": asdict(frozen),
        },
        "metadata": collect_metadata(),
        "groups": world.groups,
        "tuning": tuning,
        "development": summarize_snapshots(dev, budgets),
        "heldout": summarize_snapshots(heldout, budgets),
        "out_of_population": summarize_snapshots(oop, budgets),
        "change": summarize_change(change),
        "runtime": measure_runtime(world, frozen),
        "raw": {"heldout": heldout, "out_of_population": oop, "change": change},
        "elapsed_seconds": time.perf_counter() - start,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(document, indent=1, default=_json) + "\n")
    document["plot"] = plot_study(document, output_dir / "adaptive.png")
    return document


def _json(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def plot_study(document: Mapping[str, Any], path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    budgets = document["config"]["budgets"]
    xs = [max(b, 1) for b in budgets]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    held = document["heldout"]
    for m in ("budget_max", "gated", "confidence_budget", "evidence_mix"):
        g = held["methods"][m]
        axes[0].plot(xs, [g[str(b)]["gain"]["mean"] for b in budgets], marker="o", label=m)
        axes[0].fill_between(
            xs,
            [g[str(b)]["gain"]["ci_low"] for b in budgets],
            [g[str(b)]["gain"]["ci_high"] for b in budgets],
            alpha=0.15,
        )
    axes[0].axhline(
        held["oracle"]["oracle_budget_gain"]["mean"],
        ls="--",
        color="k",
        lw=1,
        label="oracle, same budget",
    )
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_title("Held-out: payoff gain over CFR (chips/hand), budget +0.1")
    axes[0].legend(fontsize=8)
    front = held["frontier"]["br_map"]
    for b in (20, 100, 500):
        if str(b) not in front:
            continue
        pts = sorted(front[str(b)].items(), key=lambda kv: float(kv[0]))
        axes[1].plot(
            [float(d) for d, _ in pts], [v["gain"] for _, v in pts], marker="o", label=f"{b} hands"
        )
    axes[1].set_xlabel("exploitability budget above CFR")
    axes[1].set_title("Gain vs exploitability budget (held-out)")
    axes[1].legend(fontsize=8)
    sw = document["change"]["switching"]
    for label in ("cumulative", "detection"):
        g = sw[label]["budget_max"]
        ks = sorted(int(k) for k in g)
        axes[2].plot(ks, [g[str(k)]["mean"] for k in ks], marker="o", label=f"budget_max, {label}")
    ks = sorted(int(k) for k in sw["detection"]["oracle_budget_gain"])
    axes[2].plot(
        ks,
        [sw["detection"]["oracle_budget_gain"][str(k)] for k in ks],
        "k--",
        label="oracle, same budget",
    )
    axes[2].axhline(0, color="grey", lw=0.8)
    axes[2].set_title("Switching opponents: gain after the switch")
    axes[2].set_xlabel("hands since switch")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


__all__ = ["AdaptiveStudyConfig", "run_adaptive_study"]
