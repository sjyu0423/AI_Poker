"""M7A: RNR and DBR versus M6's CFR/best-response mixture at equal exploitability.

For every opponent, seed and observation budget, the **same** M6
reconstruction (EM estimate and expected counts from M6's adapter, on M6's
observation streams) is turned into three families of hero strategies:

* ``mix``: M6's realization mixture ``(1 − α)·CFR + α·BR(estimate)``;
* ``rnr``: Restricted Nash Responses to the estimate, on a grid of ``p``;
* ``dbr``: Data-Biased Responses (frequency model, count-based confidence),
  on a grid of ``p_max``.

For each exploitability budget ``CFR + δ``, each family plays its member with
the highest value **against the estimate** among those whose **exact**
exploitability is within budget. The true opponent is never used to choose.
Payoffs are then evaluated exactly against the true policy. M6's
``budget_max`` and ``gated`` policies (δ = 0.1) are recomputed and checked
against M6's recorded results.

DBR's count threshold is tuned on development opponents; everything else is
fixed in advance. Tasks run in parallel processes and are deterministic.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from pokermind.adaptive.adapter import AdaptationSettings, Adapter, ChangeAwareObserver
from pokermind.adaptive.response import (
    Mixer,
    exploitability,
    hero_best_response,
    hero_value,
)
from pokermind.experiments.adaptive_study import World, build_world, observe
from pokermind.experiments.opponent_study import clustered
from pokermind.robust.responses import dbr_model, dbr_strategy, rnr_strategy
from pokermind.robust.tree import CompiledTree
from pokermind.simulation import collect_metadata
from pokermind.strategy import GameStrategy, game_strategy_digest

HANDS: Final = (0, 20, 50, 100, 200, 500)
DELTAS: Final = (0.0, 0.02, 0.05, 0.1, 0.25)
P_GRID: Final = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
P_MAX_GRID: Final = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
DBR_THRESHOLDS: Final = (3.0, 10.0, 30.0)
ITERATIONS: Final = 500
M6_SETTINGS: Final = AdaptationSettings(
    frontier_budgets=(), confidence_k=(3.0,), fixed_alpha=0.25, evidence_k=10.0, gate_z=0.5
)
"""M6's frozen settings (its tuned values); its frontier is recomputed here."""
FAMILIES: Final = ("mix", "rnr", "dbr")


@dataclass(frozen=True, slots=True)
class RobustStudyConfig:
    hero_path: Path
    m6_metrics: Path | None = None
    development_seeds: int = 2
    seeds: int = 3
    change_seeds: int = 3
    hands: tuple[int, ...] = HANDS
    deltas: tuple[float, ...] = DELTAS
    p_grid: tuple[float, ...] = P_GRID
    p_max_grid: tuple[float, ...] = P_MAX_GRID
    dbr_thresholds: tuple[float, ...] = DBR_THRESHOLDS
    iterations: int = ITERATIONS
    workers: int = 0

    def __post_init__(self) -> None:
        if min(self.seeds, self.development_seeds, self.change_seeds) <= 0:
            raise ValueError("seed counts must be positive")
        if any(d < 0 for d in self.deltas) or not self.p_grid or not self.p_max_grid:
            raise ValueError("invalid grids")


_STATE: dict[str, Any] = {}


def _init(hero_path: Path) -> None:
    world = build_world(hero_path, 4, 2, 8)
    _STATE["world"] = world
    _STATE["tree"] = CompiledTree(world.game)


def _world() -> tuple[World, CompiledTree]:
    return _STATE["world"], _STATE["tree"]


@dataclass(slots=True)
class Candidate:
    family: str
    parameter: float
    policy: GameStrategy
    exploitability: float
    model_value: float


def _grid(
    world: World,
    family: str,
    params: Sequence[float],
    build: Any,
    model: GameStrategy,
    stop_above: float,
) -> tuple[list[Candidate], float]:
    """Exact exploitability and model value of each grid member, ascending; stops
    after two consecutive members above ``stop_above``."""
    g, specs = world.game, world.specs
    out: list[Candidate] = []
    over = 0
    seconds = 0.0
    for p in params:
        t0 = time.perf_counter()
        policy = build(p)
        seconds += time.perf_counter() - t0
        e = exploitability(g, policy, specs)
        out.append(Candidate(family, p, policy, e, hero_value(g, policy, model, specs)))
        over = over + 1 if e > stop_above else 0
        if over >= 2:
            break
    return out, seconds


def _select(cands: Sequence[Candidate], budget: float) -> Candidate | None:
    feasible = [c for c in cands if c.exploitability <= budget + 1e-12]
    if not feasible:
        return None
    return max(feasible, key=lambda c: (c.model_value, -c.parameter))


def _families(
    world: World,
    tree: CompiledTree,
    estimate: GameStrategy,
    counts: Any,
    mixer: Mixer,
    config: RobustStudyConfig,
    thresholds: Sequence[float],
) -> tuple[dict[str, list[Candidate]], dict[str, float]]:
    g, specs, cfr_e = world.game, world.specs, world.priors.cfr_exploitability
    stop = cfr_e + max(config.deltas)
    timings: dict[str, float] = {}
    families: dict[str, list[Candidate]] = {}
    t0 = time.perf_counter()
    mix: list[Candidate] = []
    for d in config.deltas:
        s = mixer.largest_alpha(cfr_e + d)
        policy = mixer.policy(s.alpha)
        mix.append(
            Candidate(
                "mix", s.alpha, policy, s.exploitability, hero_value(g, policy, estimate, specs)
            )
        )
    families["mix"] = mix
    timings["mix"] = time.perf_counter() - t0
    families["rnr"], timings["rnr_solves"] = _grid(
        world,
        "rnr",
        config.p_grid,
        lambda p: rnr_strategy(tree, estimate, p, config.iterations),
        estimate,
        stop,
    )
    acts = {k: s.actions for k, s in specs.items()}
    model = dbr_model(counts, acts)
    for th in thresholds:
        name = "dbr" if th == thresholds[0] else f"dbr_t{th:g}"
        families[name], timings[f"{name}_solves"] = _grid(
            world, name, config.p_max_grid,
            lambda pm, th=th: dbr_strategy(tree, model, pm, th, config.iterations), estimate, stop,
        )  # fmt: skip
    return families, timings


def _evaluate(
    world: World,
    families: Mapping[str, list[Candidate]],
    truth: GameStrategy,
    cfr_value: float,
    deltas: Sequence[float],
) -> dict[str, Any]:
    g, specs, cfr_e = world.game, world.specs, world.priors.cfr_exploitability
    cache: dict[int, float] = {}

    def gain(c: Candidate) -> float:
        if id(c) not in cache:
            cache[id(c)] = hero_value(g, c.policy, truth, specs) - cfr_value
        return cache[id(c)]

    out: dict[str, Any] = {}
    for name, cands in families.items():
        per: dict[str, Any] = {}
        for d in deltas:
            if name == "mix":
                chosen: Candidate | None = cands[list(deltas).index(d)]
            else:
                chosen = _select(cands, cfr_e + d)
            if chosen is None:  # nothing within budget: keep CFR
                per[str(d)] = {
                    "gain": 0.0,
                    "exploitability": cfr_e,
                    "parameter": None,
                    "fallback": True,
                }
            else:
                per[str(d)] = {
                    "gain": gain(chosen),
                    "exploitability": chosen.exploitability,
                    "parameter": chosen.parameter,
                    "fallback": False,
                }
        out[name] = per
    out["frontier"] = {
        name: [
            {
                "parameter": c.parameter,
                "exploitability": c.exploitability,
                "model_value": c.model_value,
                "gain": gain(c),
            }
            for c in cands
        ]
        for name, cands in families.items()
        if name != "mix"
    }
    return out


def _oracle(
    world: World, tree: CompiledTree, name: str, phase: int, config: RobustStudyConfig
) -> dict[str, Any]:
    key = f"oracle:{name}:{phase}"
    if key in _STATE:
        return dict(_STATE[key])
    g, specs, cfr = world.game, world.specs, world.cfr
    truth = world.opponents[name].policies[phase]
    cfr_value = hero_value(g, cfr, truth, specs)
    br = hero_best_response(g, truth, specs)
    mixer = Mixer(g, specs, cfr, br, world.priors.cfr_exploitability)

    def oracle_rnr(p: float) -> GameStrategy:
        return rnr_strategy(tree, truth, p, config.iterations)

    stop = world.priors.cfr_exploitability + max(config.deltas)
    rnr, _ = _grid(world, "rnr", config.p_grid, oracle_rnr, truth, stop)
    res: dict[str, Any] = {
        "cfr_value": cfr_value,
        "br_gain": hero_value(g, br, truth, specs) - cfr_value,
    }
    for d in config.deltas:
        s = mixer.largest_alpha(world.priors.cfr_exploitability + d)
        res[f"mix:{d}"] = hero_value(g, mixer.policy(s.alpha), truth, specs) - cfr_value
        c = _select(rnr, world.priors.cfr_exploitability + d)
        res[f"rnr:{d}"] = 0.0 if c is None else c.model_value - cfr_value
    _STATE[key] = res
    return dict(res)


def snapshot_task(args: tuple[str, int, RobustStudyConfig, tuple[float, ...]]) -> dict[str, Any]:
    name, seed, config, thresholds = args
    world, tree = _world()
    opponent = world.opponents[name]
    truth = opponent.policies[0]
    oracle = _oracle(world, tree, name, 0, config)
    adapter = Adapter(world.game, world.specs, world.priors, M6_SETTINGS)
    records = observe(world, opponent, seed, max(config.hands))
    rows = []
    for n in config.hands:
        t0 = time.perf_counter()
        snap = adapter.snapshot(records[:n])
        m6_seconds = time.perf_counter() - t0
        g, specs = world.game, world.specs
        m6 = {
            m: hero_value(g, snap.policies[m], truth, specs) - oracle["cfr_value"]
            for m in ("budget_max", "gated", "br_map")
        }
        families, timings = _families(
            world,
            tree,
            snap.estimate,
            snap.expected_counts,
            snap.responses["br_map"].mixer,
            config,
            thresholds,
        )
        timings["m6_snapshot"] = m6_seconds
        row = {"hands": n, "m6": m6, "timings": timings, "prior": list(snap.prior)}
        row.update(_evaluate(world, families, truth, oracle["cfr_value"], config.deltas))
        rows.append(row)
    return {
        "opponent": name,
        "family": opponent.family,
        "seed": seed,
        "oracle": oracle,
        "rows": rows,
    }


def change_task(args: tuple[str, int, RobustStudyConfig, float, bool]) -> dict[str, Any]:
    name, seed, config, threshold, detect = args
    world, tree = _world()
    opponent = world.opponents[name]
    assert opponent.switch_hand is not None
    switch = opponent.switch_hand
    offsets = (0, 20, 50, 100, 200)
    checkpoints = sorted({switch + o for o in offsets})
    records = observe(world, opponent, seed, max(checkpoints))
    observer = ChangeAwareObserver(world.priors.public_prior, detect=detect)
    adapter = Adapter(world.game, world.specs, world.priors, M6_SETTINGS)
    small = RobustStudyConfig(
        config.hero_path,
        deltas=(0.1,),
        p_grid=config.p_grid,
        p_max_grid=config.p_max_grid,
        iterations=config.iterations,
    )
    out: dict[str, Any] = {"opponent": name, "seed": seed, "detect": detect, "checkpoints": {}}
    for n, record in enumerate(records, start=1):
        observer.observe(record)
        if n in checkpoints:
            phase = 1 if n >= switch else 0
            truth = opponent.policies[phase]
            cfr_value = hero_value(world.game, world.cfr, truth, world.specs)
            snap = adapter.snapshot(observer.retained(), public_model=observer.public_model())
            families, _ = _families(
                world,
                tree,
                snap.estimate,
                snap.expected_counts,
                snap.responses["br_map"].mixer,
                small,
                (threshold,),
            )
            ev = _evaluate(world, families, truth, cfr_value, (0.1,))
            out["checkpoints"][str(n - switch)] = {f: ev[f]["0.1"] for f in FAMILIES} | {
                "gated": hero_value(world.game, snap.policies["gated"], truth, world.specs)
                - cfr_value
            }
    out["resets"] = observer.resets
    return out


# -- aggregation ------------------------------------------------------------------------------


def _by_opp(results: Sequence[Mapping[str, Any]], fn: Any, n: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in results:
        rows = {row["hands"]: row for row in r["rows"]}
        if n in rows:
            out.setdefault(r["opponent"], []).append(fn(rows, n))
    return out


def summarize(
    results: Sequence[Mapping[str, Any]], config: RobustStudyConfig, families: Sequence[str]
) -> dict[str, Any]:
    hands, deltas = config.hands, [str(d) for d in config.deltas]
    out: dict[str, Any] = {
        "gain": {},
        "personal_increment": {},
        "paired": {},
        "compliance": {},
        "harm": {},
        "worst": {},
    }
    for f in families:
        out["gain"][f] = {
            d: {
                str(n): clustered(
                    _by_opp(results, lambda rows, n, f=f, d=d: rows[n][f][d]["gain"], n)
                )
                for n in hands
            }
            for d in deltas
        }
        out["personal_increment"][f] = {
            d: {
                str(n): clustered(
                    _by_opp(
                        results,
                        lambda rows, n, f=f, d=d: rows[n][f][d]["gain"] - rows[0][f][d]["gain"],
                        n,
                    )
                )
                for n in hands
                if n > 0
            }
            for d in deltas
        }
        all_rows = [(r, row) for r in results for row in r["rows"]]
        out["compliance"][f] = {
            d: {
                "max_exploitability": max(row[f][d]["exploitability"] for _, row in all_rows),
                "fallback_rate": float(np.mean([row[f][d]["fallback"] for _, row in all_rows])),
            }
            for d in deltas
        }
        out["harm"][f] = {
            d: float(np.mean([row[f][d]["gain"] < -1e-9 for _, row in all_rows])) for d in deltas
        }
        out["worst"][f] = {
            d: sorted(
                (
                    {"gain": row[f][d]["gain"], "opponent": r["opponent"], "hands": row["hands"]}
                    for r, row in all_rows
                ),
                key=lambda x: x["gain"],
            )[:3]
            for d in deltas
        }
    for a, b in (("rnr", "mix"), ("dbr", "mix"), ("rnr", "dbr")):
        if a in families and b in families:
            out["paired"][f"{a}-{b}"] = {
                d: {
                    str(n): clustered(
                        _by_opp(
                            results,
                            lambda rows, n, a=a, b=b, d=d: (
                                rows[n][a][d]["gain"] - rows[n][b][d]["gain"]
                            ),
                            n,
                        )
                    )
                    for n in hands
                }
                for d in deltas
            }
    out["m6"] = {
        m: {
            str(n): clustered(_by_opp(results, lambda rows, n, m=m: rows[n]["m6"][m], n))
            for n in hands
        }
        for m in ("budget_max", "gated", "br_map")
    }
    by_opp = {r["opponent"]: r for r in results}
    out["oracle"] = {
        k: clustered({n: [r["oracle"][k]] for n, r in by_opp.items()})
        for k in by_opp[next(iter(by_opp))]["oracle"]
        if k != "cfr_value"
    }
    fams: dict[str, list[Mapping[str, Any]]] = {}
    for r in results:
        fams.setdefault(r["family"], []).append(r)
    out["by_family"] = {
        fam: {
            f: {
                d: {
                    str(n): statistics.fmean(
                        row[f][d]["gain"] for r in rs for row in r["rows"] if row["hands"] == n
                    )
                    for n in (0, 100, 500)
                    if n in hands
                }
                for d in ("0.1", "0.25")
                if d in deltas
            }
            for f in families
        }
        | {
            "oracle_rnr": {
                d: statistics.fmean(r["oracle"][f"rnr:{d}"] for r in rs)
                for d in ("0.1", "0.25")
                if d in deltas
            }
        }
        for fam, rs in fams.items()
    }
    out["selected_parameter"] = {
        f: {
            d: {
                str(n): statistics.fmean(
                    row[f][d]["parameter"] or 0.0
                    for r in results
                    for row in r["rows"]
                    if row["hands"] == n
                )
                for n in hands
            }
            for d in deltas
        }
        for f in families
    }
    timing_keys = results[0]["rows"][0]["timings"].keys()
    out["timings"] = {
        k: statistics.fmean(row["timings"][k] for r in results for row in r["rows"])
        for k in timing_keys
    }
    return out


def tune_dbr(results: Sequence[Mapping[str, Any]], config: RobustStudyConfig) -> dict[str, Any]:
    names = {
        th: ("dbr" if i == 0 else f"dbr_t{th:g}") for i, th in enumerate(config.dbr_thresholds)
    }
    scores = {
        th: statistics.fmean(
            row[name][str(d)]["gain"]
            for r in results
            for row in r["rows"]
            if row["hands"] > 0
            for d in config.deltas
        )
        for th, name in names.items()
    }
    return {
        "objective": "mean exact gain over CFR on development opponents, all budgets, hands > 0",
        "scores": {str(k): v for k, v in scores.items()},
        "chosen": max(scores, key=lambda k: scores[k]),
    }


def check_m6(results: Sequence[Mapping[str, Any]], m6_metrics: Path | None) -> dict[str, Any]:
    """Compare recomputed M6 policies with M6's recorded held-out results."""
    if m6_metrics is None or not m6_metrics.is_file():
        return {"available": False}
    m6 = json.loads(m6_metrics.read_text())
    recorded = {
        (r["opponent"], r["seed"], row["hands"]): row["methods"]
        for r in m6["raw"]["heldout"] + m6["raw"]["out_of_population"]
        for row in r["rows"]
    }
    diffs: dict[str, float] = {"budget_max": 0.0, "gated": 0.0, "br_map": 0.0}
    compared = 0
    for r in results:
        for row in r["rows"]:
            key = (r["opponent"], r["seed"], row["hands"])
            if key not in recorded:
                continue
            compared += 1
            for m in diffs:
                diffs[m] = max(diffs[m], abs(row["m6"][m] - recorded[key][m]["gain"]))
    return {"available": True, "rows_compared": compared, "max_abs_gain_difference": diffs}


def summarize_change(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group, bait in (("switching", False), ("bait_switch", True)):
        sel = [r for r in results if ("bait_switch" in r["opponent"]) == bait]
        g: dict[str, Any] = {}
        for detect in (False, True):
            rs = [r for r in sel if r["detect"] == detect]
            offsets = sorted({int(k) for r in rs for k in r["checkpoints"]})
            g["detection" if detect else "cumulative"] = {
                m: {
                    str(o): clustered(
                        {
                            name: [
                                (
                                    r["checkpoints"][str(o)][m]["gain"]
                                    if m != "gated"
                                    else r["checkpoints"][str(o)][m]
                                )
                                for r in rs
                                if r["opponent"] == name
                            ]
                            for name in {r["opponent"] for r in rs}
                        }
                    )
                    for o in offsets
                }
                for m in (*FAMILIES, "gated")
            } | {
                "max_exploitability": max(
                    r["checkpoints"][str(o)][m]["exploitability"]
                    for r in rs
                    for o in offsets
                    for m in FAMILIES
                )
            }
        out[group] = g
    return out


def measure_runtime(config: RobustStudyConfig) -> dict[str, Any]:
    _init(config.hero_path)
    world, tree = _world()
    opponent = world.opponents[world.groups["heldout"][0]]
    records = observe(world, opponent, 0, 200)
    snap = Adapter(world.game, world.specs, world.priors, M6_SETTINGS).snapshot(records)
    t0 = time.perf_counter()
    rnr_strategy(tree, snap.estimate, 0.3, config.iterations)
    rnr_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    exploitability(world.game, snap.estimate, world.specs)
    expl_seconds = time.perf_counter() - t0
    tracemalloc.start()
    rnr_strategy(tree, snap.estimate, 0.3, config.iterations)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t0 = time.perf_counter()
    CompiledTree(world.game)
    compile_seconds = time.perf_counter() - t0
    return {
        "rnr_solve_seconds": rnr_seconds,
        "rnr_iterations": config.iterations,
        "exact_exploitability_seconds": expl_seconds,
        "rnr_solve_tracemalloc_peak_bytes": peak,
        "tree_compile_seconds": compile_seconds,
        "tree": {
            "nodes": tree.num_nodes,
            "slots": tree.num_slots,
            "information_sets": tree.num_isets,
        },
    }


def run_robust_study(config: RobustStudyConfig, output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    world = build_world(config.hero_path, 4, 2, 8)
    workers = config.workers or (os.cpu_count() or 1)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init, initargs=(config.hero_path,)
    ) as pool:
        dev = list(
            pool.map(
                snapshot_task,
                [
                    (n, s, config, config.dbr_thresholds)
                    for n in world.groups["development"]
                    for s in range(config.development_seeds)
                ],
            )
        )
        tuning = tune_dbr(dev, config)
        th = (tuning["chosen"],)
        held = list(
            pool.map(
                snapshot_task,
                [
                    (n, s, config, th)
                    for grp in ("heldout", "out_of_population")
                    for n in world.groups[grp]
                    for s in range(config.seeds)
                ],
            )
        )
        change = list(
            pool.map(
                change_task,
                [
                    (n, s, config, th[0], d)
                    for grp in ("heldout_switching", "bait_switch")
                    for n in world.groups[grp]
                    for s in range(config.change_seeds)
                    for d in (False, True)
                ],
            )
        )
    heldout = [r for r in held if r["opponent"] in set(world.groups["heldout"])]
    oop = [r for r in held if r["opponent"] in set(world.groups["out_of_population"])]
    dev_families = [
        "mix",
        "rnr",
        *[("dbr" if i == 0 else f"dbr_t{t:g}") for i, t in enumerate(config.dbr_thresholds)],
    ]
    doc: dict[str, Any] = {
        "experiment": "m7a-robust-response",
        "config": {
            "hero_path": str(config.hero_path),
            "hero_digest": game_strategy_digest(world.cfr, world.game.action_codes),
            "cfr_exploitability": world.priors.cfr_exploitability,
            "hands": list(config.hands),
            "deltas": list(config.deltas),
            "p_grid": list(config.p_grid),
            "p_max_grid": list(config.p_max_grid),
            "dbr_thresholds": list(config.dbr_thresholds),
            "iterations": config.iterations,
            "seeds": config.seeds,
            "development_seeds": config.development_seeds,
            "change_seeds": config.change_seeds,
            "m6_settings": {
                "confidence_k": M6_SETTINGS.confidence_k,
                "gate_z": M6_SETTINGS.gate_z,
                "budget": M6_SETTINGS.budget,
            },
        },
        "metadata": collect_metadata(),
        "dbr_tuning": tuning,
        "m6_reproduction": check_m6(heldout + oop, config.m6_metrics),
        "development": summarize(dev, config, dev_families),
        "heldout": summarize(heldout, config, FAMILIES),
        "out_of_population": summarize(oop, config, FAMILIES),
        "change": summarize_change(change),
        "runtime": measure_runtime(config),
        "raw": {"heldout": heldout, "out_of_population": oop, "change": change},
        "elapsed_seconds": time.perf_counter() - start,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(doc, indent=1) + "\n")
    doc["plot"] = plot_study(doc, output_dir / "frontier.png")
    return doc


def plot_study(doc: Mapping[str, Any], path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    deltas = [str(d) for d in doc["config"]["deltas"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, group, n in ((axes[0], "heldout", "100"), (axes[1], "out_of_population", "100")):
        g = doc[group]["gain"]
        for f in FAMILIES:
            mean = [g[f][d][n]["mean"] for d in deltas]
            low = [m - g[f][d][n]["ci_low"] for m, d in zip(mean, deltas, strict=True)]
            high = [g[f][d][n]["ci_high"] - m for m, d in zip(mean, deltas, strict=True)]
            xs = [float(d) for d in deltas]
            ax.errorbar(xs, mean, yerr=[low, high], marker="o", capsize=3, label=f)
        orc = doc[group]["oracle"]
        ax.plot(
            [float(d) for d in deltas],
            [orc[f"rnr:{d}"]["mean"] for d in deltas],
            "k--",
            label="oracle RNR",
        )
        ax.set_xlabel("exploitability budget above CFR")
        ax.set_title(f"{group}: gain over CFR after {n} hands")
        ax.legend(fontsize=8)
    # Total gain against hands (not the per-family increment): DBR's 0-hand policy is
    # plain CFR, so its increment would count re-learning the population prior.
    hands = [h for h in doc["config"]["hands"] if h > 0]
    for split, marker, ls in (("heldout", "o", "-"), ("out_of_population", "x", ":")):
        gain = doc[split]["gain"]
        for f in FAMILIES:
            axes[2].plot(
                hands,
                [gain[f]["0.1"][str(h)]["mean"] for h in hands],
                marker=marker,
                ls=ls,
                label=f"{f}, {split}",
            )
        axes[2].axhline(gain["rnr"]["0.1"]["0"]["mean"], color="grey", lw=0.8, ls=ls)
    axes[2].set_xscale("log")
    axes[2].set_xticks(hands, [str(h) for h in hands])
    axes[2].minorticks_off()
    axes[2].set_xlabel("observed hands (grey: RNR with 0 hands)")
    axes[2].set_title("Gain over CFR at δ = 0.1")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
