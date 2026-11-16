"""M7B: reliability of the adaptive RNR baseline.

Four stages, each a ``pokermind reliability <stage>`` command writing one JSON
file under ``runs/m7b-adaptive-reliability/``:

1. ``convergence``: RNR at 500, 1,000 and 2,000 CFR+ iterations (one
   continued run), on M7A's held-out and out-of-population opponents
   (seed 0). For each checkpoint it records:
   * the exact RNR-game gap;
   * exact exploitability and payoffs;
   * strategy differences;
   * runtime;
   * whether M7A's RNR-vs-mixture conclusion changes.
2. ``calibrate``: development opponents only. Posterior samples (three
   temperatures) predict each candidate's gain over CFR. It measures the
   coverage of those intervals and of the opponent model's own intervals,
   then simulates the decision rules offline on the stored candidates. It
   proposes the frozen calibration and rule by rules fixed in this module
   (:func:`choose_calibration`, :func:`choose_rule`).
3. ``evaluate``: the frozen configuration
   (``configs/pokermind-research-v1.json``) on held-out, out-of-population,
   equilibrium, switching and a larger bait-and-switch population. M7A's
   point rule runs alongside, and its gains are checked against M7A's
   recorded rows.
4. ``benchmark``: one process, identical inputs. It compares M7A's
   selection procedure with the batched implementation (bit-identical
   strategies) and with warm-started and shortened variants.

The true opponent strategy is used to *evaluate* everywhere. It is used to
*choose hyperparameters* only on development opponents (stage 2), and never
as an input of an acting policy.
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

from pokermind.adaptive.adapter import Adapter, ChangeAwareObserver
from pokermind.adaptive.reconstruction import (
    CardConditionalModel,
    HandEvidence,
    ObserverHandRecord,
    Reconstruction,
    Theta,
    canonical_actions,
    canonical_infoset,
    evidence,
    expand,
    opponent_infoset_weights,
)
from pokermind.adaptive.response import Mixer, hero_best_response, hero_value
from pokermind.adaptive.robust_opponents import make_bait_switch_population
from pokermind.experiments.adaptive_study import World, build_world, observe
from pokermind.experiments.opponent_study import clustered, mean_ci, stream_id
from pokermind.experiments.robust_study import DELTAS, HANDS, M6_SETTINGS, _grid, _select
from pokermind.opponents.bayes import DetectorConfig
from pokermind.opponents.population import Opponent
from pokermind.research import ResearchConfig, load_research_config
from pokermind.robust.batch import BatchedRNR
from pokermind.robust.modified_game import rnr_bounds
from pokermind.robust.responses import rnr_strategy
from pokermind.robust.selection import (
    P_GRID,
    CandidateSet,
    SelectionRule,
    Selector,
    stop_rule,
)
from pokermind.robust.tree import CompiledTree, FloatArray
from pokermind.simulation import collect_metadata
from pokermind.strategy import GameStrategy, game_strategy_digest

CHECKPOINTS: Final = (500, 1000, 2000)
GAP_PS: Final = (0.1, 0.3, 0.6, 1.0)
CONVERGENCE_HANDS: Final = (20, 100, 500)
TEMPERATURES: Final = (1.0, 0.5, 0.25)
INFLATIONS: Final = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
LEVELS: Final = (0.5, 0.8, 0.9, 0.95)
TWO_SIDED_Z: Final = (0.6745, 1.2816, 1.6449, 1.9600)
DECISION_Z: Final = (0.0, 0.6745, 1.2816, 1.6449)
"""One-sided levels 50 %, 75 %, 90 %, 95 % (z = 0 is the posterior mean)."""
SAMPLES: Final = 16
PRIMARY_DELTA: Final = 0.1
CHANGE_OFFSETS: Final = (0, 20, 50, 100, 200)
BENCH_HANDS: Final = (0, 20, 50, 100, 200, 500)
NEAR_BEST: Final = 0.9
"""Rule choice: among variants within 90 % of the best development mean gain, take the
one with the fewest negative outcomes."""
OUT_DIR: Final = Path("runs/m7b-adaptive-reliability")


@dataclass(frozen=True, slots=True)
class ReliabilityConfig:
    hero_path: Path
    m7a_metrics: Path | None = Path("runs/m7a-robust-response/metrics.json")
    iterations: int = 500
    samples: int = SAMPLES
    development_seeds: int = 2
    seeds: int = 3
    change_seeds: int = 2
    bait_count: int = 24
    hands: tuple[int, ...] = HANDS
    deltas: tuple[float, ...] = DELTAS
    workers: int = 0

    def __post_init__(self) -> None:
        if min(self.seeds, self.development_seeds, self.change_seeds, self.bait_count) <= 0:
            raise ValueError("counts must be positive")
        if self.samples < 2:
            raise ValueError("need at least 2 posterior samples")


# -- worker state ----------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


def build_reliability_world(hero_path: Path, bait_count: int) -> tuple[World, CompiledTree]:
    """The M6/M7A world plus two equilibrium opponents and the large bait-and-switch set."""
    world = build_world(hero_path, 4, 2, 8)
    tree = CompiledTree(world.game)

    def add(group: str, opponent: Opponent) -> None:
        world.opponents[opponent.name] = opponent
        world.groups.setdefault(group, []).append(opponent.name)

    solver = BatchedRNR(tree, tree.normalize(np.zeros(tree.num_slots)), [0.0])
    solver.iterate(500)
    cfr_plus = tree.to_table(solver.result().hero[0])
    for name, policy in (("teacher", world.cfr), ("cfr_plus_500", cfr_plus)):
        add(
            "equilibrium",
            Opponent(
                f"eq/{name}",
                "equilibrium",
                "out_of_population",
                {},
                ({k: dict(v) for k, v in policy.items()},),
            ),
        )
    for o in make_bait_switch_population(bait_count, world.specs):
        add("bait_switch_large", o)
    return world, tree


def _init(hero_path: Path, bait_count: int) -> None:
    world, tree = build_reliability_world(hero_path, bait_count)
    _STATE["world"], _STATE["tree"] = world, tree
    _STATE["selector"] = Selector(
        world.game, world.specs, tree, world.cfr, world.priors.cfr_exploitability
    )
    _STATE["adapter"] = Adapter(world.game, world.specs, world.priors, M6_SETTINGS)


def _ctx() -> tuple[World, CompiledTree, Selector, Adapter]:
    return _STATE["world"], _STATE["tree"], _STATE["selector"], _STATE["adapter"]


# -- reconstruction and posterior ----------------------------------------------------------


@dataclass(slots=True)
class Recon:
    estimate: GameStrategy
    model: CardConditionalModel
    hands: list[HandEvidence]
    fit: Reconstruction
    prior: tuple[str, float]


def reconstruct(world: World, adapter: Adapter, records: Sequence[ObserverHandRecord]) -> Recon:
    """M6's reconstruction exactly (prior selection, penalized EM)."""
    hands = [evidence(r) for r in records]
    prior, _ = adapter.select_prior(hands)
    model = CardConditionalModel(world.priors.theta(prior[0]), prior[1])
    fit = model.fit(hands)
    return Recon(expand(fit.theta, world.specs), model, hands, fit, prior)


def posterior(rec: Recon, samples: int, temperature: float, key: Sequence[int]) -> list[Theta]:
    rng = np.random.default_rng(np.random.SeedSequence(list(key)))
    return rec.model.posterior_samples(
        rec.hands, samples, rng, start=rec.fit.theta, temperature=temperature
    )


def canonical_truth(truth: GameStrategy, specs: Mapping[str, Any]) -> Theta:
    """Suit-averaged true policy per canonical information set (no clipping)."""
    actions = canonical_actions(specs)
    sums = {k: np.zeros(len(a)) for k, a in actions.items()}
    counts = dict.fromkeys(actions, 0)
    for key, spec in specs.items():
        ck = canonical_infoset(key)
        sums[ck] += np.array([truth[key][a] for a in spec.actions])
        counts[ck] += 1
    return {k: v / counts[k] for k, v in sums.items()}


def parameter_coverage(
    thetas: Sequence[Theta], truth: Theta, weights: Mapping[str, float]
) -> dict[str, Any]:
    """Reach-weighted coverage of the opponent model's own intervals
    ``mean ± z·κ·sd`` (per canonical information set and action)."""
    hits = np.zeros((len(INFLATIONS), len(LEVELS)))
    total = 0.0
    for key, w in weights.items():
        if w <= 0:
            continue
        draws = np.array([th[key] for th in thetas])
        mean, sd = draws.mean(axis=0), draws.std(axis=0, ddof=1)
        err = np.abs(truth[key] - mean)
        for i, kappa in enumerate(INFLATIONS):
            for j, z in enumerate(TWO_SIDED_Z):
                hits[i, j] += w * float(np.mean(err <= z * kappa * sd))
        total += w
    return {"hits": hits.tolist(), "weight": total}


# -- stage 1: convergence --------------------------------------------------------------------


def _tv(a: FloatArray, b: FloatArray, tree: CompiledTree) -> FloatArray:
    """Total-variation distance per information set."""
    out: FloatArray = 0.5 * np.bincount(
        tree.slot_iset, weights=np.abs(a - b), minlength=tree.num_isets
    )
    return out


def convergence_task(args: tuple[str, int, ReliabilityConfig]) -> dict[str, Any]:
    name, n, config = args
    world, tree, selector, adapter = _ctx()
    g, specs, ce = world.game, world.specs, world.priors.cfr_exploitability
    opponent = world.opponents[name]
    truth = opponent.policies[0]
    records = observe(world, opponent, 0, max(n, 1))[:n]
    rec = reconstruct(world, adapter, records)
    model = tree.to_vector(rec.estimate)
    truth_vec = tree.to_vector(truth)
    cfr_true = hero_value(g, world.cfr, truth, specs)
    solver = BatchedRNR(tree, model, P_GRID)
    out: dict[str, Any] = {"opponent": name, "hands": n, "checkpoints": {}}
    heroes: dict[int, FloatArray] = {}
    done = 0
    for cp in CHECKPOINTS:
        t0 = time.perf_counter()
        solver.iterate(cp - done)
        done = cp
        seconds = time.perf_counter() - t0
        res = solver.result()
        heroes[cp] = res.hero
        ev = selector.evaluator
        expl = ev.exploitability(res.hero)
        values = ev.hero_values(res.hero, np.vstack([model, truth_vec]))
        keep = stop_rule(expl, ce + max(config.deltas))
        cands = CandidateSet(tuple(P_GRID[:keep]), res.hero[:keep], expl[:keep], values[:keep, 0])
        chosen = {}
        for d in config.deltas:
            s = selector.select(cands, ce + d, SelectionRule())
            gain = 0.0 if s.index is None else float(values[s.index, 1] - cfr_true)
            chosen[str(d)] = {"p": s.p, "gain": gain, "exploitability": s.exploitability}
        gaps = {}
        for p in GAP_PS:
            j = P_GRID.index(p)
            b = rnr_bounds(
                g,
                specs,
                tree.to_table(res.hero[j]),
                tree.to_table(res.opponent[j]),
                rec.estimate,
                p,
            )
            gaps[str(p)] = {"gap": b.gap, "lower": b.lower, "upper": b.upper}
        out["checkpoints"][str(cp)] = {
            "seconds_increment": seconds,
            "exploitability": expl.tolist(),
            "model_value": values[:, 0].tolist(),
            "true_gain": (values[:, 1] - cfr_true).tolist(),
            "selected": chosen,
            "gaps": gaps,
        }
    final = heroes[CHECKPOINTS[-1]]
    diffs: dict[str, Any] = {}
    for cp in CHECKPOINTS[:-1]:
        per_p = []
        for j, p in enumerate(P_GRID):
            w = selector.evaluator.hero_reach_weights(final[j], model)
            tv = _tv(heroes[cp][j], final[j], tree)
            hero_sets = w > 0
            per_p.append(
                {
                    "p": p,
                    "max_abs": float(np.abs(heroes[cp][j] - final[j]).max()),
                    "max_tv": float(tv[hero_sets].max()),
                    "reach_weighted_tv": float((w * tv).sum() / w.sum()),
                }
            )
        diffs[f"{cp}_vs_{CHECKPOINTS[-1]}"] = per_p
    out["strategy_differences"] = diffs
    return out


# -- stage 2: calibration (development only) -----------------------------------------------


def calibration_task(args: tuple[str, int, ReliabilityConfig]) -> dict[str, Any]:
    name, seed, config = args
    world, tree, selector, adapter = _ctx()
    g, specs, ce = world.game, world.specs, world.priors.cfr_exploitability
    opponent = world.opponents[name]
    truth = opponent.policies[0]
    truth_vec = tree.to_vector(truth)
    cfr_true = hero_value(g, world.cfr, truth, specs)
    ctruth = canonical_truth(truth, specs)
    cweights: dict[str, float] = {}
    for key, w in opponent_infoset_weights(g, world.cfr, truth).items():
        cweights[canonical_infoset(key)] = cweights.get(canonical_infoset(key), 0.0) + w
    records = observe(world, opponent, seed, max(config.hands))
    rows = []
    for n in config.hands:
        rec = reconstruct(world, adapter, records[:n])
        cands, _ = selector.candidates(
            rec.estimate, ce + max(config.deltas), iterations=config.iterations
        )
        true_gain = selector.evaluator.hero_values(cands.hero, truth_vec[None, :])[:, 0] - cfr_true
        row: dict[str, Any] = {
            "hands": n,
            "ps": list(cands.ps),
            "exploitability": cands.exploitability.tolist(),
            "model_value": cands.model_value.tolist(),
            "true_gain": true_gain.tolist(),
            "sample_gain": {},
            "parameter_coverage": {},
            "timings": dict(cands.timings),
        }
        for tau in TEMPERATURES:
            t0 = time.perf_counter()
            thetas = posterior(
                rec, config.samples, tau, (stream_id(name, seed, "posterior", str(tau)), n)
            )
            row["timings"][f"gibbs_{tau}"] = time.perf_counter() - t0
            selector.add_samples(cands, [expand(th, specs) for th in thetas])
            assert cands.sample_gain is not None
            row["sample_gain"][str(tau)] = cands.sample_gain.tolist()
            row["parameter_coverage"][str(tau)] = parameter_coverage(thetas, ctruth, cweights)
        rows.append(row)
    return {"opponent": name, "family": opponent.family, "seed": seed, "rows": rows}


def _offline_select(
    ps: Sequence[float], expl: Sequence[float], score: FloatArray, budget: float, rule: str
) -> int | None:
    """:meth:`Selector.select` on stored candidates (no certificate; stored values are
    the batched exact ones)."""
    feasible = [i for i in range(len(ps)) if expl[i] <= budget + 1e-12]
    if not feasible:
        return None
    best = max(feasible, key=lambda i: (score[i], -ps[i]))
    if rule == "lcb" and score[best] <= 0.0:
        return None
    return best


def _scores(row: Mapping[str, Any], rule: str, tau: float, z: float, kappa: float) -> FloatArray:
    if rule == "point":
        return np.asarray(row["model_value"])
    g = np.asarray(row["sample_gain"][str(tau)])
    out: FloatArray = g.mean(axis=1) - z * kappa * g.std(axis=1, ddof=1)
    return out


def gain_coverage(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Coverage of ``mean ± z·κ·sd`` (two-sided) and of ``mean − z·κ·sd`` (one-sided lower
    bound) for the true gain over CFR, over every candidate with ``p > 0``."""
    out: dict[str, Any] = {}
    for tau in TEMPERATURES:
        err, sd, below = [], [], []
        for r in results:
            for row in r["rows"]:
                g = np.asarray(row["sample_gain"][str(tau)])
                truth = np.asarray(row["true_gain"])
                for i, p in enumerate(row["ps"]):
                    if p == 0.0:
                        continue
                    m, spread = float(g[i].mean()), float(g[i].std(ddof=1))
                    err.append(abs(truth[i] - m))
                    sd.append(spread)
                    below.append(m - truth[i])
        e, s, b = np.array(err), np.array(sd), np.array(below)
        per_kappa = {}
        for kappa in INFLATIONS:
            two = [float(np.mean(e <= z * kappa * s)) for z in TWO_SIDED_Z]
            one = [float(np.mean(b <= z * kappa * s)) for z in DECISION_Z[1:]]
            per_kappa[str(kappa)] = {
                "two_sided": dict(zip(map(str, LEVELS), two, strict=True)),
                "two_sided_error": float(
                    np.mean([abs(c - lv) for c, lv in zip(two, LEVELS, strict=True)])
                ),
                "one_sided_lower": dict(zip(("0.75", "0.9", "0.95"), one, strict=True)),
            }
        # the κ that makes the 90 % two-sided interval exact
        ratio = np.divide(e, s, out=np.full_like(e, np.inf), where=s > 0)
        kappa90 = float(np.quantile(ratio, 0.9) / TWO_SIDED_Z[2])
        out[str(tau)] = {
            "intervals": len(e),
            "median_sd": float(np.median(s)),
            "by_inflation": per_kappa,
            "empirical_inflation_90": kappa90,
        }
    return out


def model_coverage(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tau in TEMPERATURES:
        hits = np.zeros((len(INFLATIONS), len(LEVELS)))
        weight = 0.0
        for r in results:
            for row in r["rows"]:
                pc = row["parameter_coverage"][str(tau)]
                hits += np.asarray(pc["hits"])
                weight += pc["weight"]
        cov = hits / weight
        out[str(tau)] = {
            str(k): {
                "two_sided": dict(zip(map(str, LEVELS), cov[i].tolist(), strict=True)),
                "two_sided_error": float(np.mean(np.abs(cov[i] - np.array(LEVELS)))),
            }
            for i, k in enumerate(INFLATIONS)
        }
    return out


def choose_calibration(coverage: Mapping[str, Any]) -> dict[str, Any]:
    """The (temperature, inflation) with the smallest two-sided calibration error of the
    gain intervals; ties go to τ = 1 and the smaller inflation (fewer departures)."""
    best: tuple[float, float, float] | None = None
    for tau in TEMPERATURES:
        for kappa in INFLATIONS:
            err = coverage[str(tau)]["by_inflation"][str(kappa)]["two_sided_error"]
            key = (round(err, 6), -tau, kappa)
            if best is None or key < (round(best[0], 6), -best[1], best[2]):
                best = (err, tau, kappa)
    assert best is not None
    return {"temperature": best[1], "inflation": best[2], "two_sided_error": best[0]}


def decision_variants(calibration: Mapping[str, Any]) -> dict[str, tuple[str, float, float, float]]:
    tau, kappa = calibration["temperature"], calibration["inflation"]
    out: dict[str, tuple[str, float, float, float]] = {"point": ("point", 1.0, 0.0, 1.0)}
    for z in DECISION_Z:
        out[f"lcb_calibrated_z{z:g}"] = ("lcb", tau, z, kappa)
        if (tau, kappa) != (1.0, 1.0):
            out[f"lcb_raw_z{z:g}"] = ("lcb", 1.0, z, 1.0)
    return out


def simulate_rules(
    results: Sequence[Mapping[str, Any]],
    variants: Mapping[str, tuple[str, float, float, float]],
    deltas: Sequence[float],
    cfr_exploitability: float,
) -> dict[str, Any]:
    """Offline decisions on stored candidates: gain, harm, tails, exploitability."""
    out: dict[str, Any] = {}
    for vname, (rule, tau, z, kappa) in variants.items():
        per_delta: dict[str, Any] = {}
        for d in deltas:
            by_opp: dict[str, list[float]] = {}
            flat, expl, fallbacks = [], [], 0
            for r in results:
                for row in r["rows"]:
                    idx = _offline_select(
                        row["ps"],
                        row["exploitability"],
                        _scores(row, rule, tau, z, kappa),
                        cfr_exploitability + d,
                        rule,
                    )
                    gain = 0.0 if idx is None else float(row["true_gain"][idx])
                    fallbacks += idx is None
                    expl.append(cfr_exploitability if idx is None else row["exploitability"][idx])
                    by_opp.setdefault(r["opponent"], []).append(gain)
                    flat.append(gain)
            arr = np.array(flat)
            per_delta[str(d)] = {
                "gain": clustered(by_opp),
                "harm_rate": float(np.mean(arr < -1e-9)),
                "mean_loss": float(np.mean(np.minimum(arr, 0.0))),
                "worst": float(arr.min()),
                "p10": float(np.percentile(arr, 10)),
                "fallback_rate": fallbacks / len(arr),
                "max_exploitability": float(max(expl)),
            }
        out[vname] = per_delta
    return out


def choose_rule(simulated: Mapping[str, Any]) -> str:
    """Among variants whose development mean gain at the primary budget is at least
    ``NEAR_BEST`` of the best, the one with the lowest harm rate (then higher mean)."""
    d = str(PRIMARY_DELTA)
    means = {v: s[d]["gain"]["mean"] for v, s in simulated.items()}
    best = max(means.values())
    near = [v for v, m in means.items() if m >= NEAR_BEST * best]
    return min(near, key=lambda v: (simulated[v][d]["harm_rate"], -means[v]))


# -- stage 3: frozen evaluation --------------------------------------------------------------


def _decide(
    selector: Selector,
    cands: CandidateSet,
    budget: float,
    rule: SelectionRule,
    truth_vec: FloatArray,
    cfr_true: float,
) -> dict[str, Any]:
    s = selector.select(cands, budget, rule)
    if s.index is None:
        gain = 0.0
    else:
        gain = float(
            selector.evaluator.hero_values(cands.hero[s.index][None, :], truth_vec[None, :])[0, 0]
            - cfr_true
        )
    return {
        "gain": gain,
        "exploitability": s.exploitability,
        "p": s.p,
        "fallback": s.index is None,
        "score": s.score,
    }


def _mix(
    world: World,
    estimate: GameStrategy,
    truth: GameStrategy,
    cfr_true: float,
    deltas: Sequence[float],
) -> dict[str, Any]:
    g, specs, ce = world.game, world.specs, world.priors.cfr_exploitability
    mixer = Mixer(g, specs, world.cfr, hero_best_response(g, estimate, specs), ce)
    out = {}
    for d in deltas:
        s = mixer.largest_alpha(ce + d)
        out[str(d)] = {
            "gain": hero_value(g, mixer.policy(s.alpha), truth, specs) - cfr_true,
            "exploitability": s.exploitability,
            "alpha": s.alpha,
        }
    return out


def _snapshot_decisions(
    world: World,
    selector: Selector,
    rec: Recon,
    research: ResearchConfig,
    config: ReliabilityConfig,
    truths: Sequence[GameStrategy],
    key: Sequence[int],
    with_mix: bool,
) -> list[dict[str, Any]]:
    """Candidates and decisions for one snapshot, evaluated against each of ``truths``
    (two truths only around an opponent's switch)."""
    tree = selector.tree
    ce = world.priors.cfr_exploitability
    sel = research.selection
    t0 = time.perf_counter()
    cands, _ = selector.candidates(
        rec.estimate,
        ce + max(config.deltas),
        research.response.p_grid,
        research.response.iterations,
    )
    timings = dict(cands.timings)
    frozen = SelectionRule(sel.rule, sel.z, sel.inflation)
    if sel.rule == "lcb":
        t1 = time.perf_counter()
        thetas = posterior(
            rec, research.reconstruction.posterior_samples, research.reconstruction.temperature, key
        )
        timings["gibbs"] = time.perf_counter() - t1
        selector.add_samples(cands, [expand(th, world.specs) for th in thetas])
    timings["total_selection"] = time.perf_counter() - t0
    outs = []
    for truth in truths:
        truth_vec = tree.to_vector(truth)
        cfr_true = hero_value(world.game, world.cfr, truth, world.specs)
        out: dict[str, Any] = {"point": {}, "frozen": {}}
        for d in config.deltas:
            b = ce + d
            out["point"][str(d)] = _decide(selector, cands, b, SelectionRule(), truth_vec, cfr_true)
            out["frozen"][str(d)] = _decide(selector, cands, b, frozen, truth_vec, cfr_true)
        if with_mix:
            t2 = time.perf_counter()
            out["mix"] = _mix(world, rec.estimate, truth, cfr_true, config.deltas)
            timings["mix"] = time.perf_counter() - t2
        out["timings"] = dict(timings)
        out["prior"] = list(rec.prior)
        outs.append(out)
    return outs


def evaluate_task(args: tuple[str, int, ReliabilityConfig, ResearchConfig, bool]) -> dict[str, Any]:
    name, seed, config, research, with_mix = args
    world, _, selector, adapter = _ctx()
    opponent = world.opponents[name]
    truth = opponent.policies[0]
    records = observe(world, opponent, seed, max(config.hands))
    rows = []
    for n in config.hands:
        rec = reconstruct(world, adapter, records[:n])
        key = (stream_id(name, seed, "posterior", "frozen"), n)
        row = _snapshot_decisions(world, selector, rec, research, config, [truth], key, with_mix)[0]
        row["hands"] = n
        rows.append(row)
    return {"opponent": name, "family": opponent.family, "seed": seed, "rows": rows}


def change_eval_task(
    args: tuple[str, int, ReliabilityConfig, ResearchConfig],
) -> dict[str, Any]:
    name, seed, config, research = args
    world, _, selector, adapter = _ctx()
    opponent = world.opponents[name]
    assert opponent.switch_hand is not None
    switch = opponent.switch_hand
    checkpoints = sorted({switch + o for o in CHANGE_OFFSETS})
    records = observe(world, opponent, seed, max(checkpoints))
    cd = research.change_detection
    detector = DetectorConfig(short_decay=cd.short_decay, threshold=cd.threshold, drift=cd.drift)
    observer = ChangeAwareObserver(world.priors.public_prior, detector, retain=cd.retain)
    out: dict[str, Any] = {"opponent": name, "seed": seed, "checkpoints": {}}
    for n, record in enumerate(records, start=1):
        observer.observe(record)
        if n not in checkpoints:
            continue
        rec = reconstruct(world, adapter, observer.retained())
        key = (stream_id(name, seed, "posterior", "change"), n)
        # at the switch: also against the old policy (what the adaptation was earning)
        tags = ["pre", "0"] if n == switch else [str(n - switch)]
        truths = (
            [opponent.policies[0], opponent.policies[1]] if n == switch else [opponent.policies[1]]
        )
        rows = _snapshot_decisions(world, selector, rec, research, config, truths, key, True)
        for tag, row in zip(tags, rows, strict=True):
            out["checkpoints"][tag] = row
    out["resets"] = observer.resets
    return out


# -- aggregation -----------------------------------------------------------------------------


def _clustered_rows(results: Sequence[Mapping[str, Any]], fn: Any, n: int) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for r in results:
        rows = {row["hands"]: row for row in r["rows"]}
        if n in rows:
            per.setdefault(r["opponent"], []).append(fn(rows, n))
    return clustered(per)


def summarize_evaluation(
    results: Sequence[Mapping[str, Any]],
    config: ReliabilityConfig,
    mix_rows: Mapping[tuple[str, int, int], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Totals, increments over zero-hand play, paired differences, harm and worst cases."""
    hands, deltas = config.hands, [str(d) for d in config.deltas]

    def mix(rows: Mapping[int, Any], n: int, d: str, r: Mapping[str, Any]) -> float:
        if "mix" in rows[n]:
            return float(rows[n]["mix"][d]["gain"])
        assert mix_rows is not None
        return float(mix_rows[(r["opponent"], r["seed"], n)]["mix"][d]["gain"])

    out: dict[str, Any] = {"gain": {}, "increment_vs_zero_hand": {}, "paired": {}, "risk": {}}
    for pol in ("point", "frozen"):
        out["gain"][pol] = {
            d: {
                str(n): _clustered_rows(
                    results, lambda rows, n, pol=pol, d=d: rows[n][pol][d]["gain"], n
                )
                for n in hands
            }
            for d in deltas
        }
        out["increment_vs_zero_hand"][pol] = {
            d: {
                str(n): _clustered_rows(
                    results,
                    lambda rows, n, pol=pol, d=d: (
                        rows[n][pol][d]["gain"] - rows[0]["point"][d]["gain"]
                    ),
                    n,
                )
                for n in hands
                if n > 0
            }
            for d in deltas
        }
        all_rows = [(r, row) for r in results for row in r["rows"]]
        out["risk"][pol] = {
            d: {
                "harm_rate": float(np.mean([row[pol][d]["gain"] < -1e-9 for _, row in all_rows])),
                "worst": sorted(
                    (
                        {
                            "gain": row[pol][d]["gain"],
                            "opponent": r["opponent"],
                            "seed": r["seed"],
                            "hands": row["hands"],
                        }
                        for r, row in all_rows
                    ),
                    key=lambda x: x["gain"],
                )[:3],
                "max_exploitability": max(row[pol][d]["exploitability"] for _, row in all_rows),
                "fallback_rate": float(np.mean([row[pol][d]["fallback"] for _, row in all_rows])),
            }
            for d in deltas
        }
    mixed: dict[str, Any] = {}
    for d in deltas:
        mixed[d] = {
            str(n): _clustered_rows(
                results,
                lambda rows, n, d=d: rows[n]["frozen"][d]["gain"] - rows[n]["point"][d]["gain"],
                n,
            )
            for n in hands
        }
    out["paired"]["frozen-point"] = mixed
    if mix_rows is not None or all("mix" in row for r in results for row in r["rows"]):
        out["gain"]["mix"] = {}
        out["paired"]["frozen-mix"] = {}
        for d in deltas:
            vals: dict[str, dict[str, list[float]]] = {str(n): {} for n in hands}
            diffs: dict[str, dict[str, list[float]]] = {str(n): {} for n in hands}
            for r in results:
                rows = {row["hands"]: row for row in r["rows"]}
                for n in hands:
                    m = mix(rows, n, d, r)
                    vals[str(n)].setdefault(r["opponent"], []).append(m)
                    diffs[str(n)].setdefault(r["opponent"], []).append(
                        rows[n]["frozen"][d]["gain"] - m
                    )
            out["gain"]["mix"][d] = {n: clustered(v) for n, v in vals.items()}
            out["paired"]["frozen-mix"][d] = {n: clustered(v) for n, v in diffs.items()}
    fams: dict[str, list[Mapping[str, Any]]] = {}
    for r in results:
        fams.setdefault(r["family"], []).append(r)
    out["by_family"] = {
        fam: {
            pol: {
                str(n): statistics.fmean(
                    row[pol][str(PRIMARY_DELTA)]["gain"]
                    for r in rs
                    for row in r["rows"]
                    if row["hands"] == n
                )
                for n in (0, 100, 500)
                if n in hands
            }
            for pol in ("point", "frozen")
        }
        for fam, rs in fams.items()
    }
    keys = {k for r in results for row in r["rows"] for k in row["timings"]}
    out["timings"] = {
        k: statistics.fmean(
            row["timings"][k] for r in results for row in r["rows"] if k in row["timings"]
        )
        for k in sorted(keys)
    }
    return out


def summarize_changes(
    results: Sequence[Mapping[str, Any]], deltas: Sequence[float]
) -> dict[str, Any]:
    tags = ["pre", *[str(o) for o in CHANGE_OFFSETS]]
    out: dict[str, Any] = {}
    for pol in ("point", "frozen", "mix"):
        out[pol] = {}
        for d in map(str, deltas):
            out[pol][d] = {}
            for tag in tags:
                per: dict[str, list[float]] = {}
                for r in results:
                    if tag in r["checkpoints"]:
                        per.setdefault(r["opponent"], []).append(
                            r["checkpoints"][tag][pol][d]["gain"]
                        )
                if per:
                    out[pol][d][tag] = clustered(per)
    d = str(PRIMARY_DELTA)
    flat = {
        pol: [
            (r["checkpoints"][t][pol][d]["gain"], r["opponent"], t)
            for r in results
            for t in tags
            if t in r["checkpoints"] and t != "pre"
        ]
        for pol in ("point", "frozen", "mix")
    }
    out["risk"] = {
        pol: {
            "harm_rate": float(np.mean([g < -1e-9 for g, _, _ in v])),
            "worst": [{"gain": g, "opponent": o, "offset": t} for g, o, t in sorted(v)[:3]],
            "max_exploitability": max(
                r["checkpoints"][t][pol][d]["exploitability"]
                for r in results
                for t in tags
                if t in r["checkpoints"]
            ),
        }
        for pol, v in flat.items()
    }
    out["paired_frozen_minus_mix"] = {}
    for tag in tags:
        pairs: dict[str, list[float]] = {}
        for r in results:
            if tag in r["checkpoints"]:
                c = r["checkpoints"][tag]
                pairs.setdefault(r["opponent"], []).append(
                    c["frozen"][d]["gain"] - c["mix"][d]["gain"]
                )
        if pairs:
            out["paired_frozen_minus_mix"][tag] = clustered(pairs)
    out["resets_per_run"] = statistics.fmean(len(r["resets"]) for r in results)
    return out


def m7a_rows(path: Path | None) -> dict[tuple[str, int, int], Mapping[str, Any]] | None:
    if path is None or not path.is_file():
        return None
    doc = json.loads(path.read_text())
    out = {}
    for r in doc["raw"]["heldout"] + doc["raw"]["out_of_population"]:
        for row in r["rows"]:
            out[(r["opponent"], r["seed"], row["hands"])] = row
    return out


def check_m7a(
    results: Sequence[Mapping[str, Any]],
    recorded: Mapping[tuple[str, int, int], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Point-rule (M7A) gains, parameters and exploitability recomputed here vs M7A's rows."""
    if recorded is None:
        return {"available": False}
    diff, param_mismatch, compared = 0.0, 0, 0
    for r in results:
        for row in r["rows"]:
            ref = recorded.get((r["opponent"], r["seed"], row["hands"]))
            if ref is None:
                continue
            for d, mine in row["point"].items():
                theirs = ref["rnr"][d]
                diff = max(
                    diff,
                    abs(mine["gain"] - theirs["gain"]),
                    abs(mine["exploitability"] - theirs["exploitability"]),
                )
                param_mismatch += mine["p"] != theirs["parameter"]
                compared += 1
    return {
        "available": True,
        "compared": compared,
        "max_abs_difference": diff,
        "parameter_mismatches": param_mismatch,
    }


def _pool(config: ReliabilityConfig) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=config.workers or (os.cpu_count() or 1),
        initializer=_init,
        initargs=(config.hero_path, config.bait_count),
    )


def _write(doc: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")


def _header(world: World, config: ReliabilityConfig, stage: str) -> dict[str, Any]:
    return {
        "experiment": f"m7b-adaptive-reliability/{stage}",
        "hero_path": str(config.hero_path),
        "hero_digest": game_strategy_digest(world.cfr, world.game.action_codes),
        "cfr_exploitability": world.priors.cfr_exploitability,
        "iterations": config.iterations,
        "hands": list(config.hands),
        "deltas": list(config.deltas),
        "metadata": collect_metadata(),
    }


# -- stage drivers ---------------------------------------------------------------------------


def run_convergence(config: ReliabilityConfig, output_dir: Path = OUT_DIR) -> dict[str, Any]:
    start = time.perf_counter()
    world, _ = build_reliability_world(config.hero_path, config.bait_count)
    first = world.groups["heldout"][0]
    tasks = [(first, 0, config)] + [
        (n, h, config)
        for grp in ("heldout", "out_of_population")
        for n in world.groups[grp]
        for h in CONVERGENCE_HANDS
    ]
    with _pool(config) as pool:
        results = list(pool.map(convergence_task, tasks))
    recorded = m7a_rows(config.m7a_metrics)
    doc = _header(world, config, "convergence")
    doc["summary"] = summarize_convergence(results, recorded, world)
    doc["raw"] = results
    doc["elapsed_seconds"] = time.perf_counter() - start
    _write(doc, output_dir / "convergence.json")
    return doc


def summarize_convergence(
    results: Sequence[Mapping[str, Any]],
    recorded: Mapping[tuple[str, int, int], Mapping[str, Any]] | None,
    world: World,
) -> dict[str, Any]:
    out: dict[str, Any] = {"gap": {}, "strategy_difference": {}, "selected": {}, "seconds": {}}
    for cp in map(str, CHECKPOINTS):
        out["gap"][cp] = {
            p: mean_ci([r["checkpoints"][cp]["gaps"][p]["gap"] for r in results])
            | {"max": max(r["checkpoints"][cp]["gaps"][p]["gap"] for r in results)}
            for p in map(str, GAP_PS)
        }
        out["seconds"][cp] = statistics.fmean(
            r["checkpoints"][cp]["seconds_increment"] for r in results
        )
    for label in results[0]["strategy_differences"]:
        out["strategy_difference"][label] = {
            str(p): {
                k: max(r["strategy_differences"][label][j][k] for r in results)
                for k in ("max_abs", "max_tv", "reach_weighted_tv")
            }
            | {
                "mean_reach_weighted_tv": statistics.fmean(
                    r["strategy_differences"][label][j]["reach_weighted_tv"] for r in results
                )
            }
            for j, p in enumerate(P_GRID)
        }
    ref = str(CHECKPOINTS[-1])
    for cp in map(str, CHECKPOINTS[:-1]):
        per_delta = {}
        for d in results[0]["checkpoints"][cp]["selected"]:
            a = [r["checkpoints"][cp]["selected"][d] for r in results]
            b = [r["checkpoints"][ref]["selected"][d] for r in results]
            per_delta[d] = {
                "same_p": float(np.mean([x["p"] == y["p"] for x, y in zip(a, b, strict=True)])),
                "gain_difference": mean_ci(
                    [x["gain"] - y["gain"] for x, y in zip(a, b, strict=True)]
                ),
                "max_abs_gain_difference": max(
                    abs(x["gain"] - y["gain"]) for x, y in zip(a, b, strict=True)
                ),
                "max_exploitability": {
                    cp: max(x["exploitability"] for x in a),
                    ref: max(y["exploitability"] for y in b),
                },
            }
        out["selected"][f"{cp}_vs_{ref}"] = per_delta
    # does the M7A conclusion (RNR beats the mixture) depend on 500 iterations?
    if recorded is not None:
        concl: dict[str, Any] = {}
        held = set(world.groups["heldout"])
        for grp, names in (
            ("heldout", held),
            ("out_of_population", set(world.groups["out_of_population"])),
        ):
            concl[grp] = {}
            for cp in map(str, CHECKPOINTS):
                concl[grp][cp] = {}
                for d in ("0.02", "0.1", "0.25"):
                    concl[grp][cp][d] = {}
                    for n in CONVERGENCE_HANDS:
                        diffs = [
                            r["checkpoints"][cp]["selected"][d]["gain"]
                            - recorded[(r["opponent"], 0, n)]["mix"][d]["gain"]
                            for r in results
                            if r["opponent"] in names and r["hands"] == n
                        ]
                        if diffs:
                            concl[grp][cp][d][str(n)] = mean_ci(diffs)
        out["rnr_minus_mix"] = concl
    return out


def run_calibration(config: ReliabilityConfig, output_dir: Path = OUT_DIR) -> dict[str, Any]:
    start = time.perf_counter()
    world, _ = build_reliability_world(config.hero_path, config.bait_count)
    tasks = [
        (n, s, config) for n in world.groups["development"] for s in range(config.development_seeds)
    ]
    with _pool(config) as pool:
        results = list(pool.map(calibration_task, tasks))
    doc = _header(world, config, "calibration")
    doc.update(analyse_calibration(results, config, world.priors.cfr_exploitability))
    doc["raw"] = results
    doc["elapsed_seconds"] = time.perf_counter() - start
    _write(doc, output_dir / "calibration.json")
    return doc


def analyse_calibration(
    results: Sequence[Mapping[str, Any]], config: ReliabilityConfig, cfr_exploitability: float
) -> dict[str, Any]:
    gains = gain_coverage(results)
    chosen = choose_calibration(gains)
    variants = decision_variants(chosen)
    simulated = simulate_rules(results, variants, config.deltas, cfr_exploitability)
    rule = choose_rule(simulated)
    kind, tau, z, kappa = variants[rule]
    timing_keys = {k for r in results for row in r["rows"] for k in row["timings"]}
    return {
        "gain_coverage": gains,
        "model_coverage": model_coverage(results),
        "calibration_choice": chosen
        | {"rule": "smallest two-sided calibration error of gain intervals"},
        "decision_simulation": simulated,
        "frozen_proposal": {
            "variant": rule,
            "rule": kind,
            "temperature": tau if kind == "lcb" else 1.0,
            "z": z,
            "inflation": kappa if kind == "lcb" else 1.0,
            "criterion": (
                f"variants with development mean gain at delta={PRIMARY_DELTA} >= "
                f"{NEAR_BEST} x best; then lowest harm rate"
            ),
        },
        "timings": {
            k: statistics.fmean(
                row["timings"][k] for r in results for row in r["rows"] if k in row["timings"]
            )
            for k in sorted(timing_keys)
        },
    }


def run_evaluation(
    config: ReliabilityConfig, research: ResearchConfig, output_dir: Path = OUT_DIR
) -> dict[str, Any]:
    start = time.perf_counter()
    world, _ = build_reliability_world(config.hero_path, config.bait_count)
    digest = game_strategy_digest(world.cfr, world.game.action_codes)
    if digest != research.hero.digest:
        raise ValueError("hero strategy does not match the frozen research configuration")
    recorded = m7a_rows(config.m7a_metrics)
    stationary = [
        (n, s, config, research, grp == "equilibrium" or recorded is None)
        for grp in ("heldout", "out_of_population", "equilibrium")
        for n in world.groups[grp]
        for s in range(config.seeds)
    ]
    changing = [
        (n, s, config, research)
        for grp in ("heldout_switching", "bait_switch_large")
        for n in world.groups[grp]
        for s in range(config.change_seeds)
    ]
    with _pool(config) as pool:
        stat = list(pool.map(evaluate_task, stationary))
        change = list(pool.map(change_eval_task, changing))
    groups = {
        grp: [r for r in stat if r["opponent"] in set(world.groups[grp])]
        for grp in ("heldout", "out_of_population", "equilibrium")
    }
    balanced = [r for r in groups["heldout"] if r["family"] == "balanced"] + groups["equilibrium"]
    doc = _header(world, config, "evaluation")
    doc["research_config"] = research.to_dict() | {"digest": research.digest()}
    doc["m7a_reproduction"] = check_m7a(groups["heldout"] + groups["out_of_population"], recorded)
    doc["heldout"] = summarize_evaluation(groups["heldout"], config, recorded)
    doc["out_of_population"] = summarize_evaluation(groups["out_of_population"], config, recorded)
    doc["equilibrium"] = summarize_evaluation(groups["equilibrium"], config, None)
    doc["balanced_and_equilibrium"] = summarize_evaluation(balanced, config, recorded)
    doc["switching"] = summarize_changes(
        [r for r in change if r["opponent"] in set(world.groups["heldout_switching"])],
        config.deltas,
    )
    doc["bait_switch_large"] = summarize_changes(
        [r for r in change if r["opponent"] in set(world.groups["bait_switch_large"])],
        config.deltas,
    )
    doc["raw"] = {"stationary": stat, "change": change}
    doc["elapsed_seconds"] = time.perf_counter() - start
    _write(doc, output_dir / "evaluation.json")
    return doc


# -- stage 4: benchmark ------------------------------------------------------------------------


def _m7a_procedure(
    world: World,
    tree: CompiledTree,
    estimate: GameStrategy,
    deltas: Sequence[float],
    iterations: int,
) -> dict[str, Any]:
    """M7A's selection, unchanged: sequential solves, recursive exact evaluation."""
    ce = world.priors.cfr_exploitability
    cands, _ = _grid(
        world,
        "rnr",
        P_GRID,
        lambda p: rnr_strategy(tree, estimate, p, iterations),
        estimate,
        ce + max(deltas),
    )
    out = {}
    for d in deltas:
        c = _select(cands, ce + d)
        out[str(d)] = (
            {"p": None, "policy": world.cfr, "exploitability": ce}
            if c is None
            else {"p": c.parameter, "policy": c.policy, "exploitability": c.exploitability}
        )
    return out


def benchmark(config: ReliabilityConfig, research: ResearchConfig) -> dict[str, Any]:
    """Single-process timing on identical inputs (run when nothing else is running)."""
    world, tree = build_reliability_world(config.hero_path, config.bait_count)
    selector = Selector(world.game, world.specs, tree, world.cfr, world.priors.cfr_exploitability)
    adapter = Adapter(world.game, world.specs, world.priors, M6_SETTINGS)
    g, specs, ce = world.game, world.specs, world.priors.cfr_exploitability
    names = [
        world.groups["heldout"][0],
        world.groups["heldout"][17],
        world.groups["out_of_population"][3],
        world.groups["out_of_population"][7],
    ]
    variants = ("m7a", "batched", "batched_cold200", "batched_warm200")
    rows: list[dict[str, Any]] = []
    for name in names:
        opp = world.opponents[name]
        truth = opp.policies[0]
        cfr_true = hero_value(g, world.cfr, truth, specs)
        records = observe(world, opp, 0, max(BENCH_HANDS))
        warm: BatchedRNR | None = None
        for n in BENCH_HANDS:
            rec = reconstruct(world, adapter, records[:n])
            row: dict[str, Any] = {"opponent": name, "hands": n}
            for v in variants:
                t0 = time.perf_counter()
                if v == "m7a":
                    picks = _m7a_procedure(
                        world, tree, rec.estimate, config.deltas, config.iterations
                    )
                    seconds = time.perf_counter() - t0
                    res = {
                        d: {
                            "p": x["p"],
                            "exploitability": x["exploitability"],
                            "gain": hero_value(g, x["policy"], truth, specs) - cfr_true,
                        }
                        for d, x in picks.items()
                    }
                    row[v] = {"seconds": seconds, "selected": res}
                    continue
                its = config.iterations if v == "batched" else 200
                start = warm if (v == "batched_warm200" and warm is not None) else None
                if v == "batched_warm200" and warm is None:
                    its = config.iterations  # first snapshot: cold
                cands, solver = selector.candidates(
                    rec.estimate, ce + max(config.deltas), iterations=its, start=start
                )
                res = {}
                for d in config.deltas:
                    s = selector.select(cands, ce + d, SelectionRule())
                    pol = selector.policy(cands, s)
                    res[str(d)] = {
                        "p": s.p,
                        "exploitability": s.exploitability,
                        "gain": hero_value(g, pol, truth, specs) - cfr_true,
                    }
                seconds = time.perf_counter() - t0
                j = P_GRID.index(0.6)
                out = solver.result()
                gap = rnr_bounds(
                    g,
                    specs,
                    tree.to_table(out.hero[j]),
                    tree.to_table(out.opponent[j]),
                    rec.estimate,
                    0.6,
                ).gap
                row[v] = {"seconds": seconds, "iterations": its, "gap_p0.6": gap, "selected": res}
                if v == "batched_warm200":
                    warm = solver
            if research.selection.rule == "lcb":
                t0 = time.perf_counter()
                thetas = posterior(
                    rec,
                    research.reconstruction.posterior_samples,
                    research.reconstruction.temperature,
                    (1, n),
                )
                cands, _ = selector.candidates(rec.estimate, ce + max(config.deltas))
                t1 = time.perf_counter()
                selector.add_samples(cands, [expand(th, specs) for th in thetas])
                row["posterior"] = {
                    "gibbs_seconds": t1 - t0 - cands.timings["solve"] - cands.timings["evaluate"],
                    "sample_evaluation_seconds": time.perf_counter() - t1,
                }
            rows.append(row)
    # memory: one 100-hand snapshot per procedure, traced separately (tracing slows code)
    opp = world.opponents[names[0]]
    rec = reconstruct(world, adapter, observe(world, opp, 0, 100))
    memory = {}
    for v in ("m7a", "batched"):
        tracemalloc.start()
        if v == "m7a":
            _m7a_procedure(world, tree, rec.estimate, config.deltas, config.iterations)
        else:
            selector.candidates(rec.estimate, ce + max(config.deltas), iterations=config.iterations)
        memory[v] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return {
        "rows": rows,
        "peak_traced_bytes": memory,
        "summary": summarize_benchmark(rows, ce),
    }


def summarize_benchmark(rows: Sequence[Mapping[str, Any]], cfr_e: float) -> dict[str, Any]:
    variants = [v for v in ("m7a", "batched", "batched_cold200", "batched_warm200") if v in rows[0]]
    out: dict[str, Any] = {}
    for v in variants:
        sec = [r[v]["seconds"] for r in rows]
        same = {
            d: float(
                np.mean([r[v]["selected"][d]["p"] == r["m7a"]["selected"][d]["p"] for r in rows])
            )
            for d in rows[0]["m7a"]["selected"]
        }
        gdiff = {
            d: max(abs(r[v]["selected"][d]["gain"] - r["m7a"]["selected"][d]["gain"]) for r in rows)
            for d in rows[0]["m7a"]["selected"]
        }
        mean_diff = {
            d: statistics.fmean(
                r[v]["selected"][d]["gain"] - r["m7a"]["selected"][d]["gain"] for r in rows
            )
            for d in rows[0]["m7a"]["selected"]
        }
        out[v] = {
            "mean_seconds": statistics.fmean(sec),
            "max_seconds": max(sec),
            "speedup_vs_m7a": statistics.fmean(r["m7a"]["seconds"] for r in rows)
            / statistics.fmean(sec),
            "same_selection_as_m7a": same,
            "max_abs_gain_difference": gdiff,
            "mean_gain_difference": mean_diff,
            "max_exploitability_minus_budget": max(
                r[v]["selected"][d]["exploitability"] - (cfr_e + float(d))
                for r in rows
                for d in r[v]["selected"]
            ),
        }
        if v != "m7a":
            out[v]["mean_gap_p0.6"] = statistics.fmean(r[v]["gap_p0.6"] for r in rows)
            out[v]["max_gap_p0.6"] = max(r[v]["gap_p0.6"] for r in rows)
    if "posterior" in rows[0]:
        out["posterior"] = {
            k: statistics.fmean(r["posterior"][k] for r in rows) for k in rows[0]["posterior"]
        }
    return out


def run_benchmark(
    config: ReliabilityConfig, research: ResearchConfig, output_dir: Path = OUT_DIR
) -> dict[str, Any]:
    start = time.perf_counter()
    world, _ = build_reliability_world(config.hero_path, config.bait_count)
    doc = _header(world, config, "benchmark")
    doc.update(benchmark(config, research))
    doc["elapsed_seconds"] = time.perf_counter() - start
    _write(doc, output_dir / "benchmark.json")
    return doc


def load_frozen(path: Path) -> ResearchConfig:
    return load_research_config(path)


__all__ = [
    "CHECKPOINTS",
    "DECISION_Z",
    "INFLATIONS",
    "TEMPERATURES",
    "ReliabilityConfig",
    "analyse_calibration",
    "build_reliability_world",
    "choose_calibration",
    "choose_rule",
    "gain_coverage",
    "run_benchmark",
    "run_calibration",
    "run_convergence",
    "run_evaluation",
    "simulate_rules",
]
