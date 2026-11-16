"""Strategy recomputation for one session: the frozen M7B procedure as a job (M8A).

:func:`compute_policy_update` is the acting procedure that M7B evaluated on
held-out opponents (``reliability evaluate``, frozen rule), step for step:
1. **Prior and reconstruction.** M6 prior selection, then penalised EM over
   the retained observer records.
2. **Candidates.** Batched RNR over the frozen ``p`` grid and iteration count,
   cut by M7A's stop rule at the largest reported budget.
3. **Posterior samples and rule.** Tempered posterior samples, then the
   calibrated lower-bound rule with the frozen budget. The pick is certified
   by the M3 exact evaluator, and hero falls back to CFR if no score is
   positive.

The job carries only what the AI legitimately observed (``ObserverHandRecord``s:
public actions, the AI's own cards, public cards, showdown cards). It runs
outside the request path, in a worker process by default. The session
installs the result only after :func:`certify_policy` has checked it again,
independently, in the serving process.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pokermind.adaptive.reconstruction import (
    CardConditionalModel,
    ObserverHandRecord,
    evidence,
    expand,
)
from pokermind.adaptive.response import exploitability
from pokermind.app.context import AppContext, load_app_context
from pokermind.evaluation.game_exact import validate_game_strategy
from pokermind.robust.selection import TOLERANCE
from pokermind.strategy import GameStrategy

CERTIFICATE_MATCH = 1e-9
"""Allowed difference between the worker's certificate and the serving process's."""


class CertificationError(ValueError):
    """A recomputed strategy failed validation and must not be played."""


@dataclass(frozen=True, slots=True)
class UpdateJob:
    session_id: str
    job_id: int
    """Strictly increasing within a session."""
    generation: int
    """Number of change-detection resets when the job was created."""
    based_on_hands: int
    """Hands observed (all, not only retained) when the job was created."""
    records: tuple[ObserverHandRecord, ...]
    """Retained observations: what the AI legitimately saw of the human."""
    seed_key: tuple[int, ...]
    """Posterior-sampling seed (deterministic per session and job)."""


@dataclass(frozen=True, slots=True)
class UpdateResult:
    session_id: str
    job_id: int
    policy: GameStrategy
    p: float | None
    """RNR restriction of the chosen candidate; None for the CFR fallback."""
    certified_exploitability: float
    """M3 exact exploitability computed in the worker."""
    estimate: dict[str, Any] = field(default_factory=dict)
    """Model-based quantities (probabilistic, not verified): prior, scores, posterior gain."""
    seconds: float = 0.0


def compute_policy_update(ctx: AppContext, job: UpdateJob) -> UpdateResult:
    t0 = time.perf_counter()
    research = ctx.research
    rec = research.reconstruction
    hands = [evidence(r) for r in job.records]
    prior, _ = ctx.adapter.select_prior(hands)
    model = CardConditionalModel(ctx.priors.theta(prior[0]), prior[1])
    fit = model.fit(hands)
    estimate = expand(fit.theta, ctx.specs)
    cands, _ = ctx.selector.candidates(
        estimate, ctx.candidate_stop, research.response.p_grid, research.response.iterations
    )
    rule = ctx.rule
    if rule.rule == "lcb":
        rng = np.random.default_rng(np.random.SeedSequence(list(job.seed_key)))
        thetas = model.posterior_samples(
            hands,
            rec.posterior_samples,
            rng,
            start=fit.theta,
            burn_in=rec.burn_in,
            thin=rec.thin,
            temperature=rec.temperature,
        )
        ctx.selector.add_samples(cands, [expand(th, ctx.specs) for th in thetas])
    selection = ctx.selector.select(cands, ctx.budget, rule)
    info: dict[str, Any] = {
        "prior": [prior[0], prior[1]],
        "hands_used": len(hands),
        "candidates": len(cands.ps),
        "rule": rule.rule,
        "score": selection.score,
    }
    if selection.index is not None and cands.sample_gain is not None:
        g = cands.sample_gain[selection.index]
        info["posterior_gain_mean"] = float(g.mean())
        info["posterior_gain_sd"] = float(g.std(ddof=1))
    policy = ctx.selector.policy(cands, selection)
    return UpdateResult(
        session_id=job.session_id,
        job_id=job.job_id,
        policy={k: dict(v) for k, v in policy.items()},
        p=selection.p,
        certified_exploitability=selection.exploitability,
        estimate=info,
        seconds=time.perf_counter() - t0,
    )


def certify_policy(
    ctx: AppContext, policy: Mapping[str, Mapping[int, float]], claimed: float | None = None
) -> float:
    """Validate a strategy table and return its exact exploitability.

    Raises :class:`CertificationError` unless the table covers every information set
    with a valid distribution, its exploitability is within the frozen budget, and it
    agrees with ``claimed`` (the worker's certificate) when given.
    """
    table: GameStrategy = {k: dict(v) for k, v in policy.items()}
    try:
        validate_game_strategy(ctx.game, table, infosets=ctx.specs)
    except ValueError as exc:
        raise CertificationError(f"invalid strategy table: {exc}") from exc
    value = exploitability(ctx.game, table, ctx.specs)
    if not np.isfinite(value) or value > ctx.budget + TOLERANCE:
        raise CertificationError(f"exploitability {value:.6f} exceeds the limit {ctx.budget:.6f}")
    if claimed is not None and abs(value - claimed) > CERTIFICATE_MATCH:
        raise CertificationError(f"certificate mismatch: {value} vs claimed {claimed}")
    return value


# -- worker-process entry points -------------------------------------------------------------

_WORKER: dict[str, AppContext] = {}


def init_worker(config_path: Path, hero_path: Path | None) -> None:
    _WORKER["ctx"] = load_app_context(config_path, hero_path)


def run_in_worker(job: UpdateJob) -> UpdateResult:
    ctx = _WORKER.get("ctx")
    if ctx is None:
        raise RuntimeError("worker context not initialised")
    return compute_policy_update(ctx, job)


__all__ = [
    "CertificationError",
    "UpdateJob",
    "UpdateResult",
    "certify_policy",
    "compute_policy_update",
    "init_worker",
    "run_in_worker",
]
