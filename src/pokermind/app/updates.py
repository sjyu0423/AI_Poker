"""Background strategy recomputation with verified, atomic installation (M8A).

Each session owns an :class:`UpdateScheduler`. Its rules are:
* **At most one job per session.** A request that arrives while a job is
  running is coalesced into one follow-up job, which uses the newest
  observations.
* **Every job is identified** by a strictly increasing ``job_id``, the
  change-detection ``generation`` and ``based_on_hands``.
* **A result is installed only if all of these hold:**
  * the session is open;
  * the job is still the session's current job (not superseded, cancelled
    or timed out);
  * no change reset happened since the job was created;
  * the job is newer than the active strategy's job;
  * :func:`~pokermind.app.policy_update.certify_policy` re-certifies it in
    this process.

  Anything else is recorded and discarded, so an old result can never
  replace newer state.
* **The active strategy is replaced** by one reference assignment under the
  session lock. Hands already dealt keep the strategy they were dealt with.
* **Failures** (exceptions, certification errors, timeouts) leave the
  active, verified strategy in place. The next scheduled trigger retries.

Backends decide *where* a job runs:
* :class:`ProcessBackend`: worker processes; the default for serving.
* :class:`ThreadBackend`: a background thread in this process.
* :class:`InlineBackend`: synchronous; deterministic tests and simulations.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Protocol

from pokermind.app.context import AppContext
from pokermind.app.policy_update import (
    CertificationError,
    UpdateJob,
    UpdateResult,
    certify_policy,
    compute_policy_update,
    init_worker,
    run_in_worker,
)
from pokermind.strategy import GameStrategy, game_strategy_digest

Compute = Callable[[AppContext, UpdateJob], UpdateResult]


@dataclass(frozen=True, slots=True)
class StrategyVersion:
    """An immutable, certified strategy the AI may play."""

    version: int
    job_id: int
    """0 for the baseline."""
    source: str
    """``"baseline_cfr"``, ``"rnr"`` or ``"cfr_fallback"``."""
    policy: GameStrategy
    p: float | None
    certified_exploitability: float
    exploitability_limit: float
    digest: str
    based_on_hands: int
    generation: int
    installed_after_hand: int
    installed_at: float
    estimate: dict[str, Any] = field(default_factory=dict)


def baseline_version(ctx: AppContext) -> StrategyVersion:
    value = certify_policy(ctx, ctx.cfr, ctx.cfr_exploitability)
    return StrategyVersion(
        version=1,
        job_id=0,
        source="baseline_cfr",
        policy=ctx.cfr,
        p=None,
        certified_exploitability=value,
        exploitability_limit=ctx.budget,
        digest=ctx.cfr_digest,
        based_on_hands=0,
        generation=0,
        installed_after_hand=0,
        installed_at=time.time(),
    )


# -- backends ----------------------------------------------------------------------------------


class UpdateBackend(Protocol):
    def submit(self, job: UpdateJob) -> Future[UpdateResult]: ...

    def shutdown(self) -> None: ...


class InlineBackend:
    """Runs the job immediately in the caller's thread (deterministic)."""

    def __init__(self, ctx: AppContext, compute: Compute = compute_policy_update) -> None:
        self.ctx, self.compute = ctx, compute

    def submit(self, job: UpdateJob) -> Future[UpdateResult]:
        future: Future[UpdateResult] = Future()
        future.set_running_or_notify_cancel()
        try:
            future.set_result(self.compute(self.ctx, job))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self) -> None:
        return None


class _ExecutorBackend:
    def __init__(self, executor: Executor) -> None:
        self.executor = executor

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


class ThreadBackend(_ExecutorBackend):
    def __init__(
        self, ctx: AppContext, compute: Compute = compute_policy_update, workers: int = 1
    ) -> None:
        super().__init__(ThreadPoolExecutor(workers, thread_name_prefix="pokermind-update"))
        self.ctx, self.compute = ctx, compute

    def submit(self, job: UpdateJob) -> Future[UpdateResult]:
        return self.executor.submit(self.compute, self.ctx, job)


class ProcessBackend(_ExecutorBackend):
    """Worker processes, each loading the frozen context once (spawn start method)."""

    def __init__(self, config_path: Path, hero_path: Path | None, workers: int = 1) -> None:
        super().__init__(
            ProcessPoolExecutor(
                workers,
                mp_context=get_context("spawn"),
                initializer=init_worker,
                initargs=(config_path, hero_path),
            )
        )

    def submit(self, job: UpdateJob) -> Future[UpdateResult]:
        return self.executor.submit(run_in_worker, job)


# -- the per-session scheduler -------------------------------------------------------------------


@dataclass(slots=True)
class _Running:
    job: UpdateJob
    future: Future[UpdateResult]
    started: float
    reason: str


JobSource = Callable[[int], UpdateJob]
"""Builds a job with the given id from the session's current observations."""


class UpdateScheduler:
    def __init__(
        self,
        ctx: AppContext,
        backend: UpdateBackend,
        lock: threading.RLock,
        initial: StrategyVersion,
        every: int,
        timeout: float,
    ) -> None:
        if every <= 0 or timeout <= 0:
            raise ValueError("recompute interval and timeout must be positive")
        self.ctx, self.backend, self.lock = ctx, backend, lock
        self.every, self.timeout = every, timeout
        self.active = initial
        self.history: list[StrategyVersion] = [initial]
        self.closed = False
        self.running: _Running | None = None
        self.pending: str | None = None
        self.next_job_id = 1
        self.last_requested_hands = 0
        self.last_success: dict[str, Any] | None = None
        self.last_failure: dict[str, Any] | None = None
        self.counters = dict.fromkeys(
            ("started", "installed", "failed", "timed_out", "cancelled", "discarded_stale"), 0
        )
        self._source: JobSource | None = None
        self._current_generation: Callable[[], int] = lambda: 0
        self._hands_completed: Callable[[], int] = lambda: 0

    def bind(
        self,
        source: JobSource,
        generation: Callable[[], int],
        hands_completed: Callable[[], int],
    ) -> None:
        self._source, self._current_generation = source, generation
        self._hands_completed = hands_completed

    # -- triggers (caller holds the session lock) -------------------------------------------

    def after_hand(self, hands_observed: int, reset: bool) -> None:
        self.poll()
        if reset:
            self.request("change_detected", hands_observed)
        elif hands_observed - self.last_requested_hands >= self.every:
            self.request("scheduled", hands_observed)

    def request(self, reason: str, hands_observed: int) -> None:
        if self.closed:
            return
        self.last_requested_hands = hands_observed
        if self.running is not None:
            self.pending = reason  # coalesce: one follow-up with the newest data
            return
        self._start(reason)

    def poll(self) -> None:
        """Time out an overdue job; its late result will be discarded as stale."""
        r = self.running
        if r is None or time.monotonic() - r.started <= self.timeout:
            return
        r.future.cancel()
        self.running = None
        self.counters["timed_out"] += 1
        self._fail(r.job, f"timed out after {self.timeout:g} s", "timeout")
        self._follow_up()

    def close(self) -> None:
        self.closed = True
        self.pending = None
        if self.running is not None:
            if self.running.future.cancel():
                self.counters["cancelled"] += 1
            self.running = None

    # -- internals ---------------------------------------------------------------------------

    def _start(self, reason: str) -> None:
        assert self._source is not None, "scheduler not bound to a session"
        job = self._source(self.next_job_id)
        self.next_job_id += 1
        self.counters["started"] += 1
        try:
            future = self.backend.submit(job)
        except Exception as exc:  # e.g. the pool is shutting down
            self._fail(job, f"could not start: {exc}", "submit")
            return
        self.running = _Running(job, future, time.monotonic(), reason)
        job_id = job.job_id

        def done(f: Future[UpdateResult]) -> None:
            self._on_done(job_id, f)

        future.add_done_callback(done)

    def _on_done(self, job_id: int, future: Future[UpdateResult]) -> None:
        with self.lock:
            r = self.running
            if r is None or r.job.job_id != job_id:
                if not future.cancelled():
                    self.counters["discarded_stale"] += 1
                return
            self.running = None
            if future.cancelled():
                self.counters["cancelled"] += 1
            else:
                exc = future.exception()
                if exc is not None:
                    self._fail(r.job, f"{type(exc).__name__}: {exc}", "error")
                else:
                    self._install(r, future.result())
            self._follow_up()

    def _follow_up(self) -> None:
        if self.pending is not None and not self.closed:
            reason, self.pending = self.pending, None
            self._start(reason)

    def _install(self, r: _Running, result: UpdateResult) -> None:
        job = r.job
        if self.closed:
            self.counters["discarded_stale"] += 1
            return
        if job.generation != self._current_generation() or job.job_id <= self.active.job_id:
            self.counters["discarded_stale"] += 1
            if self.pending is None:
                self.pending = "stale_result"
            return
        if result.session_id != job.session_id or result.job_id != job.job_id:
            self._fail(job, "result does not belong to this job", "certification")
            return
        try:
            value = certify_policy(self.ctx, result.policy, result.certified_exploitability)
        except CertificationError as exc:
            self._fail(job, str(exc), "certification")
            return
        source = "cfr_fallback" if result.p is None else "rnr"
        version = StrategyVersion(
            version=self.active.version + 1,
            job_id=job.job_id,
            source=source,
            policy=result.policy,
            p=result.p,
            certified_exploitability=value,
            exploitability_limit=self.ctx.budget,
            digest=game_strategy_digest(result.policy, self.ctx.game.action_codes),
            based_on_hands=job.based_on_hands,
            generation=job.generation,
            installed_after_hand=self._hands_completed(),
            installed_at=time.time(),
            estimate=dict(result.estimate),
        )
        self.active = version  # the atomic switch
        self.history.append(version)
        self.counters["installed"] += 1
        self.last_success = {
            "job_id": job.job_id,
            "version": version.version,
            "reason": r.reason,
            "based_on_hands": job.based_on_hands,
            "seconds": result.seconds,
            "finished_at": version.installed_at,
        }

    def _fail(self, job: UpdateJob, message: str, kind: str) -> None:
        if kind != "timeout":
            self.counters["failed"] += 1
        self.last_failure = {
            "job_id": job.job_id,
            "kind": kind,
            "message": message,
            "based_on_hands": job.based_on_hands,
            "at": time.time(),
        }

    def status(self) -> dict[str, Any]:
        r = self.running
        return {
            "running": r is not None,
            "running_job": None
            if r is None
            else {
                "job_id": r.job.job_id,
                "reason": r.reason,
                "based_on_hands": r.job.based_on_hands,
                "elapsed_seconds": time.monotonic() - r.started,
            },
            "pending": self.pending is not None,
            "recompute_every_hands": self.every,
            "next_scheduled_after_hands": self.last_requested_hands + self.every,
            "timeout_seconds": self.timeout,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "counters": dict(self.counters),
        }


__all__ = [
    "InlineBackend",
    "ProcessBackend",
    "StrategyVersion",
    "ThreadBackend",
    "UpdateBackend",
    "UpdateScheduler",
    "baseline_version",
]
