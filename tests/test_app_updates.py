"""M8A: background strategy recomputation, certification and stale-result protection."""

from __future__ import annotations

import dataclasses
import random
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from pokermind.app.context import AppContext
from pokermind.app.manager import SessionManager
from pokermind.app.policy_update import (
    CertificationError,
    UpdateJob,
    UpdateResult,
    certify_policy,
)
from pokermind.app.session import GameSession
from pokermind.app.updates import (
    InlineBackend,
    ProcessBackend,
    ThreadBackend,
    UpdateBackend,
)
from pokermind.research import DEFAULT_CONFIG_PATH
from pokermind.strategy import GameStrategy, game_strategy_digest

Compute = Callable[[AppContext, UpdateJob], UpdateResult]


def cfr_result(ctx: AppContext, job: UpdateJob, **kw: Any) -> UpdateResult:
    """A valid result: the certified CFR baseline (as the frozen rule's fallback)."""
    fields: dict[str, Any] = {
        "session_id": job.session_id,
        "job_id": job.job_id,
        "policy": {k: dict(v) for k, v in ctx.cfr.items()},
        "p": None,
        "certified_exploitability": ctx.cfr_exploitability,
    }
    fields.update(kw)
    return UpdateResult(**fields)


def session_with(
    ctx: AppContext, backend: UpdateBackend, every: int, timeout: float = 600.0
) -> GameSession:
    m = SessionManager(ctx, backend, allow_client_seed=True)
    return m.create(seed=3, recompute_every=every, update_timeout=timeout)


def play_hand(session: GameSession, rnd: random.Random) -> None:
    view = session.deal()
    while view["hand"]["status"] == "awaiting_human":
        view = session.act(
            rnd.choice(view["hand"]["legal_actions"])["action"], view["state_version"]
        )


def wait_until(predicate: Callable[[], bool], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


class Gated:
    """A compute function whose job ``n`` blocks until ``release(n)``."""

    def __init__(self, result: Compute = cfr_result) -> None:
        self.result = result
        self.gates: dict[int, threading.Event] = {}
        self.started: list[UpdateJob] = []
        self.lock = threading.Lock()

    def gate(self, job_id: int) -> threading.Event:
        with self.lock:
            return self.gates.setdefault(job_id, threading.Event())

    def release(self, job_id: int) -> None:
        self.gate(job_id).set()

    def __call__(self, ctx: AppContext, job: UpdateJob) -> UpdateResult:
        with self.lock:
            self.started.append(job)
        if not self.gate(job.job_id).wait(30):
            raise TimeoutError("test gate never released")
        return self.result(ctx, job)


# -- scheduling --------------------------------------------------------------------------------


def test_recompute_is_scheduled_every_n_hands(app_ctx: AppContext) -> None:
    jobs: list[UpdateJob] = []

    def compute(ctx: AppContext, job: UpdateJob) -> UpdateResult:
        jobs.append(job)
        return cfr_result(ctx, job)

    session = session_with(app_ctx, InlineBackend(app_ctx, compute), every=3)
    rnd = random.Random(0)
    for _ in range(7):
        play_hand(session, rnd)
    assert [j.based_on_hands for j in jobs] == [3, 6]
    assert [j.job_id for j in jobs] == [1, 2]
    assert [v.version for v in session.updates.history] == [1, 2, 3]
    # strategies apply from the next hand on
    assert [h.strategy.version for h in session.hands] == [1, 1, 1, 2, 2, 2, 3]
    # the job contains only observer records (what the AI saw)
    assert len(jobs[-1].records) == 6
    status = session.strategy_status()["updates"]
    assert status["counters"]["installed"] == 2 and not status["running"]
    assert status["next_scheduled_after_hands"] == 9


def test_change_detection_requests_an_immediate_recompute(app_ctx: AppContext) -> None:
    jobs: list[UpdateJob] = []

    def compute(ctx: AppContext, job: UpdateJob) -> UpdateResult:
        jobs.append(job)
        return cfr_result(ctx, job)

    session = session_with(app_ctx, InlineBackend(app_ctx, compute), every=1000)
    with session.lock:
        session.updates.after_hand(4, reset=False)
        assert jobs == []
        session.updates.after_hand(5, reset=True)
    assert len(jobs) == 1 and session.updates.last_success is not None
    assert session.updates.last_success["reason"] == "change_detected"


# -- failures ----------------------------------------------------------------------------------


def test_solver_failure_keeps_the_verified_strategy_and_recovers(app_ctx: AppContext) -> None:
    calls = {"n": 0}

    def flaky(ctx: AppContext, job: UpdateJob) -> UpdateResult:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("solver crashed")
        return cfr_result(ctx, job)

    session = session_with(app_ctx, InlineBackend(app_ctx, flaky), every=2)
    rnd = random.Random(1)
    for _ in range(2):
        play_hand(session, rnd)
    up = session.updates
    assert up.active.version == 1 and up.active.source == "baseline_cfr"
    assert up.last_failure is not None and up.last_failure["kind"] == "error"
    assert "solver crashed" in up.last_failure["message"]
    play_hand(session, rnd)  # play continues on the verified baseline
    play_hand(session, rnd)
    assert up.active.version == 2 and up.counters["failed"] == 1
    assert up.counters["installed"] == 1


def _uniform(ctx: AppContext) -> GameStrategy:
    return {k: {a: 1.0 / len(s.actions) for a in s.actions} for k, s in ctx.specs.items()}


@pytest.mark.parametrize("case", ["over_budget", "missing_infoset", "bad_claim", "wrong_job"])
def test_uncertifiable_results_are_rejected(app_ctx: AppContext, case: str) -> None:
    def bad(ctx: AppContext, job: UpdateJob) -> UpdateResult:
        if case == "over_budget":
            return cfr_result(ctx, job, policy=_uniform(ctx), p=0.5, certified_exploitability=0.0)
        if case == "missing_infoset":
            policy = {k: dict(v) for k, v in ctx.cfr.items()}
            policy.pop(next(iter(policy)))
            return cfr_result(ctx, job, policy=policy)
        if case == "bad_claim":
            return cfr_result(ctx, job, certified_exploitability=ctx.cfr_exploitability + 0.01)
        return cfr_result(ctx, job, job_id=job.job_id + 7)

    session = session_with(app_ctx, InlineBackend(app_ctx, bad), every=1)
    play_hand(session, random.Random(2))
    up = session.updates
    assert up.active.version == 1 and len(up.history) == 1
    assert up.last_failure is not None and up.last_failure["kind"] == "certification"
    assert up.counters["failed"] == 1 and up.counters["installed"] == 0


def test_certify_policy_checks_budget_and_validity(app_ctx: AppContext) -> None:
    assert certify_policy(app_ctx, app_ctx.cfr) == pytest.approx(app_ctx.cfr_exploitability)
    with pytest.raises(CertificationError, match="exceeds"):
        certify_policy(app_ctx, _uniform(app_ctx))
    broken = {k: dict(v) for k, v in app_ctx.cfr.items()}
    key = next(iter(broken))
    broken[key] = dict.fromkeys(broken[key], 0.9)
    with pytest.raises(CertificationError, match="invalid"):
        certify_policy(app_ctx, broken)


# -- background execution, overlap and staleness -----------------------------------------------


def test_gameplay_continues_during_an_update_and_install_is_atomic(app_ctx: AppContext) -> None:
    gated = Gated()
    backend = ThreadBackend(app_ctx, gated)
    try:
        session = session_with(app_ctx, backend, every=2)
        rnd = random.Random(4)
        play_hand(session, rnd)
        play_hand(session, rnd)  # triggers job 1 (blocked)
        wait_until(lambda: len(gated.started) == 1)
        assert session.strategy_status()["updates"]["running"]
        t0 = time.perf_counter()
        for _ in range(3):  # play continues on version 1; a request after hand 4 is coalesced
            play_hand(session, rnd)
        assert time.perf_counter() - t0 < 5
        assert all(h.strategy.version == 1 for h in session.hands)
        assert session.updates.pending is not None
        view = session.deal()  # the human always acts at least once in a Leduc hand
        assert view["hand"]["status"] == "awaiting_human"
        in_progress = session.current
        assert in_progress is not None
        gated.release(1)
        wait_until(lambda: session.updates.counters["installed"] == 1)
        assert session.updates.active.version == 2
        assert in_progress.strategy.version == 1  # the dealt hand keeps its strategy
        # exactly one follow-up job, built from the newest observations
        wait_until(lambda: len(gated.started) == 2)
        assert gated.started[1].based_on_hands == len(session.observer.records)
        assert session.updates.pending is None
        while view["hand"]["status"] == "awaiting_human":
            view = session.act(
                rnd.choice(view["hand"]["legal_actions"])["action"], view["state_version"]
            )
        assert session.deal()["hand"]["strategy_version"] == 2
        gated.release(2)
        wait_until(lambda: session.updates.counters["installed"] == 2)
    finally:
        for job_id in range(1, 10):
            gated.release(job_id)
        backend.shutdown()


def test_timed_out_result_is_never_installed(app_ctx: AppContext) -> None:
    gated = Gated()
    backend = ThreadBackend(app_ctx, gated, workers=2)
    try:
        session = session_with(app_ctx, backend, every=1000, timeout=0.05)
        session.request_update()  # job 1
        wait_until(lambda: len(gated.started) == 1)
        time.sleep(0.1)
        metrics = session.metrics()  # reading polls the deadline
        up = metrics["strategy_updates"]
        assert up["counters"]["timed_out"] == 1 and not up["running"]
        assert up["last_failure"]["kind"] == "timeout"
        session.updates.timeout = 600.0
        session.request_update()  # job 2
        wait_until(lambda: len(gated.started) == 2)
        gated.release(1)  # the late result of job 1
        wait_until(lambda: session.updates.counters["discarded_stale"] == 1)
        assert session.updates.active.version == 1
        gated.release(2)
        wait_until(lambda: session.updates.counters["installed"] == 1)
        assert session.updates.active.job_id == 2
    finally:
        for job_id in range(1, 10):
            gated.release(job_id)
        backend.shutdown()


def test_result_from_before_a_change_reset_is_discarded(app_ctx: AppContext) -> None:
    gated = Gated()
    backend = ThreadBackend(app_ctx, gated)
    try:
        session = session_with(app_ctx, backend, every=1000)
        play_hand(session, random.Random(5))
        session.request_update()
        wait_until(lambda: len(gated.started) == 1)
        with session.lock:
            session.observer.resets.append(0)  # a change was detected meanwhile
        gated.release(1)
        wait_until(lambda: len(gated.started) == 2)
        up = session.updates
        assert up.counters["discarded_stale"] == 1 and up.active.version == 1
        assert gated.started[1].generation == 1
        gated.release(2)
        wait_until(lambda: up.counters["installed"] == 1)
        assert up.active.generation == 1 and up.active.job_id == 2
    finally:
        for job_id in range(1, 10):
            gated.release(job_id)
        backend.shutdown()


def test_closing_a_session_discards_its_running_update(app_ctx: AppContext) -> None:
    gated = Gated()
    backend = ThreadBackend(app_ctx, gated)
    try:
        session = session_with(app_ctx, backend, every=1000)
        session.request_update()
        wait_until(lambda: len(gated.started) == 1)
        session.close()
        gated.release(1)
        time.sleep(0.2)
        assert session.updates.active.version == 1
        assert session.updates.counters["installed"] == 0
        assert len(gated.started) == 1  # no follow-up after close
    finally:
        for job_id in range(1, 10):
            gated.release(job_id)
        backend.shutdown()


# -- real recomputation ------------------------------------------------------------------------


def _fast(ctx: AppContext) -> AppContext:
    r = ctx.research
    research = dataclasses.replace(r, response=dataclasses.replace(r.response, iterations=40))
    return dataclasses.replace(ctx, research=research)


def test_every_active_strategy_is_certified(app_ctx: AppContext) -> None:
    ctx = _fast(app_ctx)
    session = session_with(ctx, InlineBackend(ctx), every=6)
    rnd = random.Random(6)
    for _ in range(18):
        play_hand(session, rnd)
    history = session.updates.history
    assert len(history) == 4
    for v in history:
        assert v.certified_exploitability <= v.exploitability_limit == ctx.budget
        assert certify_policy(ctx, v.policy) == pytest.approx(v.certified_exploitability, abs=1e-12)
        assert v.digest == game_strategy_digest(v.policy, ctx.game.action_codes)
    for h in session.hands:  # every hand was played by a certified version
        assert h.strategy in history


def test_process_backend_runs_the_frozen_update(app_ctx: AppContext) -> None:
    backend = ProcessBackend(DEFAULT_CONFIG_PATH, None, workers=1)
    try:
        session = session_with(app_ctx, backend, every=1000)
        rnd = random.Random(7)
        for _ in range(12):
            play_hand(session, rnd)
        session.request_update()
        wait_until(lambda: not session.strategy_status()["updates"]["running"], timeout=180)
        up = session.updates
        assert up.counters["installed"] == 1, up.last_failure
        assert up.active.certified_exploitability <= app_ctx.budget
        assert up.active.based_on_hands == 12
    finally:
        backend.shutdown()
