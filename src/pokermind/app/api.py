"""FastAPI adapter over :class:`~pokermind.app.manager.SessionManager` (M8A).

A thin layer: request validation, routing and error mapping only. Game logic,
legality checks, hidden information and strategy updates live in
:mod:`pokermind.app.session` and :mod:`pokermind.app.updates`. The contract is
documented in ``docs/design/app_backend.md``.

Request bodies forbid unknown fields. A client can submit only an action name
and the ``state_version`` it acted on, never cards, states, payouts or
observations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from pokermind.app.manager import SessionManager
from pokermind.app.session import SessionError

API_VERSION = "1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSession(_Strict):
    recompute_every: int | None = Field(default=None, ge=1, le=10_000)
    seed: int | None = Field(default=None, ge=0, lt=2**63)
    """Accepted only if the server allows client seeds (tests, simulations)."""


class HumanAction(_Strict):
    action: Literal["fold", "call", "raise"]
    state_version: int = Field(ge=0)


def create_app(manager: SessionManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown()

    app = FastAPI(title="PokerMind", version=API_VERSION, lifespan=lifespan)

    @app.exception_handler(SessionError)
    async def _session_error(_: Request, exc: SessionError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "api_version": API_VERSION}

    @app.get("/config")
    def config() -> dict[str, Any]:
        ctx = manager.ctx
        r = ctx.research
        return {
            "game": r.game,
            "research_version": r.version,
            "research_digest": ctx.config_digest,
            "hero_digest": ctx.cfr_digest,
            "baseline_cfr_exploitability": ctx.cfr_exploitability,
            "budget_delta": r.selection.budget_delta,
            "exploitability_limit": ctx.budget,
            "selection_rule": r.selection.rule,
            "rnr_iterations": r.response.iterations,
        }

    @app.post("/sessions", status_code=201)
    def create_session(body: CreateSession | None = None) -> dict[str, Any]:
        body = body or CreateSession()
        session = manager.create(seed=body.seed, recompute_every=body.recompute_every)
        return session.view()

    @app.get("/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        return manager.list()

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return manager.get(session_id).view()

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> None:
        manager.close(session_id)

    @app.post("/sessions/{session_id}/hands", status_code=201)
    def deal(session_id: str) -> dict[str, Any]:
        return manager.get(session_id).deal()

    @app.post("/sessions/{session_id}/actions")
    def act(session_id: str, body: HumanAction) -> dict[str, Any]:
        return manager.get(session_id).act(body.action, body.state_version)

    @app.get("/sessions/{session_id}/history")
    def history(session_id: str, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        return manager.get(session_id).history(offset, limit)

    @app.get("/sessions/{session_id}/metrics")
    def metrics(session_id: str) -> dict[str, Any]:
        return manager.get(session_id).metrics()

    @app.get("/sessions/{session_id}/strategy")
    def strategy(session_id: str) -> dict[str, Any]:
        return manager.get(session_id).strategy_status()

    @app.post("/sessions/{session_id}/strategy/recompute", status_code=202)
    def recompute(session_id: str) -> dict[str, Any]:
        return manager.get(session_id).request_update()

    return app


__all__ = ["API_VERSION", "create_app"]
