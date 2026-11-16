"""In-memory session store shared by all requests (M8A, local single-user server)."""

from __future__ import annotations

import secrets
import threading
import uuid
from collections.abc import Callable
from typing import Any, Final

from pokermind.app.context import AppContext
from pokermind.app.session import GameSession, NotFoundError, SessionError, SessionSettings
from pokermind.app.updates import UpdateBackend

MAX_SESSIONS: Final = 32


class SessionManager:
    """Creates, finds and closes sessions; owns the shared update backend.

    Session ids are random UUIDs. Seeds are drawn from :mod:`secrets` and never
    returned to clients, because a seed reveals every future deal. Fixed seeds
    are accepted only when ``allow_client_seed`` is set (tests and simulations).
    """

    def __init__(
        self,
        ctx: AppContext,
        backend: UpdateBackend,
        *,
        allow_client_seed: bool = False,
        max_sessions: int = MAX_SESSIONS,
        session_ids: Callable[[], str] | None = None,
        defaults: SessionSettings | None = None,
    ) -> None:
        self.ctx, self.backend = ctx, backend
        self.defaults = defaults or SessionSettings()
        """Settings for sessions that do not specify their own (operational, not research)."""
        self.allow_client_seed = allow_client_seed
        self.max_sessions = max_sessions
        self._ids = session_ids or (lambda: uuid.uuid4().hex)
        self._sessions: dict[str, GameSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        seed: int | None = None,
        recompute_every: int | None = None,
        update_timeout: float | None = None,
    ) -> GameSession:
        if seed is not None and not self.allow_client_seed:
            raise SessionError("fixed seeds are disabled on this server")
        defaults = self.defaults
        try:
            settings = SessionSettings(
                recompute_every if recompute_every is not None else defaults.recompute_every,
                update_timeout if update_timeout is not None else defaults.update_timeout,
            )
        except ValueError as exc:
            raise SessionError(str(exc)) from exc
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise SessionError(f"at most {self.max_sessions} open sessions")
            session_id = self._ids()
            if session_id in self._sessions:
                raise SessionError("session id collision")
            session = GameSession(
                session_id,
                self.ctx,
                self.backend,
                seed if seed is not None else secrets.randbits(63),
                settings,
            )
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> GameSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise NotFoundError("unknown session")
        return session

    def close(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise NotFoundError("unknown session")
        session.close()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [
            {"session_id": s.id, "hands_played": len(s.hands), "created_at": s.created_at}
            for s in sessions
        ]

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            s.close()
        self.backend.shutdown()


__all__ = ["MAX_SESSIONS", "SessionManager"]
