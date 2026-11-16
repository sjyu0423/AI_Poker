"""M8A: the HTTP contract (FastAPI TestClient; skipped without the [app] extra)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from pokermind.app.api import create_app
from pokermind.app.context import AppContext
from pokermind.app.manager import SessionManager
from pokermind.app.policy_update import UpdateJob, UpdateResult
from pokermind.app.updates import InlineBackend
from pokermind.games.leduc import card_name


def _cfr(ctx: AppContext, job: UpdateJob) -> UpdateResult:
    return UpdateResult(job.session_id, job.job_id, ctx.cfr, None, ctx.cfr_exploitability)


@pytest.fixture
def manager(app_ctx: AppContext) -> SessionManager:
    return SessionManager(app_ctx, InlineBackend(app_ctx, _cfr))


@pytest.fixture
def client(manager: SessionManager) -> Iterator[TestClient]:
    with TestClient(create_app(manager)) as c:
        yield c


def _finish(client: TestClient, sid: str, view: dict[str, Any]) -> dict[str, Any]:
    while view["hand"]["status"] == "awaiting_human":
        r = client.post(
            f"/sessions/{sid}/actions",
            json={"action": "call", "state_version": view["state_version"]},
        )
        assert r.status_code == 200, r.text
        view = r.json()
    return view


def test_health_and_config(client: TestClient, app_ctx: AppContext) -> None:
    assert client.get("/health").json()["status"] == "ok"
    cfg = client.get("/config").json()
    assert cfg["research_digest"] == app_ctx.config_digest
    assert cfg["exploitability_limit"] == pytest.approx(app_ctx.budget)


def test_play_hands_over_http(client: TestClient, manager: SessionManager) -> None:
    r = client.post("/sessions", json={"recompute_every": 3})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    assert r.json()["hand"] is None
    for _ in range(4):
        r = client.post(f"/sessions/{sid}/hands")
        assert r.status_code == 201
        view = _finish(client, sid, r.json())
        assert view["hand"]["status"] == "complete" and view["hand"]["result"] is not None
    state = client.get(f"/sessions/{sid}").json()
    assert state["hands_played"] == 4
    hist = client.get(f"/sessions/{sid}/history", params={"offset": 1, "limit": 2}).json()
    assert hist["total"] == 4 and [h["hand_number"] for h in hist["hands"]] == [2, 3]
    strat = client.get(f"/sessions/{sid}/strategy").json()
    assert strat["active"]["version"] == 2  # one recompute after hand 3
    assert strat["updates"]["counters"]["installed"] == 1
    r = client.post(f"/sessions/{sid}/strategy/recompute")
    assert r.status_code == 202
    assert client.get(f"/sessions/{sid}/strategy").json()["active"]["version"] == 3
    # internal state agrees with what was served
    session = manager.get(sid)
    nets = [h.result["human_net"] for h in session.hands if h.result is not None]
    assert len(nets) == 4 and state["stats"]["human_net_chips"] == sum(nets)


def test_metrics_separate_verified_facts_from_estimates(client: TestClient) -> None:
    sid = client.post("/sessions").json()["session_id"]
    _finish(client, sid, client.post(f"/sessions/{sid}/hands").json())
    m = client.get(f"/sessions/{sid}/metrics").json()
    verified, estimates = m["verified"], m["estimates"]
    assert set(verified) >= {
        "active_strategy",
        "exploitability_limit",
        "observation_counts",
        "results",
        "research_config",
    }
    assert (
        verified["active_strategy"]["certified_exploitability"] <= verified["exploitability_limit"]
    )
    assert "not calibrated for human players" in estimates["disclaimer"]
    tendencies = estimates["opponent_tendencies"]
    assert len(tendencies) == 6
    for t in tendencies:
        for a in t["actions"].values():
            lo, hi = a["interval_90"]
            assert 0 <= lo <= a["posterior_mean"] <= hi <= 1
    up = m["strategy_updates"]
    assert set(up) >= {"running", "last_success", "recompute_every_hands", "counters"}


def test_requests_cannot_inject_state(client: TestClient) -> None:
    r = client.post("/sessions", json={"seed": 1})
    assert r.status_code == 400  # fixed seeds are disabled
    r = client.post("/sessions", json={"ai_card": "K0"})
    assert r.status_code == 422
    sid = client.post("/sessions").json()["session_id"]
    view = client.post(f"/sessions/{sid}/hands").json()
    v = view["state_version"]
    for body in (
        {"action": "call", "state_version": v, "payout": 100},
        {"action": "call", "state_version": v, "human_card": "K0"},
        {"action": "call", "state_version": v, "observations": []},
        {"action": "check", "state_version": v},
        {"action": "call"},
        {"action": 1, "state_version": v},
    ):
        assert client.post(f"/sessions/{sid}/actions", json=body).status_code == 422, body
    r = client.post(f"/sessions/{sid}/actions", json={"action": "call", "state_version": v + 5})
    assert r.status_code == 409
    assert client.post(f"/sessions/{sid}/hands").status_code == 409  # hand in progress
    legal = {a["action"] for a in view["hand"]["legal_actions"]}
    if "fold" not in legal:
        r = client.post(f"/sessions/{sid}/actions", json={"action": "fold", "state_version": v})
        assert r.status_code == 400
    assert client.get(f"/sessions/{sid}").json()["state_version"] == v  # nothing changed


def test_hidden_card_never_served_before_showdown(
    client: TestClient, manager: SessionManager
) -> None:
    sid = client.post("/sessions").json()["session_id"]
    session = manager.get(sid)
    for _ in range(30):
        view = client.post(f"/sessions/{sid}/hands").json()
        hand = session.current
        assert hand is not None
        ai = card_name(hand.state.private[hand.ai_seat])
        while view["hand"]["status"] == "awaiting_human":
            for path in ("", "/metrics", "/strategy"):
                assert ai not in client.get(f"/sessions/{sid}{path}").text
            legal = [a["action"] for a in view["hand"]["legal_actions"]]
            action = "fold" if "fold" in legal else "call"
            r = client.post(
                f"/sessions/{sid}/actions",
                json={"action": action, "state_version": view["state_version"]},
            )
            view = r.json()
        result = view["hand"]["result"]
        if result["ended_by"] == "fold":
            assert result["ai_card"] is None and ai not in json.dumps(view)
        else:
            assert result["ai_card"] == ai


def test_unknown_and_deleted_sessions(client: TestClient) -> None:
    assert client.get("/sessions/nope").status_code == 404
    sid = client.post("/sessions").json()["session_id"]
    assert len(sid) == 32
    assert any(s["session_id"] == sid for s in client.get("/sessions").json())
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404
    assert client.post(f"/sessions/{sid}/hands").status_code == 404
    assert client.delete(f"/sessions/{sid}").status_code == 404
