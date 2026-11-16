"""M8A: playable sessions, server-side legality, hidden information, payouts, observations."""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

from pokermind.app.context import AppContext, ContextError, load_app_context
from pokermind.app.manager import SessionManager
from pokermind.app.policy_update import UpdateJob, compute_policy_update
from pokermind.app.session import (
    ConflictError,
    GameSession,
    IllegalActionError,
    NotFoundError,
    SessionError,
)
from pokermind.app.updates import InlineBackend
from pokermind.games.leduc import FOLD, card_name, rank
from pokermind.research import DEFAULT_CONFIG_PATH

# -- helpers -----------------------------------------------------------------------------------


def manager(ctx: AppContext, **kwargs: Any) -> SessionManager:
    return SessionManager(ctx, InlineBackend(ctx), allow_client_seed=True, **kwargs)


def play(session: GameSession, hands: int, human_seed: int) -> list[dict[str, Any]]:
    """Play ``hands`` hands with a seeded random human; return every view seen."""
    rnd = random.Random(human_seed)
    views = []
    for _ in range(hands):
        view = session.deal()
        views.append(view)
        while view["hand"]["status"] == "awaiting_human":
            choice = rnd.choice(view["hand"]["legal_actions"])["action"]
            view = session.act(choice, view["state_version"])
            views.append(view)
    return views


def expected_human_net(session: GameSession, index: int) -> float:
    """Leduc settlement from the rules, independently of ``LeducState.returns``."""
    hand = session.hands[index]
    s = hand.state
    human, ai = hand.human_seat, hand.ai_seat
    contrib = s.contributions
    last = hand.events[-1]
    if last.action == FOLD:
        return -contrib[human] if last.actor == "human" else contrib[ai]
    assert s.public is not None
    pub = rank(s.public)

    def strength(seat: int) -> tuple[bool, int]:
        return rank(s.private[seat]) == pub, rank(s.private[seat])

    if strength(human) == strength(ai):
        return 0.0
    return contrib[ai] if strength(human) > strength(ai) else -contrib[human]


# -- context -----------------------------------------------------------------------------------


def test_context_uses_the_frozen_configuration(app_ctx: AppContext) -> None:
    r = app_ctx.research
    assert app_ctx.config_digest == json.loads(DEFAULT_CONFIG_PATH.read_text())["digest"]
    assert app_ctx.cfr_digest == r.hero.digest
    assert app_ctx.budget == pytest.approx(r.hero.exploitability + r.selection.budget_delta)
    assert app_ctx.rule.rule == r.selection.rule and app_ctx.rule.z == r.selection.z


def test_context_priors_match_the_research_world(app_ctx: AppContext) -> None:
    from pokermind.experiments.adaptive_study import build_world

    w = app_ctx.research.world
    world = build_world(
        Path(app_ctx.research.hero.path),
        w.per_family,
        w.out_of_population_per_family,
        w.switching_per_split,
    )
    assert world.priors.population_theta.keys() == app_ctx.priors.population_theta.keys()
    for key, value in world.priors.population_theta.items():
        assert (app_ctx.priors.population_theta[key] == value).all()
    for mine, theirs in zip(
        app_ctx.priors.public_prior.levels, world.priors.public_prior.levels, strict=True
    ):
        assert mine.keys() == theirs.keys()
        assert all((mine[k] == theirs[k]).all() for k in mine)


def test_context_rejects_a_wrong_hero(tmp_path: Path, app_ctx: AppContext) -> None:
    other = tmp_path / "hero.json"
    from pokermind.strategy import save_game_strategy

    table = {k: dict(v) for k, v in app_ctx.cfr.items()}
    key = next(iter(table))
    actions = list(table[key])
    table[key] = {a: 1.0 / len(actions) for a in actions}
    save_game_strategy(other, app_ctx.game.name, app_ctx.game.action_codes, table)
    with pytest.raises(ContextError, match="digest"):
        load_app_context(DEFAULT_CONFIG_PATH, other)
    with pytest.raises(ContextError, match="not found"):
        load_app_context(DEFAULT_CONFIG_PATH, tmp_path / "missing.json")


# -- complete hands and payouts ----------------------------------------------------------------


def test_complete_hands_settle_correctly(app_ctx: AppContext) -> None:
    session = manager(app_ctx).create(seed=11, recompute_every=10_000)
    play(session, 60, human_seed=3)
    assert len(session.hands) == 60
    total = 0.0
    seen = {"fold", "showdown"}
    ended = set()
    for i, hand in enumerate(session.hands):
        assert hand.state.is_terminal and hand.result is not None
        assert hand.human_seat == i % 2
        assert hand.result["human_net"] == expected_human_net(session, i)
        assert hand.result["human_net"] == -hand.result["ai_net"]
        assert hand.result["pot"] == sum(hand.state.contributions)
        replay = app_ctx.game.state_from_moves(tuple(hand.moves))
        assert replay.returns()[hand.human_seat] == hand.result["human_net"]
        total += hand.result["human_net"]
        ended.add(hand.result["ended_by"])
        rounds = {e.round for e in hand.events}
        if hand.result["ended_by"] == "showdown":
            assert rounds == {1, 2} and hand.state.public is not None
    assert ended == seen
    stats = session.stats()
    assert stats["human_net_chips"] == total == -stats["ai_net_chips"]
    assert stats["hands_played"] == 60
    assert stats["human_wins"] + stats["ai_wins"] + stats["splits"] == 60


def test_two_betting_rounds_and_public_card_reveal(app_ctx: AppContext) -> None:
    session = manager(app_ctx).create(seed=5, recompute_every=10_000)
    for _ in range(40):
        view = session.deal()
        assert view["hand"]["public_card"] is None and view["hand"]["round"] == 1
        while view["hand"]["status"] == "awaiting_human":
            view = session.act("call", view["state_version"])  # call is always legal
        hand = view["hand"]
        # a human who only calls never folds: the hand ends at a showdown after both rounds
        # (public card revealed) unless the AI folds
        if hand["result"]["ended_by"] == "showdown":
            assert hand["public_card"] is not None
            assert {a["round"] for a in hand["actions"]} == {1, 2}
        else:
            assert hand["result"]["winner"] == "human"


# -- legality and request validation -----------------------------------------------------------


def test_illegal_and_out_of_turn_requests_are_rejected(app_ctx: AppContext) -> None:
    session = manager(app_ctx).create(seed=2, recompute_every=10_000)
    with pytest.raises(ConflictError):
        session.act("call", 0)  # no hand yet
    view = session.deal()
    with pytest.raises(ConflictError):
        session.deal()  # hand in progress
    version = view["state_version"]
    with pytest.raises(ConflictError):
        session.act("call", version - 1)  # stale view
    with pytest.raises(ConflictError):
        session.act("call", version + 1)
    with pytest.raises(IllegalActionError):
        session.act("check", version)  # not an action name
    legal = {a["action"] for a in view["hand"]["legal_actions"]}
    illegal = (
        ({"fold", "call", "raise"} - legal).pop() if legal != {"fold", "call", "raise"} else None
    )
    before = (session.state_version, list(session.current.moves) if session.current else [])
    if illegal is not None:
        with pytest.raises(IllegalActionError):
            session.act(illegal, version)
        assert (session.state_version, session.current.moves if session.current else []) == before
    while view["hand"]["status"] == "awaiting_human":
        view = session.act("call", view["state_version"])
    with pytest.raises(ConflictError):
        session.act("call", view["state_version"])  # hand over
    session.close()
    with pytest.raises(NotFoundError):
        session.deal()


def test_fold_is_illegal_when_not_facing_a_bet(app_ctx: AppContext) -> None:
    session = manager(app_ctx).create(seed=0, recompute_every=10_000)
    view = session.deal()
    assert view["hand"]["human_seat"] == 0 and view["hand"]["to_act"] == "human"
    assert [a["action"] for a in view["hand"]["legal_actions"]] == ["call", "raise"]
    assert [a["label"] for a in view["hand"]["legal_actions"]] == ["check", "bet"]
    with pytest.raises(IllegalActionError):
        session.act("fold", view["state_version"])


def test_client_seeds_are_refused_by_default(app_ctx: AppContext) -> None:
    m = SessionManager(app_ctx, InlineBackend(app_ctx))
    with pytest.raises(SessionError):
        m.create(seed=1)
    s = m.create()
    assert "seed" not in json.dumps(s.view())
    with pytest.raises(SessionError):
        m.create(recompute_every=0)


# -- hidden information ------------------------------------------------------------------------


def _public_json(session: GameSession) -> str:
    """Every response that can describe the hand in progress. (The history holds only
    completed hands, with independent deals; it is checked separately.)"""
    return json.dumps([session.view(), session.metrics(), session.strategy_status()])


def test_ai_card_is_never_exposed_before_showdown(app_ctx: AppContext) -> None:
    session = manager(app_ctx).create(seed=21, recompute_every=10_000)
    rnd = random.Random(9)
    folds = showdowns = 0
    for _ in range(50):
        view = session.deal()
        hand = session.current
        assert hand is not None
        ai_card = card_name(hand.state.private[hand.ai_seat])
        while view["hand"]["status"] == "awaiting_human":
            assert ai_card not in _public_json(session)
            choice = rnd.choice(view["hand"]["legal_actions"])["action"]
            view = session.act(choice, view["state_version"])
        result = view["hand"]["result"]
        if result["ended_by"] == "fold":
            folds += 1
            assert result["ai_card"] is None
            assert ai_card not in _public_json(session)
            assert ai_card not in json.dumps(session.history(len(session.hands) - 1, 1))
        else:
            showdowns += 1
            assert result["ai_card"] == ai_card
    assert folds and showdowns
    # no folded AI card anywhere in the history
    hist = session.history(0, 500)
    for h, internal in zip(hist["hands"], session.hands, strict=True):
        if h["result"]["ended_by"] == "fold":
            assert h["result"]["ai_card"] is None
            assert card_name(internal.state.private[internal.ai_seat]) not in json.dumps(h)


# -- observation logging -----------------------------------------------------------------------


def test_observations_contain_only_what_the_ai_saw(app_ctx: AppContext) -> None:
    session = manager(app_ctx).create(seed=4, recompute_every=10_000)
    play(session, 40, human_seed=8)
    records = session.observer.records
    assert len(records) == 40
    for hand, record in zip(session.hands, records, strict=True):
        pub = record.public
        assert pub.hand_index == hand.index and pub.opponent_seat == hand.human_seat
        assert record.hero_card == hand.state.private[hand.ai_seat]  # the AI's own card
        assert record.public_card == hand.state.public
        human_actions = [e.action for e in hand.events if e.actor == "human"]
        assert [d.action for d in pub.decisions] == human_actions
        if hand.result is not None and hand.result["ended_by"] == "fold":
            assert pub.showdown_card is None and pub.ended_by_fold
        else:
            assert pub.showdown_card == hand.state.private[hand.human_seat]
    counts = session.metrics()["verified"]["observation_counts"]
    assert counts["hands_observed"] == 40
    assert counts["human_decisions_in_model"] == sum(len(r.public.decisions) for r in records)


# -- determinism and isolation -----------------------------------------------------------------


def _fast(ctx: AppContext) -> AppContext:
    """The frozen context with a short RNR run (test speed only)."""
    r = ctx.research
    research = dataclasses.replace(r, response=dataclasses.replace(r.response, iterations=40))
    return dataclasses.replace(ctx, research=research)


def _transcript(session: GameSession) -> list[Any]:
    return [
        (h.index, h.moves, h.result, h.strategy.version, h.strategy.digest) for h in session.hands
    ]


def test_sessions_are_deterministic_given_seed_and_actions(app_ctx: AppContext) -> None:
    ctx = _fast(app_ctx)
    a = manager(ctx).create(seed=99, recompute_every=8)
    b = manager(ctx).create(seed=99, recompute_every=8)
    play(a, 20, human_seed=5)
    play(b, 20, human_seed=5)
    assert _transcript(a) == _transcript(b)
    assert [v.digest for v in a.updates.history] == [v.digest for v in b.updates.history]
    assert len(a.updates.history) == 3  # baseline + updates after hands 8 and 16
    c = manager(ctx).create(seed=100, recompute_every=8)
    play(c, 20, human_seed=5)
    assert _transcript(c) != _transcript(a)


def test_sessions_are_isolated(app_ctx: AppContext) -> None:
    m = manager(app_ctx)
    a = m.create(seed=1, recompute_every=10_000)
    b = m.create(seed=2, recompute_every=10_000)
    ra, rb = random.Random(1), random.Random(2)
    for _ in range(15):  # interleave every request
        va, vb = a.deal(), b.deal()
        while "awaiting_human" in (va["hand"]["status"], vb["hand"]["status"]):
            if va["hand"]["status"] == "awaiting_human":
                va = a.act(ra.choice(va["hand"]["legal_actions"])["action"], va["state_version"])
            if vb["hand"]["status"] == "awaiting_human":
                vb = b.act(rb.choice(vb["hand"]["legal_actions"])["action"], vb["state_version"])
    solo = manager(app_ctx).create(seed=1, recompute_every=10_000)
    play(solo, 15, human_seed=1)
    assert _transcript(a) == _transcript(solo)
    assert a.observer.records is not b.observer.records
    assert len(a.observer.records) == len(b.observer.records) == 15
    with pytest.raises(ConflictError):
        b.act("call", a.state_version + 1000)
    m.close(a.id)
    with pytest.raises(NotFoundError):
        m.get(a.id)
    assert m.get(b.id) is b


# -- the frozen procedure ----------------------------------------------------------------------


def test_recompute_reproduces_the_frozen_m7b_procedure(app_ctx: AppContext) -> None:
    """Same observations and posterior seed: the app picks what ``reliability evaluate`` picked."""
    import pokermind.experiments.reliability_study as rs
    from pokermind.adaptive.adapter import Adapter
    from pokermind.experiments.adaptive_study import build_world, observe
    from pokermind.experiments.robust_study import M6_SETTINGS
    from pokermind.robust.selection import Selector

    world = build_world(Path(app_ctx.research.hero.path), 4, 2, 8)
    selector = Selector(
        world.game, world.specs, app_ctx.tree, world.cfr, world.priors.cfr_exploitability
    )
    adapter = Adapter(world.game, world.specs, world.priors, M6_SETTINGS)
    opponent = world.opponents[world.groups["heldout"][1]]
    records = observe(world, opponent, 0, 30)
    key = (12345, 30)
    rec = rs.reconstruct(world, adapter, records)
    config = rs.ReliabilityConfig(Path("x"), hands=(30,))
    row = rs._snapshot_decisions(
        world, selector, rec, app_ctx.research, config, [opponent.policies[0]], key, False
    )[0]
    expected = row["frozen"][str(app_ctx.research.selection.budget_delta)]
    job = UpdateJob("s", 1, 0, 30, tuple(records), key)
    got = compute_policy_update(app_ctx, job)
    assert got.p == expected["p"]
    assert got.certified_exploitability == expected["exploitability"]
    assert got.certified_exploitability <= app_ctx.budget
