"""One human playing heads-up Leduc Hold'em against the adaptive AI (M8A).

The server owns everything that decides a hand: the deck, both private cards,
the betting state, the payouts and the AI's observations. The client submits
only an action name and the ``state_version`` it was shown. Every action is
checked against the engine's legal actions (:mod:`pokermind.games.leduc`).

**Seats.** Seats alternate: the human sits in seat ``hand % 2``, as the
opponent did in every M5-M7B study. Seat 0 acts first in both rounds.

**Hidden information.** Public views never contain the AI's card unless the
hand ended in a showdown. The AI acts from its own information-set key, which
is opponent-free. After each hand the AI's opponent model receives
:func:`~pokermind.adaptive.reconstruction.observer_record`. That record
holds:
* the human's public actions;
* the AI's own card and the public card;
* the human's card only if it was shown down.

The AI never learns a folded human card.

**Strategy.** The AI plays the strategy that was active when the hand was
dealt. Recomputed strategies, installed by the
:class:`~pokermind.app.updates.UpdateScheduler`, apply from the next hand on.

**Determinism.** Each hand's deck and AI randomness come from
``SeedSequence((seed, hand))``. With the inline backend, a session is a pure
function of its seed and the human's actions.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from pokermind.adaptive.adapter import M5_MODEL, ChangeAwareObserver
from pokermind.adaptive.reconstruction import observer_record
from pokermind.agents.cfr_agent import sample_action
from pokermind.app.context import AppContext
from pokermind.app.policy_update import UpdateJob
from pokermind.app.updates import StrategyVersion, UpdateBackend, UpdateScheduler, baseline_version
from pokermind.games.leduc import (
    ACTION_NAMES,
    CALL,
    FOLD,
    RAISE,
    LeducState,
    card_name,
)
from pokermind.opponents.bayes import beta_ppf
from pokermind.opponents.context import NUM_LEVELS
from pokermind.simulation.game_runner import sample_chance

ACTION_IDS: Final = {name: i for i, name in enumerate(ACTION_NAMES)}
DEFAULT_RECOMPUTE_EVERY: Final = 50
"""Hands between scheduled recomputations. M7B found that re-using the previous checkpoint's
strategy for the next 20-100 hands costs about 0.01 chips/hand on held-out opponents, so
re-solving in the background every 20-100 hands is enough; 50 is the middle of that range."""
DEFAULT_UPDATE_TIMEOUT: Final = 600.0
INTERVAL_MASS: Final = 0.9
ESTIMATE_DISCLAIMER: Final = (
    "Probabilistic estimates from the AI's Bayesian opponent model. Their calibration was "
    "measured only against simulated development opponents (M7B); they are not calibrated "
    "for human players."
)
TENDENCY_CLASSES: Final = (
    ("R1:cr", "round 1, no bet to face", ("check", "bet")),
    ("R1:fcr", "round 1, facing a bet, may raise", ("fold", "call", "raise")),
    ("R1:fc", "round 1, facing a raise, may not re-raise", ("fold", "call")),
    ("R2:cr", "round 2, no bet to face", ("check", "bet")),
    ("R2:fcr", "round 2, facing a bet, may raise", ("fold", "call", "raise")),
    ("R2:fc", "round 2, facing a raise, may not re-raise", ("fold", "call")),
)
COARSE_LEVEL: Final = 2
"""``context_levels`` index of (round, action set)."""


class SessionError(Exception):
    status = 400


class NotFoundError(SessionError):
    status = 404


class ConflictError(SessionError):
    """The request does not match the current state (stale view, wrong turn, ...)."""

    status = 409


class IllegalActionError(SessionError):
    status = 400


@dataclass(frozen=True, slots=True)
class SessionSettings:
    recompute_every: int = DEFAULT_RECOMPUTE_EVERY
    update_timeout: float = DEFAULT_UPDATE_TIMEOUT

    def __post_init__(self) -> None:
        if not 1 <= self.recompute_every <= 10_000:
            raise ValueError("recompute_every must be between 1 and 10000 hands")
        if self.update_timeout <= 0:
            raise ValueError("update_timeout must be positive")


@dataclass(frozen=True, slots=True)
class ActionEvent:
    actor: str
    """``"human"`` or ``"ai"``."""
    round: int
    action: int
    label: str
    """As the action read at the table: check, bet, call, raise or fold."""


@dataclass(slots=True)
class Hand:
    index: int
    human_seat: int
    strategy: StrategyVersion
    state: LeducState
    rng: np.random.Generator
    """Chance and AI decisions, in order."""
    moves: list[int] = field(default_factory=list)
    events: list[ActionEvent] = field(default_factory=list)
    result: dict[str, Any] | None = None

    @property
    def ai_seat(self) -> int:
        return 1 - self.human_seat

    @property
    def complete(self) -> bool:
        return self.state.is_terminal


def _action_label(state: LeducState, action: int) -> str:
    facing = state.contributions[0] != state.contributions[1]
    if action == CALL:
        return "call" if facing else "check"
    if action == RAISE:
        return "raise" if facing else "bet"
    return "fold"


class GameSession:
    def __init__(
        self,
        session_id: str,
        ctx: AppContext,
        backend: UpdateBackend,
        seed: int,
        settings: SessionSettings | None = None,
    ) -> None:
        self.id, self.ctx, self.seed = session_id, ctx, seed
        self.settings = settings or SessionSettings()
        self.lock = threading.RLock()
        self.created_at = time.time()
        self.closed = False
        self.state_version = 0
        self.hands: list[Hand] = []
        self.current: Hand | None = None
        cd = ctx.research.change_detection
        self.observer = ChangeAwareObserver(ctx.priors.public_prior, ctx.detector, retain=cd.retain)
        self.updates = UpdateScheduler(
            ctx,
            backend,
            self.lock,
            baseline_version(ctx),
            self.settings.recompute_every,
            self.settings.update_timeout,
        )
        self.updates.bind(self._job, lambda: len(self.observer.resets), lambda: len(self.hands))

    # -- gameplay ----------------------------------------------------------------------------

    def deal(self) -> dict[str, Any]:
        with self.lock:
            self._check_open()
            self.updates.poll()
            if self.current is not None and not self.current.complete:
                raise ConflictError("the current hand is not finished")
            index = len(self.hands)
            rng = np.random.default_rng(np.random.SeedSequence((self.seed, index)))
            state = self.ctx.game.initial_state()
            hand = Hand(index, index % 2, self.updates.active, state, rng)
            self.current = hand
            self._advance(hand)
            self.state_version += 1
            return self.view()

    def act(self, action: str, state_version: int) -> dict[str, Any]:
        with self.lock:
            self._check_open()
            self.updates.poll()
            if state_version != self.state_version:
                raise ConflictError(
                    f"stale state_version {state_version}; current is {self.state_version}"
                )
            hand = self.current
            if hand is None or hand.complete:
                raise ConflictError("no hand in progress; deal a new hand")
            if hand.state.current_player != hand.human_seat:
                raise ConflictError("it is not the human's turn")
            if action not in ACTION_IDS:
                raise IllegalActionError(f"unknown action {action!r}")
            move = ACTION_IDS[action]
            legal = hand.state.legal_actions()
            if move not in legal:
                names = [ACTION_NAMES[a] for a in legal]
                raise IllegalActionError(f"{action!r} is not legal now; legal actions: {names}")
            self._apply(hand, move, "human")
            self._advance(hand)
            self.state_version += 1
            return self.view()

    def _apply(self, hand: Hand, move: int, actor: str) -> None:
        label = _action_label(hand.state, move)
        hand.events.append(ActionEvent(actor, hand.state.round_index + 1, move, label))
        hand.moves.append(move)
        hand.state = hand.state.apply(move)

    def _advance(self, hand: Hand) -> None:
        """Deal chance outcomes and play the AI until the human must act or the hand ends."""
        while not hand.state.is_terminal:
            state = hand.state
            if state.is_chance_node:
                card = sample_chance(state, hand.rng)
                hand.moves.append(card)
                hand.state = state.apply(card)
            elif state.current_player == hand.ai_seat:
                key = state.information_state_key(hand.ai_seat)
                legal = state.legal_actions()
                table = hand.strategy.policy[key]
                move = sample_action(legal, [table[a] for a in legal], hand.rng, key)
                self._apply(hand, move, "ai")
            else:
                return
        self._finish(hand)

    def _finish(self, hand: Hand) -> None:
        state = hand.state
        returns = state.returns()
        if returns[0] + returns[1] != 0:
            raise AssertionError("payouts are not zero-sum")
        folded = any(r and r[-1] == FOLD for r in state.rounds)
        winner = state.winner()
        human_net = returns[hand.human_seat]
        hand.result = {
            "ended_by": "fold" if folded else "showdown",
            "winner": None if winner is None else ("human" if winner == hand.human_seat else "ai"),
            "human_net": human_net,
            "ai_net": -human_net,
            "pot": sum(state.contributions),
        }
        self.hands.append(hand)
        resets = len(self.observer.resets)
        record = observer_record(self.ctx.game, tuple(hand.moves), hand.human_seat, hand.index)
        self.observer.observe(record)
        self.updates.after_hand(len(self.observer.records), len(self.observer.resets) > resets)

    def _job(self, job_id: int) -> UpdateJob:
        return UpdateJob(
            session_id=self.id,
            job_id=job_id,
            generation=len(self.observer.resets),
            based_on_hands=len(self.observer.records),
            records=tuple(self.observer.retained()),
            seed_key=(self.seed, job_id, 0x5EED),
        )

    def request_update(self) -> dict[str, Any]:
        with self.lock:
            self._check_open()
            self.updates.poll()
            self.updates.request("manual", len(self.observer.records))
            return self.updates.status()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.updates.close()

    def _check_open(self) -> None:
        if self.closed:
            raise NotFoundError("session closed")

    # -- public views (never contain hidden information) -------------------------------------

    def hand_view(self, hand: Hand) -> dict[str, Any]:
        state = hand.state
        human, ai = hand.human_seat, hand.ai_seat
        contrib = state.contributions
        to_act = None
        legal: list[dict[str, str]] = []
        if not state.is_terminal:
            to_act = "human" if state.current_player == human else "ai"
            if to_act == "human":
                legal = [
                    {"action": ACTION_NAMES[a], "label": _action_label(state, a)}
                    for a in state.legal_actions()
                ]
        view: dict[str, Any] = {
            "hand_number": hand.index + 1,
            "status": "complete" if state.is_terminal else "awaiting_human",
            "human_seat": human,
            "first_to_act": "human" if human == 0 else "ai",
            "human_card": card_name(state.private[human]),
            "public_card": None if state.public is None else card_name(state.public),
            "round": state.round_index + 1,
            "actions": [
                {
                    "actor": e.actor,
                    "round": e.round,
                    "action": ACTION_NAMES[e.action],
                    "label": e.label,
                }
                for e in hand.events
            ],
            "contributions": {"human": contrib[human], "ai": contrib[ai]},
            "pot": contrib[0] + contrib[1],
            "to_act": to_act,
            "legal_actions": legal,
            "strategy_version": hand.strategy.version,
            "result": None,
        }
        if hand.result is not None:
            result = dict(hand.result)
            # the AI's card is public only after a showdown
            result["ai_card"] = (
                card_name(state.private[ai]) if result["ended_by"] == "showdown" else None
            )
            view["result"] = result
        return view

    def view(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.id,
                "state_version": self.state_version,
                "hands_played": len(self.hands),
                "hand": None if self.current is None else self.hand_view(self.current),
                "stats": self.stats(),
                "strategy": self.strategy_summary(),
            }

    def history(self, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        with self.lock:
            if offset < 0 or not 1 <= limit <= 500:
                raise SessionError("need offset >= 0 and 1 <= limit <= 500")
            return {
                "session_id": self.id,
                "total": len(self.hands),
                "offset": offset,
                "hands": [self.hand_view(h) for h in self.hands[offset : offset + limit]],
            }

    def stats(self) -> dict[str, Any]:
        nets = [float(h.result["human_net"]) for h in self.hands if h.result is not None]
        n = len(nets)
        mean = sum(nets) / n if n else 0.0
        se = (
            math.sqrt(sum((x - mean) ** 2 for x in nets) / (n - 1) / n) if n > 1 else None
        )  # standard error of the human's mean result
        results = [h.result for h in self.hands if h.result is not None]
        return {
            "hands_played": n,
            "human_net_chips": sum(nets),
            "ai_net_chips": -sum(nets) + 0.0,  # + 0.0 turns -0.0 into 0.0
            "human_mean_per_hand": mean,
            "human_mean_standard_error": se,
            "human_wins": sum(r["winner"] == "human" for r in results),
            "ai_wins": sum(r["winner"] == "ai" for r in results),
            "splits": sum(r["winner"] is None for r in results),
            "showdowns": sum(r["ended_by"] == "showdown" for r in results),
            "human_folds": sum(r["ended_by"] == "fold" and r["winner"] == "ai" for r in results),
            "ai_folds": sum(r["ended_by"] == "fold" and r["winner"] == "human" for r in results),
        }

    def strategy_summary(self) -> dict[str, Any]:
        a = self.updates.active
        return {
            "active_version": a.version,
            "source": a.source,
            "certified_exploitability": a.certified_exploitability,
            "exploitability_limit": a.exploitability_limit,
            "recomputation_running": self.updates.running is not None,
        }

    def strategy_status(self) -> dict[str, Any]:
        with self.lock:
            self.updates.poll()
            return {
                "session_id": self.id,
                "active": _version_view(self.updates.active),
                "versions": [_version_view(v, brief=True) for v in self.updates.history],
                "updates": self.updates.status(),
            }

    def metrics(self) -> dict[str, Any]:
        """Read-only research metrics, split into exact facts and model estimates."""
        with self.lock:
            self.updates.poll()
            ctx = self.ctx
            retained = self.observer.retained()
            active = self.updates.active
            detector = self.observer.detector
            return {
                "session_id": self.id,
                "hands_played": len(self.hands),
                "verified": {
                    "description": (
                        "Exact quantities: computed by the exact game evaluator or counted "
                        "directly. They involve no estimation."
                    ),
                    "active_strategy": {
                        **_version_view(active, brief=True),
                        "certified_by": "M3 recursive exact evaluator (chips/hand, NashConv/2)",
                    },
                    "exploitability_limit": ctx.budget,
                    "baseline_cfr_exploitability": ctx.cfr_exploitability,
                    "budget_delta": ctx.research.selection.budget_delta,
                    "observation_counts": {
                        "hands_observed": len(self.observer.records),
                        "hands_in_model": len(retained),
                        "human_decisions_in_model": sum(len(r.public.decisions) for r in retained),
                        "showdowns_in_model": sum(
                            r.public.showdown_card is not None for r in retained
                        ),
                        "change_resets_after_hands": [h + 1 for h in self.observer.resets],
                    },
                    "results": self.stats(),
                    "research_config": {
                        "version": ctx.research.version,
                        "digest": ctx.config_digest,
                        "hero_digest": ctx.cfr_digest,
                    },
                },
                "estimates": {
                    "disclaimer": ESTIMATE_DISCLAIMER,
                    "opponent_tendencies": self._tendencies(),
                    "active_strategy_model_estimates": dict(active.estimate) or None,
                    "change_detector": {
                        "statistic": float(detector.statistic),
                        "alarm_threshold": ctx.research.change_detection.threshold,
                    },
                },
                "strategy_updates": self.updates.status(),
            }

    def _tendencies(self) -> list[dict[str, Any]]:
        """Posterior of the human's action frequencies per coarse betting situation.

        Each action also reports the development-population prior mean (what the model
        assumes before seeing this player) and the raw observed frequency (None without
        observations). The prior weighs ``prior_strength`` pseudo-observations, so the
        posterior moves away from it only as the player's own decisions accumulate."""
        model = self.observer.public_model()
        prior, alpha = model.prior, M5_MODEL.alpha_coarse
        assert COARSE_LEVEL < NUM_LEVELS
        out = []
        tail = (1.0 - INTERVAL_MASS) / 2
        for key, description, labels in TENDENCY_CLASSES:
            size = len(labels)
            counts = model.counts[COARSE_LEVEL].get(key)
            n = np.zeros(size) if counts is None else counts
            params = n + alpha * prior.mean(COARSE_LEVEL, key, size)
            a0 = float(params.sum())
            actions = {}
            base = prior.mean(COARSE_LEVEL, key, size)
            total = float(n.sum())
            for i, label in enumerate(labels):
                ai = float(params[i])
                actions[label] = {
                    "posterior_mean": ai / a0,
                    "interval_90": [beta_ppf(tail, ai, a0 - ai), beta_ppf(1 - tail, ai, a0 - ai)],
                    "observed": float(n[i]),
                    "observed_frequency": float(n[i]) / total if total > 0 else None,
                    "population_prior_mean": float(base[i]),
                }
            out.append(
                {
                    "situation": key,
                    "description": description,
                    "observations": total,
                    "prior_strength": float(alpha),
                    "actions": actions,
                }
            )
        return out


def _version_view(v: StrategyVersion, brief: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "version": v.version,
        "source": v.source,
        "rnr_p": v.p,
        "certified_exploitability": v.certified_exploitability,
        "exploitability_limit": v.exploitability_limit,
        "digest": v.digest,
        "based_on_hands": v.based_on_hands,
        "installed_after_hand": v.installed_after_hand,
        "installed_at": v.installed_at,
    }
    if not brief:
        out["model_estimates"] = dict(v.estimate) or None
    return out


__all__ = [
    "DEFAULT_RECOMPUTE_EVERY",
    "ESTIMATE_DISCLAIMER",
    "ConflictError",
    "GameSession",
    "IllegalActionError",
    "NotFoundError",
    "SessionError",
    "SessionSettings",
]
