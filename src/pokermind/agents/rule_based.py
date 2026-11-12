"""Deterministic rule-based baselines.

These are intentionally simple, readable reference opponents. They are not
equilibrium strategies; they exist to sanity-check the engine and to serve as
fixed, exploitable opponents in later milestones.
"""

from __future__ import annotations

import numpy as np

from pokermind.games.kuhn import Action, Card, InfoState


def _passive(legal_actions: tuple[Action, ...]) -> Action:
    return Action.CHECK if Action.CHECK in legal_actions else Action.FOLD


def _aggressive(legal_actions: tuple[Action, ...]) -> Action:
    return Action.BET if Action.BET in legal_actions else Action.CALL


class ThresholdAgent:
    """Card-strength thresholds.

    * Not facing a bet: BET with a card ``>= bet_threshold``, else CHECK.
    * Facing a bet: CALL with a card ``>= call_threshold``, else FOLD.

    The defaults (bet only with K, call with Q or K) describe an "honest"
    player who never bluffs.
    """

    def __init__(self, bet_threshold: Card = Card.K, call_threshold: Card = Card.Q) -> None:
        self.bet_threshold = bet_threshold
        self.call_threshold = call_threshold

    @property
    def name(self) -> str:
        return f"threshold(bet>={self.bet_threshold},call>={self.call_threshold})"

    def act(
        self,
        info_state: InfoState,
        legal_actions: tuple[Action, ...],
        rng: np.random.Generator,
    ) -> Action:
        card = info_state.private_card
        if info_state.facing_bet:
            strong = card >= self.call_threshold
        else:
            strong = card >= self.bet_threshold
        return _aggressive(legal_actions) if strong else _passive(legal_actions)


class AlwaysPassAgent:
    """Always checks or folds (maximally passive)."""

    name = "always-pass"

    def act(
        self,
        info_state: InfoState,
        legal_actions: tuple[Action, ...],
        rng: np.random.Generator,
    ) -> Action:
        return _passive(legal_actions)


class AlwaysBetAgent:
    """Always bets or calls (maximally aggressive)."""

    name = "always-bet"

    def act(
        self,
        info_state: InfoState,
        legal_actions: tuple[Action, ...],
        rng: np.random.Generator,
    ) -> Action:
        return _aggressive(legal_actions)
