"""Rules, transitions and payoffs for every legal path of Kuhn Poker."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from pokermind.games.kuhn import (
    TERMINAL_HISTORIES,
    Action,
    Card,
    History,
    IllegalActionError,
    KuhnConfig,
    KuhnState,
    all_deals,
    deal,
    history_key,
    initial_state,
)

X, B, C, F = Action.CHECK, Action.BET, Action.CALL, Action.FOLD
J, Q, K = Card.J, Card.Q, Card.K


def walk(state: KuhnState) -> Iterator[KuhnState]:
    """Yield every state reachable from ``state`` (including itself)."""
    yield state
    for action in state.legal_actions():
        yield from walk(state.apply(action))


# -- structure -------------------------------------------------------------


def test_six_distinct_deals() -> None:
    deals = all_deals()
    assert len(deals) == 6
    assert len(set(deals)) == 6
    assert all(a != b for a, b in deals)


EXPECTED_LEGAL: dict[History, tuple[Action, ...]] = {
    (): (X, B),
    (X,): (X, B),
    (B,): (F, C),
    (X, B): (F, C),
}


@pytest.mark.parametrize("history", list(EXPECTED_LEGAL))
def test_legal_actions(history: History) -> None:
    state = KuhnState((K, J), history)
    assert not state.is_terminal
    assert state.legal_actions() == EXPECTED_LEGAL[history]
    assert state.current_player == len(history) % 2


def test_game_tree_is_exactly_as_specified() -> None:
    for cards in all_deals():
        states = list(walk(initial_state(cards)))
        histories = {s.history for s in states}
        assert histories == set(EXPECTED_LEGAL) | TERMINAL_HISTORIES
        terminals = {s.history for s in states if s.is_terminal}
        assert terminals == TERMINAL_HISTORIES
        assert len(terminals) == 5
        assert all(len(h) <= 3 for h in histories)


TERMINALS: list[History] = sorted(TERMINAL_HISTORIES, key=lambda h: history_key(h))


@pytest.mark.parametrize("history", TERMINALS)
def test_terminal_states_have_no_actions(history: History) -> None:
    state = KuhnState((Q, K), history)
    assert state.is_terminal
    assert state.legal_actions() == ()
    with pytest.raises(ValueError):
        _ = state.current_player
    for action in Action:
        with pytest.raises(IllegalActionError):
            state.apply(action)


@pytest.mark.parametrize("history", list(EXPECTED_LEGAL))
def test_illegal_actions_rejected(history: History) -> None:
    state = KuhnState((J, Q), history)
    for action in Action:
        if action in EXPECTED_LEGAL[history]:
            assert state.apply(action).history == (*history, action)
        else:
            assert not state.is_legal(action)
            with pytest.raises(IllegalActionError):
                state.apply(action)


@pytest.mark.parametrize("bad", ["check", "x", None, 0, 1])
def test_non_action_values_rejected(bad: object) -> None:
    state = initial_state((J, Q))
    assert not state.is_legal(bad)
    with pytest.raises(IllegalActionError):
        state.apply(bad)  # type: ignore[arg-type]


def test_apply_does_not_mutate() -> None:
    state = initial_state((K, Q))
    child = state.apply(B)
    assert state.history == ()
    assert child.history == (B,)
    assert child.cards == state.cards
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.history = (X,)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        KuhnConfig().ante = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("cards", "history"),
    [
        ((K, K), ()),
        ((K,), ()),
        ((K, Q), (C,)),
        ((K, Q), (X, X, X)),
        ((K, Q), (B, B)),
        ((K, Q), (X, F)),
    ],
)
def test_invalid_states_rejected(cards: tuple[Card, ...], history: History) -> None:
    with pytest.raises(ValueError):
        KuhnState(cards, history)  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"ante": 0}, {"bet_size": -1}, {"ante": 1.5}, {"ante": True}])
def test_invalid_config_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        KuhnConfig(**kwargs)  # type: ignore[arg-type]


# -- payoffs ---------------------------------------------------------------

# Player 0's net payoff with ante=1, bet=1, written out by hand from the rules.
# Showdown outcomes depend on whether player 0 holds the higher card.
P0_PAYOFF: dict[History, Callable[[bool], int]] = {
    (X, X): lambda p0_wins: 1 if p0_wins else -1,
    (B, F): lambda p0_wins: 1,
    (B, C): lambda p0_wins: 2 if p0_wins else -2,
    (X, B, F): lambda p0_wins: -1,
    (X, B, C): lambda p0_wins: 2 if p0_wins else -2,
}


@pytest.mark.parametrize("cards", all_deals(), ids=lambda c: f"{c[0]}{c[1]}")
@pytest.mark.parametrize("history", list(P0_PAYOFF), ids=history_key)
def test_payoffs_every_terminal(cards: tuple[Card, Card], history: History) -> None:
    state = initial_state(cards)
    for action in history:
        state = state.apply(action)
    assert state.is_terminal
    expected = P0_PAYOFF[history](cards[0] > cards[1])
    assert state.payoffs() == (expected, -expected)
    assert sum(state.payoffs()) == 0


def test_payoff_table_spot_checks() -> None:
    assert KuhnState((K, J), (X, X)).payoffs() == (1, -1)
    assert KuhnState((J, K), (B, C)).payoffs() == (-2, 2)
    assert KuhnState((J, K), (B, F)).payoffs() == (1, -1)  # successful bluff
    assert KuhnState((K, Q), (X, B, F)).payoffs() == (-1, 1)
    assert KuhnState((Q, J), (X, B, C)).payoffs() == (2, -2)


def test_contributions_and_pot() -> None:
    assert initial_state((J, Q)).contributions() == (1, 1)
    assert KuhnState((J, Q), (B,)).contributions() == (2, 1)
    assert KuhnState((J, Q), (X, B)).contributions() == (1, 2)
    assert KuhnState((J, Q), (X, B, C)).contributions() == (2, 2)
    assert KuhnState((J, Q), (X, B, C)).pot == 4


@pytest.mark.parametrize(("ante", "bet"), [(1, 1), (2, 3), (5, 10)])
def test_payoffs_scale_with_config(ante: int, bet: int) -> None:
    cfg = KuhnConfig(ante=ante, bet_size=bet)
    assert KuhnState((K, J), (X, X), cfg).payoffs() == (ante, -ante)
    assert KuhnState((K, J), (B, F), cfg).payoffs() == (ante, -ante)
    assert KuhnState((K, J), (B, C), cfg).payoffs() == (ante + bet, -(ante + bet))
    assert KuhnState((J, K), (X, B, C), cfg).payoffs() == (-(ante + bet), ante + bet)
    for cards in all_deals():
        for s in walk(initial_state(cards, cfg)):
            if s.is_terminal:
                assert sum(s.payoffs()) == 0


def test_winner_and_payoffs_undefined_before_terminal() -> None:
    state = KuhnState((K, J), (B,))
    with pytest.raises(ValueError):
        state.winner()
    with pytest.raises(ValueError):
        state.payoffs()


# -- dealing ---------------------------------------------------------------


def test_deal_is_deterministic_for_seed() -> None:
    a = [deal(np.random.default_rng(123)).cards for _ in range(1)]
    b = [deal(np.random.default_rng(123)).cards for _ in range(1)]
    assert a == b
    rng1, rng2 = np.random.default_rng(9), np.random.default_rng(9)
    assert [deal(rng1).cards for _ in range(50)] == [deal(rng2).cards for _ in range(50)]


def test_deal_covers_all_deals_roughly_uniformly() -> None:
    rng = np.random.default_rng(0)
    n = 60_000
    counts: dict[tuple[Card, Card], int] = {}
    for _ in range(n):
        cards = deal(rng).cards
        counts[cards] = counts.get(cards, 0) + 1
    assert set(counts) == set(all_deals())
    for c in counts.values():
        assert abs(c - n / 6) < 5 * (n * (1 / 6) * (5 / 6)) ** 0.5
