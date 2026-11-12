"""Information leakage: agents must never be able to see the opponent's card."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np

from pokermind.games.kuhn import (
    DECK,
    Action,
    Card,
    InfoState,
    KuhnState,
    all_deals,
    initial_state,
)
from pokermind.simulation import SimulationConfig, run_simulation


def walk(state: KuhnState) -> Iterator[KuhnState]:
    yield state
    for action in state.legal_actions():
        yield from walk(state.apply(action))


def test_info_state_has_only_permitted_fields() -> None:
    names = {f.name for f in dataclasses.fields(InfoState)}
    assert names == {"player", "private_card", "history"}


def test_info_state_contents() -> None:
    state = KuhnState((Card.Q, Card.K), (Action.CHECK, Action.BET))
    for player in (0, 1):
        info = state.information_state(player)
        assert info.player == player
        assert info.private_card == state.cards[player]
        assert info.history == state.history
    assert state.information_state(0).key == "Q|xb"
    assert state.information_state(1).key == "K|xb"


def test_info_state_key_identifies_seat_and_view() -> None:
    """The key omits ``player``; check that it is still unambiguous: each key
    maps to exactly one (player, card, history) and there are 12 of them."""
    by_key: dict[str, set[tuple[int, Card, tuple[Action, ...]]]] = {}
    for cards in all_deals():
        for state in walk(initial_state(cards)):
            if state.is_terminal:
                continue
            p = state.current_player
            info = state.information_state(p)
            by_key.setdefault(info.key, set()).add((info.player, info.private_card, info.history))
    assert len(by_key) == 12
    assert all(len(views) == 1 for views in by_key.values())


def test_info_state_independent_of_opponent_card() -> None:
    """For every reachable state and player, changing the opponent's card
    (to any other card consistent with the player's own) must not change the
    information state in any way, including its key and repr."""
    checked = 0
    for cards in all_deals():
        for state in walk(initial_state(cards)):
            for player in (0, 1):
                own = state.cards[player]
                views = set()
                for opp in DECK:
                    if opp == own:
                        continue
                    alt_cards = (own, opp) if player == 0 else (opp, own)
                    alt = KuhnState(alt_cards, state.history, state.config)
                    info = alt.information_state(player)
                    views.add((info, info.key, repr(info)))
                assert len(views) == 1
                checked += 1
    assert checked == 6 * 9 * 2


class SpyAgent:
    """Records every argument it receives; plays randomly."""

    name = "spy"

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def act(
        self,
        info_state: InfoState,
        legal_actions: tuple[Action, ...],
        rng: np.random.Generator,
    ) -> Action:
        self.calls.append((info_state, legal_actions, rng))
        return legal_actions[int(rng.integers(len(legal_actions)))]


def test_runner_only_passes_info_states_to_agents() -> None:
    spies = (SpyAgent(), SpyAgent())
    run_simulation(spies, SimulationConfig(num_hands=2_000, seed=5))
    for spy in spies:
        assert spy.calls
        for info, legal, rng in spy.calls:
            assert type(info) is InfoState
            assert not isinstance(info, KuhnState)
            assert isinstance(legal, tuple)
            assert all(isinstance(a, Action) for a in legal)
            assert isinstance(rng, np.random.Generator)
