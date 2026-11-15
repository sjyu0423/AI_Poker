"""Seeded hands between a fixed hero and a simulated opponent (M5).

Hero and opponent alternate seats: the opponent sits in seat ``hand % 2``.
Each hand uses its own ``SeedSequence((stream, hand))`` split into chance,
hero and opponent generators, so hand ``i`` of a stream is the same whatever
was played before, and estimation and evaluation streams are independent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pokermind.agents.cfr_agent import sample_action
from pokermind.games.leduc import LeducGame
from pokermind.opponents.context import PublicHandRecord, public_record
from pokermind.simulation.game_runner import sample_chance
from pokermind.strategy import GameStrategy

PolicyAt = Callable[[int], GameStrategy]


def play_hand(
    game: LeducGame, hero: GameStrategy, opponent: GameStrategy, hand: int, stream: int
) -> tuple[int, ...]:
    """The move sequence (chance outcomes and actions) of one hand."""
    chance_ss, hero_ss, opp_ss = np.random.SeedSequence((stream, hand)).spawn(3)
    chance_rng = np.random.default_rng(chance_ss)
    rngs = {"hero": np.random.default_rng(hero_ss), "opp": np.random.default_rng(opp_ss)}
    opp_seat = hand % 2
    state = game.initial_state()
    moves = []
    while not state.is_terminal:
        if state.is_chance_node:
            move = sample_chance(state, chance_rng)
        else:
            seat = state.current_player
            key = state.information_state_key(seat)
            who = "opp" if seat == opp_seat else "hero"
            table = opponent if who == "opp" else hero
            legal = state.legal_actions()
            move = sample_action(legal, [table[key][a] for a in legal], rngs[who], key)
        moves.append(move)
        state = state.apply(move)
    return tuple(moves)


def play_records(
    game: LeducGame,
    hero: GameStrategy,
    opponent_at: PolicyAt,
    num_hands: int,
    stream: int,
    start: int = 0,
) -> list[PublicHandRecord]:
    """Public records of hands ``start .. start + num_hands - 1`` of ``stream``."""
    records = []
    for hand in range(start, start + num_hands):
        moves = play_hand(game, hero, opponent_at(hand), hand, stream)
        records.append(public_record(game, moves, hand % 2, hand))
    return records
