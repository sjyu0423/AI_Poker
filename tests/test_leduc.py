"""Leduc Hold'em engine: rules, chance, payoffs, information hiding, perfect recall."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import pytest

from pokermind.evaluation.game_exact import (
    exploitability_report,
    information_sets,
    tree_counts,
    uniform_game_strategy,
)
from pokermind.games.base import CHANCE, TERMINAL, Game, GameState
from pokermind.games.leduc import (
    CALL,
    FOLD,
    NUM_CARDS,
    RAISE,
    LeducConfig,
    LeducGame,
    LeducObservation,
    LeducState,
    card_name,
    rank,
)

F, C, R = FOLD, CALL, RAISE
GAME = LeducGame()


def walk(state: GameState) -> Iterator[GameState]:
    yield state
    if state.is_terminal:
        return
    moves = (
        [o for o, _ in state.chance_outcomes()] if state.is_chance_node else state.legal_actions()
    )
    for move in moves:
        yield from walk(state.apply(move))


def all_states(game: Game = GAME) -> list[LeducState]:
    return [s for s in walk(game.initial_state()) if isinstance(s, LeducState)]


STATES = all_states()


def play(*moves: int) -> LeducState:
    return GAME.state_from_moves(moves)


# -- cards and structure -------------------------------------------------------------------


def test_cards() -> None:
    assert [rank(c) for c in range(NUM_CARDS)] == [0, 0, 1, 1, 2, 2]
    assert [card_name(c) for c in range(NUM_CARDS)] == ["J0", "J1", "Q0", "Q1", "K0", "K1"]


def test_reference_tree_counts_and_information_sets() -> None:
    """M2B reference values measured from OpenSpiel's leduc_poker."""
    counts = tree_counts(GAME)
    assert (counts.chance, counts.decision, counts.terminal) == (157, 3_780, 5_520)
    specs = information_sets(GAME)
    assert len(specs) == 936
    assert sum(s.player == 0 for s in specs.values()) == 468


def test_uniform_nash_conv_reference_value() -> None:
    report = exploitability_report(GAME, uniform_game_strategy(GAME))
    assert report.nash_conv == pytest.approx(4.747222222222222, abs=1e-12)


# -- chance ------------------------------------------------------------------------------


def test_conditional_chance_probabilities() -> None:
    root = GAME.initial_state()
    assert root.current_player == CHANCE
    assert root.chance_outcomes() == tuple((c, 1 / 6) for c in range(6))
    after_p0 = root.apply(4)
    assert after_p0.chance_outcomes() == tuple((c, 1 / 5) for c in (0, 1, 2, 3, 5))
    after_round1 = play(4, 0, C, C)
    assert after_round1.is_chance_node
    assert after_round1.chance_outcomes() == tuple((c, 1 / 4) for c in (1, 2, 3, 5))
    for state in STATES:
        if state.is_chance_node:
            outcomes = state.chance_outcomes()
            assert sum(p for _, p in outcomes) == pytest.approx(1.0)
            used = set(state.private) | ({state.public} if state.public is not None else set())
            assert not used & {o for o, _ in outcomes}


def test_exact_deal_distribution() -> None:
    """Every ordered (P0, P1, public) triple has probability exactly 1/120."""
    probs: dict[tuple[int, int, int], Fraction] = {}
    root = GAME.initial_state()
    for c0, p0 in root.chance_outcomes():
        s0 = root.apply(c0)
        for c1, p1 in s0.chance_outcomes():
            s1 = s0.apply(c1).apply(C).apply(C)
            for pub, p2 in s1.chance_outcomes():
                probs[(c0, c1, pub)] = (
                    Fraction(p0).limit_denominator(10)
                    * Fraction(p1).limit_denominator(10)
                    * Fraction(p2).limit_denominator(10)
                )
    assert len(probs) == 120
    assert set(probs.values()) == {Fraction(1, 120)}


def test_dealing_rejects_used_cards() -> None:
    with pytest.raises(ValueError):
        GAME.initial_state().apply(3).apply(3)
    with pytest.raises(ValueError):
        play(0, 1, C, C).apply(1)


# -- betting -----------------------------------------------------------------------------


def test_legal_actions_follow_the_rules() -> None:
    for state in STATES:
        if state.is_chance_node or state.is_terminal:
            assert state.legal_actions() == ()
            continue
        legal = state.legal_actions()
        facing = state.contributions[0] != state.contributions[1]
        raises = sum(a == R for a in state.rounds[-1])
        assert (F in legal) == facing
        assert C in legal
        assert (R in legal) == (raises < 2)
        assert legal == tuple(sorted(legal))


def test_round_action_sequences() -> None:
    """Every betting sequence a round can end with, and how it ends."""
    endings: dict[tuple[int, ...], str] = {}
    for state in STATES:
        if state.is_terminal and state.round_index == 0:
            endings[state.rounds[0]] = "fold"
        elif state.is_chance_node and len(state.private) == 2:
            endings[state.rounds[0]] = "next round"
    assert endings == {
        (C, C): "next round",
        (R, C): "next round",
        (C, R, C): "next round",
        (R, R, C): "next round",
        (C, R, R, C): "next round",
        (R, F): "fold",
        (C, R, F): "fold",
        (R, R, F): "fold",
        (C, R, R, F): "fold",
    }


def test_turn_order_and_round_boundaries() -> None:
    s = play(5, 0)
    assert s.current_player == 0 and s.round_index == 0
    s = s.apply(C)
    assert s.current_player == 1
    s = s.apply(R)
    assert s.current_player == 0 and s.contributions == (1, 3)
    s = s.apply(R)
    assert s.contributions == (5, 3) and s.legal_actions() == (F, C)  # raise cap
    s = s.apply(C)
    assert s.is_chance_node and s.contributions == (5, 5)
    s = s.apply(1)
    assert s.round_index == 1 and s.current_player == 0  # player 0 opens round 2
    assert s.legal_actions() == (C, R)
    s = s.apply(R)
    assert s.contributions == (9, 5)  # round-2 raise size is 4
    s = s.apply(C)
    assert s.is_terminal and s.current_player == TERMINAL


def test_illegal_actions_rejected() -> None:
    s = play(0, 2)
    with pytest.raises(ValueError):
        s.apply(F)  # nothing to fold to
    with pytest.raises(ValueError):
        play(0, 2, R, R).apply(R)  # third raise
    with pytest.raises(ValueError):
        play(0, 2, R, F).apply(C)  # terminal
    with pytest.raises(ValueError):
        s.apply(7)


# -- payoffs -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("moves", "expected"),
    [
        # Values cross-checked with OpenSpiel's leduc_poker.
        ((5, 0, R, R, C, 1, R, C), (-9.0, 9.0)),  # P1's J pairs the public J
        ((0, 1, C, C, 4, C, C), (0.0, 0.0)),  # J vs J, no pair: split
        ((0, 2, C, C, 1, C, C), (1.0, -1.0)),  # P0 pairs
        ((4, 0, C, C, 1, C, C), (-1.0, 1.0)),  # P1 pairs, beats a K
        ((0, 2, C, C, 4, C, C), (-1.0, 1.0)),  # Q beats J
        ((0, 2, C, R, F), (-1.0, 1.0)),  # P0 folds to a bet
        ((0, 2, R, F), (1.0, -1.0)),  # P1 folds
        ((0, 2, R, R, F), (-3.0, 3.0)),  # P0 folds after betting 2
        ((2, 0, R, C, 5, R, F), (3.0, -3.0)),  # fold in round 2
        ((4, 0, R, R, C, 5, R, R, C), (13.0, -13.0)),  # maximum pot
    ],
)
def test_payoffs(moves: tuple[int, ...], expected: tuple[float, float]) -> None:
    state = play(*moves)
    assert state.is_terminal
    assert state.returns() == expected


def test_all_terminals_zero_sum_and_bounded() -> None:
    terminals = [s for s in STATES if s.is_terminal]
    assert len(terminals) == 5_520
    for s in terminals:
        r = s.returns()
        assert r[0] + r[1] == 0
        assert abs(r[0]) <= 13


def test_ties_split_exactly_when_ranks_match_without_pairs() -> None:
    for s in STATES:
        if s.is_terminal and s._folder is None:
            r0, r1 = rank(s.private[0]), rank(s.private[1])
            tie = r0 == r1
            assert (s.returns() == (0.0, 0.0)) == tie
            if tie:
                assert s.winner() is None


def test_fold_payoff_is_the_folders_contribution() -> None:
    for s in STATES:
        if s.is_terminal and s._folder is not None:
            folder = s._folder
            assert s.returns()[folder] == -s.contributions[folder]


def test_non_terminal_returns_rejected() -> None:
    with pytest.raises(ValueError):
        play(0, 1).returns()


def test_states_are_immutable() -> None:
    s = play(0, 1)
    with pytest.raises(AttributeError):
        s.public = 3


def test_config_validation_and_custom_stakes() -> None:
    with pytest.raises(ValueError):
        LeducConfig(ante=0)
    game = LeducGame(LeducConfig(ante=2, raise_sizes=(3, 6)))
    s = game.state_from_moves((4, 0, R, C, 1, R, C))
    assert s.returns() == (-11.0, 11.0)  # 2 + 3 + 6


# -- information ------------------------------------------------------------------------------


def test_observation_has_only_observable_fields() -> None:
    assert set(LeducObservation.__dataclass_fields__) == {
        "player",
        "private_card",
        "public_card",
        "rounds",
    }


def test_keys_independent_of_opponent_card() -> None:
    """Replacing the opponent's card by any other unseen card never changes the
    acting player's observation or key."""
    checked = 0
    for s in STATES:
        if s.is_chance_node or s.is_terminal:
            continue
        p = s.current_player
        own, opp = s.private[p], s.private[1 - p]
        unseen = (
            set(range(NUM_CARDS)) - {own, opp} - ({s.public} if s.public is not None else set())
        )
        views = {s.observation(p)}
        for alt in unseen:
            private = (own, alt) if p == 0 else (alt, own)
            moves = list(private)
            moves.extend(s.rounds[0])
            if s.public is not None:
                moves.append(s.public)
                moves.extend(s.rounds[1])
            views.add(GAME.state_from_moves(tuple(moves)).observation(p))
        assert len(views) == 1
        assert s.public is None or card_name(s.public) in s.information_state_key(p)
        checked += 1
    assert checked == 3_780


def test_perfect_recall() -> None:
    """All histories in an information set share the acting player's own card,
    the public card and the full action sequence (own actions included)."""
    seen: dict[str, set[tuple[object, ...]]] = {}
    for s in STATES:
        if s.is_chance_node or s.is_terminal:
            continue
        p = s.current_player
        own_actions = tuple(a for rnd in s.rounds for i, a in enumerate(rnd) if i % 2 == p)
        seen.setdefault(s.information_state_key(p), set()).add(
            (p, s.private[p], s.public, s.rounds, own_actions)
        )
    assert len(seen) == 936
    assert all(len(v) == 1 for v in seen.values())


def test_every_information_set_has_two_to_twenty_histories() -> None:
    specs = information_sets(GAME)
    # Round 1: 5 opponent cards. Round 2: 4 opponent cards (public card fixed in the key).
    assert {s.histories for s in specs.values()} == {4, 5}


def test_keys_format() -> None:
    assert play(5, 0).information_state_key(0) == "K1|"
    assert play(5, 0, C, R).information_state_key(0) == "K1|cr"
    s = play(5, 0, R, C, 2)
    assert s.information_state_key(0) == "K1|Q0|rc|"
    assert s.apply(R).information_state_key(1) == "J0|Q0|rc|r"


def test_round_one_and_two_counts() -> None:
    """Derived from the rules: round 1 has 6 deciding histories ("", c, r, cr, rr,
    crr) x 6 own cards = 36; round 2 has 5 round-1 endings x 6 own cards x 5 public
    cards x 6 deciding histories = 900."""
    specs = information_sets(GAME)
    round1 = [k for k in specs if k.count("|") == 1]
    round2 = [k for k in specs if k.count("|") == 3]
    assert len(round1) == 36
    assert len(round2) == 900
    assert {k.split("|")[1] for k in round1} == {"", "c", "r", "cr", "rr", "crr"}
    assert {k.split("|")[3] for k in round2} == {"", "c", "r", "cr", "rr", "crr"}
