"""Exact evaluation for any two-player zero-sum ``Game`` (M3).

The generic counterpart of :mod:`pokermind.evaluation.exact`, with the same
conventions:

* values are net chips per hand; ``expected_value`` is **seat 0's** payoff;
* ``NashConv(σ) = Σ_i [u_i(BR_i(σ_{-i}), σ_{-i}) − u_i(σ)]``;
* ``exploitability = NashConv / 2`` (OpenSpiel's convention).

It shares no code with the learning package. Chance nodes are weighted by
their listed probabilities. The best response picks **one action per
information set**: for player ``i`` it maximizes
``Σ_{h∈I} π_c(h)·π_{-i}(h)·u_i(h·a)``, resolving information sets from the
deepest to the shallowest. The acting player's own key never contains the
opponent's cards, so the best response can't condition on hidden
information.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pokermind.games.base import NUM_PLAYERS, Game, GameState
from pokermind.strategy import GameStrategy

DEFAULT_ATOL = 1e-9


class InvalidGameStrategyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GameInfoSet:
    key: str
    player: int
    actions: tuple[int, ...]
    depth: int
    """Number of moves (chance outcomes and actions) from the root."""
    histories: int
    """Number of game histories in the information set."""


@dataclass(frozen=True, slots=True)
class TreeCounts:
    chance: int
    decision: int
    terminal: int


def tree_counts(game: Game) -> TreeCounts:
    counts = [0, 0, 0]

    def walk(state: GameState) -> None:
        if state.is_terminal:
            counts[2] += 1
        elif state.is_chance_node:
            counts[0] += 1
            for outcome, _ in state.chance_outcomes():
                walk(state.apply(outcome))
        else:
            counts[1] += 1
            for action in state.legal_actions():
                walk(state.apply(action))

    walk(game.initial_state())
    return TreeCounts(*counts)


def information_sets(game: Game) -> dict[str, GameInfoSet]:
    """Every information set, found by walking the full tree.

    Raises if two histories with the same key disagree on the acting player,
    the legal actions or the depth. Any of those would mean the key doesn't
    identify an information set.
    """
    found: dict[str, list[tuple[int, tuple[int, ...], int]]] = {}

    def walk(state: GameState, depth: int) -> None:
        if state.is_terminal:
            return
        if state.is_chance_node:
            for outcome, _ in state.chance_outcomes():
                walk(state.apply(outcome), depth + 1)
            return
        player = state.current_player
        actions = state.legal_actions()
        found.setdefault(state.information_state_key(player), []).append((player, actions, depth))
        for action in actions:
            walk(state.apply(action), depth + 1)

    walk(game.initial_state(), 0)
    result = {}
    for key, entries in found.items():
        if len(set(entries)) != 1:
            raise ValueError(f"information set {key!r} is inconsistent: {set(entries)}")
        player, actions, depth = entries[0]
        result[key] = GameInfoSet(key, player, actions, depth, len(entries))
    return result


def validate_game_strategy(
    game: Game,
    strategy: GameStrategy,
    players: Sequence[int] = (0, 1),
    *,
    infosets: dict[str, GameInfoSet] | None = None,
    atol: float = DEFAULT_ATOL,
) -> None:
    specs = infosets if infosets is not None else information_sets(game)
    unknown = set(strategy) - set(specs)
    if unknown:
        raise InvalidGameStrategyError(f"unknown information sets: {sorted(unknown)[:5]}")
    missing = {k for k, s in specs.items() if s.player in players} - set(strategy)
    if missing:
        raise InvalidGameStrategyError(f"missing information sets: {sorted(missing)[:5]}")
    for key, dist in strategy.items():
        if set(dist) != set(specs[key].actions):
            raise InvalidGameStrategyError(
                f"{key!r}: actions {sorted(dist)} != legal {sorted(specs[key].actions)}"
            )
        total = 0.0
        for action, prob in dist.items():
            if isinstance(prob, bool) or not isinstance(prob, int | float):
                raise InvalidGameStrategyError(f"{key!r}/{action}: not a number: {prob!r}")
            if not math.isfinite(prob) or prob < -atol or prob > 1 + atol:
                raise InvalidGameStrategyError(f"{key!r}/{action}: invalid probability {prob!r}")
            total += prob
        if abs(total - 1.0) > atol:
            raise InvalidGameStrategyError(f"{key!r}: probabilities sum to {total!r}")


def uniform_game_strategy(
    game: Game, infosets: dict[str, GameInfoSet] | None = None
) -> dict[str, dict[int, float]]:
    specs = infosets if infosets is not None else information_sets(game)
    return {k: {a: 1.0 / len(s.actions) for a in s.actions} for k, s in specs.items()}


# -- expected value ------------------------------------------------------------------------


def expected_value(
    game: Game,
    seat_strategies: tuple[GameStrategy, GameStrategy],
    *,
    infosets: dict[str, GameInfoSet] | None = None,
) -> float:
    """Exact expected payoff for seat 0 when seat ``p`` plays ``seat_strategies[p]``."""
    specs = infosets if infosets is not None else information_sets(game)
    for seat in (0, 1):
        validate_game_strategy(game, seat_strategies[seat], (seat,), infosets=specs)

    def ev(state: GameState) -> float:
        if state.is_terminal:
            return state.returns()[0]
        if state.is_chance_node:
            return sum(p * ev(state.apply(o)) for o, p in state.chance_outcomes())
        player = state.current_player
        dist = seat_strategies[player][state.information_state_key(player)]
        return sum(prob * ev(state.apply(a)) for a, prob in dist.items() if prob)

    return ev(game.initial_state())


def expected_value_seat_averaged(
    game: Game, strategy_a: GameStrategy, strategy_b: GameStrategy
) -> float:
    """Player A's expected payoff per hand when A and B swap seats every hand."""
    specs = information_sets(game)
    as_seat0 = expected_value(game, (strategy_a, strategy_b), infosets=specs)
    as_seat1 = -expected_value(game, (strategy_b, strategy_a), infosets=specs)
    return (as_seat0 + as_seat1) / 2


# -- best response --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GameBestResponse:
    player: int
    value: float
    policy: dict[str, int]

    def as_strategy(self, infosets: dict[str, GameInfoSet]) -> dict[str, dict[int, float]]:
        return {
            k: {a: (1.0 if a == chosen else 0.0) for a in infosets[k].actions}
            for k, chosen in self.policy.items()
        }


def best_response(
    game: Game,
    strategy: GameStrategy,
    player: int,
    *,
    infosets: dict[str, GameInfoSet] | None = None,
) -> GameBestResponse:
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player!r}")
    specs = infosets if infosets is not None else information_sets(game)
    validate_game_strategy(game, strategy, (1 - player,), infosets=specs)

    # 1. Histories of each of the player's information sets, weighted by π_c · π_{-i}.
    members: dict[str, list[tuple[GameState, float]]] = {
        k: [] for k, s in specs.items() if s.player == player
    }

    def collect(state: GameState, weight: float) -> None:
        if state.is_terminal or weight == 0.0:
            return
        if state.is_chance_node:
            for outcome, prob in state.chance_outcomes():
                collect(state.apply(outcome), weight * prob)
            return
        actor = state.current_player
        key = state.information_state_key(actor)
        if actor == player:
            members[key].append((state, weight))
            for action in state.legal_actions():
                collect(state.apply(action), weight)
        else:
            dist = strategy[key]
            for action in state.legal_actions():
                collect(state.apply(action), weight * dist[action])

    collect(game.initial_state(), 1.0)

    # 2. Resolve deepest first; values below an information set use resolved choices.
    policy: dict[str, int] = {}

    def value(state: GameState) -> float:
        if state.is_terminal:
            return state.returns()[player]
        if state.is_chance_node:
            return sum(p * value(state.apply(o)) for o, p in state.chance_outcomes())
        actor = state.current_player
        key = state.information_state_key(actor)
        if actor == player:
            return value(state.apply(policy[key]))
        return sum(p * value(state.apply(a)) for a, p in strategy[key].items() if p)

    for key in sorted(members, key=lambda k: specs[k].depth, reverse=True):
        actions = specs[key].actions
        best_action, best_q = actions[0], -math.inf
        for action in actions:
            q = sum(w * value(h.apply(action)) for h, w in members[key])
            if q > best_q:
                best_action, best_q = action, q
        policy[key] = best_action

    return GameBestResponse(player, value(game.initial_state()), policy)


@dataclass(frozen=True, slots=True)
class GameExploitabilityReport:
    on_policy_values: tuple[float, float]
    best_response_values: tuple[float, float]
    nash_conv: float
    exploitability: float


def exploitability_report(
    game: Game, strategy: GameStrategy, *, infosets: dict[str, GameInfoSet] | None = None
) -> GameExploitabilityReport:
    specs = infosets if infosets is not None else information_sets(game)
    validate_game_strategy(game, strategy, (0, 1), infosets=specs)
    u0 = expected_value(game, (strategy, strategy), infosets=specs)
    on_policy = (u0, -u0)
    br = tuple(best_response(game, strategy, p, infosets=specs).value for p in range(NUM_PLAYERS))
    nash_conv = sum(br[p] - on_policy[p] for p in range(NUM_PLAYERS))
    return GameExploitabilityReport(
        on_policy_values=on_policy,
        best_response_values=(br[0], br[1]),
        nash_conv=nash_conv,
        exploitability=nash_conv / NUM_PLAYERS,
    )
