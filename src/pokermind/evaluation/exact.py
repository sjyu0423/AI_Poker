"""Exact evaluation of tabular strategies by enumerating the Kuhn Poker tree.

Independent of :mod:`pokermind.learning`: the only code this shares with CFR
is the M0 game engine. Everything here is exact up to floating-point rounding
(no sampling).

Conventions (see ``docs/concepts/cfr.md`` §9):

* Values are **net chips per hand** under the game's :class:`KuhnConfig`.
* ``expected_value`` returns **seat 0's** expected payoff; seat 1 gets the
  negative.
* ``NashConv(σ) = Σ_i [u_i(BR_i(σ_{-i}), σ_{-i}) − u_i(σ)]``.
* ``exploitability(σ) = NashConv(σ) / 2`` (mean over the two players; the
  same convention as OpenSpiel's ``exploitability``).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from pokermind.games.kuhn import (
    NUM_PLAYERS,
    Action,
    KuhnConfig,
    KuhnState,
    all_deals,
    initial_state,
)
from pokermind.strategy import Strategy

DEFAULT_ATOL = 1e-9


class InvalidStrategyError(ValueError):
    """Raised when a strategy table is malformed for the game."""


@dataclass(frozen=True, slots=True)
class InfoSetSpec:
    key: str
    player: int
    actions: tuple[Action, ...]


def _walk(state: KuhnState) -> Iterator[KuhnState]:
    yield state
    for action in state.legal_actions():
        yield from _walk(state.apply(action))


def information_sets(config: KuhnConfig | None = None) -> dict[str, InfoSetSpec]:
    """Every information set in the game, derived by walking the engine's tree."""
    specs: dict[str, InfoSetSpec] = {}
    for cards in all_deals():
        for state in _walk(initial_state(cards, config)):
            if state.is_terminal:
                continue
            player = state.current_player
            key = state.information_state(player).key
            spec = InfoSetSpec(key, player, state.legal_actions())
            if specs.setdefault(key, spec) != spec:  # pragma: no cover - engine invariant
                raise AssertionError(f"inconsistent information set {key!r}")
    return specs


def validate_strategy(
    strategy: Strategy,
    players: Sequence[int] = (0, 1),
    *,
    config: KuhnConfig | None = None,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Raise :class:`InvalidStrategyError` unless ``strategy`` is a valid behaviour
    strategy for ``players``.

    Requirements: an entry for every information set of those players, no
    unknown keys, each entry assigns a probability to exactly the legal actions,
    every probability is finite and in ``[0, 1]``, and each entry sums to 1
    within ``atol``.
    """
    specs = information_sets(config)
    required = {k for k, s in specs.items() if s.player in players}
    unknown = set(strategy) - set(specs)
    if unknown:
        raise InvalidStrategyError(f"unknown information sets: {sorted(unknown)}")
    missing = required - set(strategy)
    if missing:
        raise InvalidStrategyError(f"missing information sets: {sorted(missing)}")
    for key, dist in strategy.items():
        legal = set(specs[key].actions)
        if set(dist) != legal:
            raise InvalidStrategyError(
                f"{key!r}: actions {sorted(a.code for a in dist)} != legal "
                f"{sorted(a.code for a in legal)}"
            )
        total = 0.0
        for action, prob in dist.items():
            if not isinstance(prob, int | float) or isinstance(prob, bool):
                raise InvalidStrategyError(f"{key!r}/{action.code}: not a number: {prob!r}")
            if not math.isfinite(prob) or prob < -atol or prob > 1 + atol:
                raise InvalidStrategyError(f"{key!r}/{action.code}: invalid probability {prob!r}")
            total += prob
        if abs(total - 1.0) > atol:
            raise InvalidStrategyError(f"{key!r}: probabilities sum to {total!r}, not 1")


def uniform_strategy(config: KuhnConfig | None = None) -> dict[str, dict[Action, float]]:
    """Uniform over legal actions at every information set (the RandomAgent)."""
    return {
        key: {a: 1.0 / len(spec.actions) for a in spec.actions}
        for key, spec in information_sets(config).items()
    }


# -- expected value ---------------------------------------------------------


def _ev(state: KuhnState, seat_strategies: tuple[Strategy, Strategy]) -> float:
    if state.is_terminal:
        return float(state.payoffs()[0])
    player = state.current_player
    dist = seat_strategies[player][state.information_state(player).key]
    return sum(prob * _ev(state.apply(a), seat_strategies) for a, prob in dist.items() if prob)


def expected_value(
    seat_strategies: tuple[Strategy, Strategy], config: KuhnConfig | None = None
) -> float:
    """Exact expected payoff for **seat 0** when seat ``p`` plays ``seat_strategies[p]``."""
    for seat in (0, 1):
        validate_strategy(seat_strategies[seat], (seat,), config=config)
    deals = all_deals()
    return sum(_ev(initial_state(c, config), seat_strategies) for c in deals) / len(deals)


def expected_value_seat_averaged(
    strategy_a: Strategy, strategy_b: Strategy, config: KuhnConfig | None = None
) -> float:
    """Expected payoff for player A per hand when A and B swap seats every hand."""
    as_seat0 = expected_value((strategy_a, strategy_b), config)
    as_seat1 = -expected_value((strategy_b, strategy_a), config)
    return (as_seat0 + as_seat1) / 2


# -- best response ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BestResponse:
    player: int
    value: float
    """Expected payoff for ``player`` when best-responding (chips per hand)."""
    policy: dict[str, Action]
    """A pure best-response action for each of ``player``'s information sets."""

    def as_strategy(self) -> dict[str, dict[Action, float]]:
        specs = information_sets()
        return {
            key: {a: (1.0 if a is chosen else 0.0) for a in specs[key].actions}
            for key, chosen in self.policy.items()
        }


def best_response(
    strategy: Strategy, player: int, config: KuhnConfig | None = None
) -> BestResponse:
    """Exact best response for ``player`` against ``strategy``'s other seat.

    One action is chosen per *information set*: the action that maximizes the
    counterfactual value ``Σ_{h∈I} π_c(h)·π_{-i}(h)·u_i(h·a)``. Information sets
    are resolved deepest first, so the values below ``I`` already use the best
    response. Ties go to the first legal action (passive first).
    """
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player!r}")
    opponent = 1 - player
    validate_strategy(strategy, (opponent,), config=config)
    specs = information_sets(config)
    deals = all_deals()
    chance = 1.0 / len(deals)

    # 1. Histories in each of the player's information sets, with weights π_c·π_{-i}.
    members: dict[str, list[tuple[KuhnState, float]]] = {
        k: [] for k, s in specs.items() if s.player == player
    }

    def collect(state: KuhnState, weight: float) -> None:
        if state.is_terminal or weight == 0.0:
            return
        actor = state.current_player
        if actor == player:
            members[state.information_state(actor).key].append((state, weight))
            for action in state.legal_actions():
                collect(state.apply(action), weight)
        else:
            dist = strategy[state.information_state(actor).key]
            for action in state.legal_actions():
                collect(state.apply(action), weight * dist[action])

    for cards in deals:
        collect(initial_state(cards, config), chance)

    # 2. Resolve information sets deepest first.
    policy: dict[str, Action] = {}

    def value(state: KuhnState) -> float:
        if state.is_terminal:
            return float(state.payoffs()[player])
        actor = state.current_player
        key = state.information_state(actor).key
        if actor == player:
            return value(state.apply(policy[key]))
        return sum(p * value(state.apply(a)) for a, p in strategy[key].items() if p)

    for key in sorted(members, key=lambda k: len(k), reverse=True):
        actions = specs[key].actions
        best_action, best_q = actions[0], -math.inf
        for action in actions:
            q = sum(w * value(h.apply(action)) for h, w in members[key])
            if q > best_q:
                best_action, best_q = action, q
        policy[key] = best_action

    total = sum(chance * value(initial_state(c, config)) for c in deals)
    return BestResponse(player=player, value=total, policy=policy)


# -- NashConv and exploitability ----------------------------------------------


@dataclass(frozen=True, slots=True)
class ExploitabilityReport:
    on_policy_values: tuple[float, float]
    """``u_i(σ)`` for each player."""
    best_response_values: tuple[float, float]
    """``u_i(BR_i(σ_{-i}), σ_{-i})`` for each player."""
    nash_conv: float
    exploitability: float
    """``nash_conv / 2``."""


def exploitability_report(
    strategy: Strategy, config: KuhnConfig | None = None
) -> ExploitabilityReport:
    """NashConv and exploitability of the profile in which both seats play ``strategy``."""
    validate_strategy(strategy, (0, 1), config=config)
    u0 = expected_value((strategy, strategy), config)
    on_policy = (u0, -u0)
    br = tuple(best_response(strategy, p, config).value for p in range(NUM_PLAYERS))
    nash_conv = sum(br[p] - on_policy[p] for p in range(NUM_PLAYERS))
    return ExploitabilityReport(
        on_policy_values=on_policy,
        best_response_values=(br[0], br[1]),
        nash_conv=nash_conv,
        exploitability=nash_conv / NUM_PLAYERS,
    )


def nash_conv(strategy: Strategy, config: KuhnConfig | None = None) -> float:
    return exploitability_report(strategy, config).nash_conv


def exploitability(strategy: Strategy, config: KuhnConfig | None = None) -> float:
    return exploitability_report(strategy, config).exploitability
