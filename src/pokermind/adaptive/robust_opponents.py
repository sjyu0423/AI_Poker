"""Opponents outside the M5 development population (M6 robustness tests).

None of these families occurs among the development opponents that tuned
the adapter or built its priors:

* ``station``: continues with any card and almost never raises.
* ``maniac``: raises or bets whenever allowed, with any card.
* ``trapper`` (deceptive): folds by true equity, but bets and raises with
  *weak* cards and checks or calls with strong ones, so raises signal weakness.
* ``card_blind``: fixed action frequencies per action set, independent of
  its card. Its public marginal *is* its policy.

``bait_switch`` opponents play one style long enough for hero to adapt,
then switch to the style that punishes that adaptation.
:func:`make_bait_switch_population` (M7B) builds a larger, more diverse
set of them: seven styles, ordered pairs and switch points drawn at random.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

import numpy as np

from pokermind.evaluation.game_exact import GameInfoSet
from pokermind.games.leduc import CALL, FOLD, RAISE
from pokermind.opponents.population import (
    Opponent,
    ParametricParams,
    Policy,
    parametric_distribution,
    parametric_policy,
    parse_key,
    showdown_equity,
)

OUT_OF_POPULATION: Final = ("station", "maniac", "trapper", "card_blind")
BAIT_SWITCH_HAND: Final = 100


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def trapper_policy(
    specs: Mapping[str, GameInfoSet], tightness: float, aggression: float, tremble: float
) -> Policy:
    policy = {}
    for key, spec in specs.items():
        own, public = parse_key(key)
        s = showdown_equity(own, public)
        legal = spec.actions
        raise_share = aggression * _sigmoid(10.0 * ((1.0 - s) - 0.6))
        dist = dict.fromkeys(legal, 0.0)
        if FOLD in legal:
            cont = _sigmoid(10.0 * (s - tightness))
            dist[FOLD] = 1.0 - cont
            if RAISE in legal:
                dist[RAISE], dist[CALL] = cont * raise_share, cont * (1.0 - raise_share)
            else:
                dist[CALL] = cont
        else:
            dist[RAISE], dist[CALL] = raise_share, 1.0 - raise_share
        n = len(legal)
        policy[key] = {a: (1 - tremble) * p + tremble / n for a, p in dist.items()}
    return policy


def card_blind_policy(
    specs: Mapping[str, GameInfoSet], fold: float, bet: float, raise_facing: float
) -> Policy:
    policy = {}
    for key, spec in specs.items():
        legal = spec.actions
        if FOLD in legal and RAISE in legal:
            dist = {FOLD: fold, CALL: 1 - fold - raise_facing, RAISE: raise_facing}
        elif FOLD in legal:
            dist = {FOLD: fold, CALL: 1 - fold}
        else:
            dist = {CALL: 1 - bet, RAISE: bet}
        policy[key] = dist
    return policy


def make_out_of_population(family: str, index: int, specs: Mapping[str, GameInfoSet]) -> Opponent:
    rng = np.random.default_rng([sum(map(ord, family)), index, 6])
    j = float(rng.uniform(-1, 1))
    if family == "station":
        params = ParametricParams(-0.5, 0.05 + 0.03 * j, 0.0, 0.02)
        policy, parameters = parametric_policy(params, specs), {"aggression": params.aggression}
    elif family == "maniac":
        params = ParametricParams(-0.5, 1.0, 1.0, 0.02 + 0.01 * j)
        policy, parameters = parametric_policy(params, specs), {"tremble": params.tremble}
    elif family == "trapper":
        tight, agg = 0.3 + 0.05 * j, 0.9
        policy, parameters = trapper_policy(specs, tight, agg, 0.02), {"tightness": tight}
    elif family == "card_blind":
        fold, bet, rf = 0.3 + 0.1 * j, 0.4, 0.15
        policy = card_blind_policy(specs, fold, bet, rf)
        parameters = {"fold": fold, "bet": bet, "raise_facing": rf}
    else:
        raise ValueError(f"unknown family {family!r}")
    return Opponent(
        f"out/{family}-{index}", family, "out_of_population", parameters, (policy,), None, (family,)
    )


def make_bait_switch(index: int, specs: Mapping[str, GameInfoSet]) -> Opponent:
    """index 0: station, then maniac; index 1: maniac, then a tight value-raiser (rock)."""
    station = parametric_policy(ParametricParams(-0.5, 0.05, 0.0, 0.02), specs)
    maniac = parametric_policy(ParametricParams(-0.5, 1.0, 1.0, 0.02), specs)
    rock = parametric_policy(ParametricParams(0.75, 1.0, 0.0, 0.02), specs)
    first, second, names = (
        (station, maniac, ("station", "maniac"))
        if index == 0
        else (maniac, rock, ("maniac", "rock"))
    )
    return Opponent(
        f"out/bait_switch-{index}:{names[0]}->{names[1]}@{BAIT_SWITCH_HAND}",
        "bait_switch",
        "out_of_population",
        {"first": names[0], "second": names[1]},
        (first, second),
        BAIT_SWITCH_HAND,
        names,
    )


BAIT_STYLES: Final = ("station", "maniac", "rock", "nit", "lag", "trapper", "card_blind")
BAIT_SWITCH_HANDS: Final = (50, 100, 200)


def bait_style(style: str, jitter: float, specs: Mapping[str, GameInfoSet]) -> Policy:
    """One of :data:`BAIT_STYLES`; ``jitter`` in [-1, 1] perturbs its parameters."""
    j = jitter
    if style == "station":
        return parametric_policy(ParametricParams(-0.5, 0.05 + 0.03 * j, 0.0, 0.02), specs)
    if style == "maniac":
        return parametric_policy(ParametricParams(-0.5, 1.0, 1.0, 0.02 + 0.01 * j), specs)
    if style == "rock":
        return parametric_policy(ParametricParams(0.75 + 0.05 * j, 1.0, 0.0, 0.02), specs)
    if style == "nit":
        return parametric_policy(ParametricParams(0.8 + 0.05 * j, 0.15, 0.0, 0.02), specs)
    if style == "lag":
        return parametric_policy(ParametricParams(0.05, 0.85 + 0.1 * j, 0.4, 0.03), specs)
    if style == "trapper":
        return trapper_policy(specs, 0.3 + 0.05 * j, 0.9, 0.02)
    if style == "card_blind":
        return card_blind_policy(specs, 0.3 + 0.1 * j, 0.4, 0.15)
    raise ValueError(f"unknown style {style!r}")


def make_bait_switch_population(count: int, specs: Mapping[str, GameInfoSet]) -> list[Opponent]:
    """``count`` bait-and-switch opponents (M7B): distinct ordered style pairs drawn
    without replacement (cycling if ``count`` exceeds the 42 pairs), jittered
    parameters and a switch point from :data:`BAIT_SWITCH_HANDS`."""
    pairs = [(a, b) for a in BAIT_STYLES for b in BAIT_STYLES if a != b]
    rng = np.random.default_rng([7, 2, count])
    order = rng.permutation(len(pairs))
    out = []
    for i in range(count):
        first, second = pairs[int(order[i % len(pairs)])]
        switch = int(BAIT_SWITCH_HANDS[int(rng.integers(len(BAIT_SWITCH_HANDS)))])
        j1, j2 = (float(x) for x in rng.uniform(-1, 1, size=2))
        out.append(
            Opponent(
                f"bait/{i}:{first}->{second}@{switch}",
                "bait_switch",
                "out_of_population",
                {"first": first, "second": second, "jitter": [j1, j2]},
                (bait_style(first, j1, specs), bait_style(second, j2, specs)),
                switch,
                (first, second),
            )
        )
    return out


__all__ = [
    "BAIT_STYLES",
    "BAIT_SWITCH_HANDS",
    "OUT_OF_POPULATION",
    "bait_style",
    "card_blind_policy",
    "make_bait_switch",
    "make_bait_switch_population",
    "make_out_of_population",
    "parametric_distribution",
    "trapper_policy",
]
