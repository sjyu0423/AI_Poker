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


__all__ = [
    "OUT_OF_POPULATION",
    "card_blind_policy",
    "make_bait_switch",
    "make_out_of_population",
    "parametric_distribution",
    "trapper_policy",
]
