"""Simulated opponents with information-state-dependent tendencies (M5).

Every opponent is a complete Leduc strategy: a valid distribution over the
legal actions at all 936 information sets (checked with the M3 validator).

**Parametric families** act on the *showdown equity* ``s`` of the opponent's
own card: the probability of beating a uniformly random unseen opponent card
(ties count half), averaged over the unseen public card in round 1. Given
``s``, with ``σ`` the logistic function and steepness ``k``:

* facing a bet: ``P(continue) = σ(k (s − τ))``; ``P(fold) = 1 − P(continue)``.
  The tightness ``τ`` is the equity at which the opponent is indifferent
  between folding and continuing (tight: high ``τ``).
* raise share ``r(s) = α · (σ(k (s − ρ)) + β (1 − σ(k (s − ρ))))``: the aggression
  ``α`` scales value raises with ``s > ρ`` (``ρ = 0.6``) and bluffs (share ``β``)
  with weaker cards. Facing a bet, the continue mass is split into raise
  ``r(s)`` and call ``1 − r(s)``; with nothing to call, bet ``r(s)`` and check
  the rest. When raising is capped, the raise mass goes to call.
* a tremble ``ε`` mixes in the uniform distribution over legal actions.

**CFR-based families** start from a near-equilibrium CFR strategy:
``balanced`` mixes it with a little uniform play; ``perturbed_cfr`` replaces
each information set's distribution by a draw from ``Dirichlet(c · p + 0.1)``.

**Switching** opponents play one family's policy until hand ``T`` and another
family's policy from then on.

Each family has opponents in two disjoint splits, ``development`` (used to
tune the model and to build the population prior) and ``heldout`` (reported
results only), with parameters drawn from independent seeds.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from pokermind.evaluation.game_exact import (
    GameInfoSet,
    information_sets,
    validate_game_strategy,
)
from pokermind.games.leduc import CALL, FOLD, RAISE, RANK_NAMES, LeducGame
from pokermind.strategy import GameStrategy

Policy = dict[str, dict[int, float]]
SPLITS: Final = ("development", "heldout")
PARAMETRIC_RANGES: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "tight": {"tightness": (0.55, 0.70), "aggression": (0.4, 0.6), "bluff": (0.0, 0.1)},
    "loose": {"tightness": (0.00, 0.15), "aggression": (0.4, 0.6), "bluff": (0.05, 0.15)},
    "passive": {"tightness": (0.25, 0.45), "aggression": (0.05, 0.2), "bluff": (0.0, 0.0)},
    "aggressive": {"tightness": (0.25, 0.45), "aggression": (0.8, 1.0), "bluff": (0.2, 0.4)},
}
TREMBLE_RANGE: Final = (0.01, 0.05)
STEEPNESS: Final = 10.0
VALUE_THRESHOLD: Final = 0.6
BALANCED_EPSILON_RANGE: Final = (0.0, 0.05)
PERTURBATION_CONCENTRATION_RANGE: Final = (5.0, 30.0)
STATIONARY_FAMILIES: Final = (*PARAMETRIC_RANGES, "balanced", "perturbed_cfr")
SWITCH_HANDS: Final = (150, 250)


def card_from_name(name: str) -> int:
    return RANK_NAMES.index(name[0]) * 2 + int(name[1])


def parse_key(key: str) -> tuple[int, int | None]:
    """Own card and public card (or None) from an information-set key."""
    parts = key.split("|")
    own = card_from_name(parts[0])
    public = card_from_name(parts[1]) if len(parts) == 4 else None
    return own, public


def showdown_equity(own: int, public: int | None) -> float:
    """P(win) + ½ P(tie) against a uniformly random unseen card (round 1: also
    averaged over the unseen public card)."""

    def strength(card: int, pub: int) -> tuple[int, int]:
        return (int(card // 2 == pub // 2), card // 2)

    publics = [public] if public is not None else [c for c in range(6) if c != own]
    total = 0.0
    for pub in publics:
        opponents = [c for c in range(6) if c not in (own, pub)]
        score = 0.0
        for opp in opponents:
            mine, theirs = strength(own, pub), strength(opp, pub)
            score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        total += score / len(opponents)
    return total / len(publics)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True, slots=True)
class ParametricParams:
    tightness: float
    aggression: float
    bluff: float
    tremble: float
    steepness: float = STEEPNESS

    def __post_init__(self) -> None:
        if not (0 <= self.aggression <= 1 and 0 <= self.bluff <= 1 and 0 <= self.tremble <= 1):
            raise ValueError(f"invalid parameters {self!r}")


def parametric_distribution(
    params: ParametricParams, equity: float, legal: tuple[int, ...]
) -> dict[int, float]:
    k = params.steepness
    value = _sigmoid(k * (equity - VALUE_THRESHOLD))
    raise_share = params.aggression * (value + params.bluff * (1.0 - value))
    can_raise = RAISE in legal
    dist = dict.fromkeys(legal, 0.0)
    if FOLD in legal:
        cont = _sigmoid(k * (equity - params.tightness))
        dist[FOLD] = 1.0 - cont
        if can_raise:
            dist[RAISE] = cont * raise_share
            dist[CALL] = cont * (1.0 - raise_share)
        else:
            dist[CALL] = cont
    else:
        dist[RAISE] = raise_share
        dist[CALL] = 1.0 - raise_share
    n = len(legal)
    return {a: (1.0 - params.tremble) * p + params.tremble / n for a, p in dist.items()}


def parametric_policy(params: ParametricParams, specs: Mapping[str, GameInfoSet]) -> Policy:
    policy = {}
    for key, spec in specs.items():
        own, public = parse_key(key)
        policy[key] = parametric_distribution(params, showdown_equity(own, public), spec.actions)
    return policy


def mix_with_uniform(strategy: GameStrategy, epsilon: float) -> Policy:
    return {
        k: {a: (1 - epsilon) * p + epsilon / len(d) for a, p in d.items()}
        for k, d in strategy.items()
    }


def perturbed_policy(
    strategy: GameStrategy, concentration: float, rng: np.random.Generator
) -> Policy:
    """Each information set's distribution drawn from ``Dirichlet(c · p + 0.1)``."""
    policy = {}
    for key in sorted(strategy):
        actions = sorted(strategy[key])
        alpha = np.array([concentration * strategy[key][a] + 0.1 for a in actions])
        draw = rng.dirichlet(alpha)
        draw = np.maximum(draw, 1e-12)
        draw = draw / draw.sum()
        policy[key] = {a: float(p) for a, p in zip(actions, draw, strict=True)}
    return policy


@dataclass(frozen=True, slots=True)
class Opponent:
    name: str
    family: str
    split: str
    parameters: Mapping[str, Any]
    policies: tuple[Policy, ...]
    """One policy, or two for a switching opponent."""
    switch_hand: int | None = None
    """Hand index (0-based) from which ``policies[1]`` is played."""
    phase_families: tuple[str, ...] = field(default=())

    def policy_at(self, hand: int) -> Policy:
        if self.switch_hand is not None and hand >= self.switch_hand:
            return self.policies[1]
        return self.policies[0]

    @property
    def is_switching(self) -> bool:
        return self.switch_hand is not None


def _rng(*parts: str | int) -> np.random.Generator:
    digest = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(lo if lo == hi else rng.uniform(lo, hi))


def make_stationary(
    family: str,
    split: str,
    index: int,
    base: GameStrategy,
    specs: Mapping[str, GameInfoSet],
    tag: str = "",
) -> Opponent:
    """One opponent of ``family``; parameters depend only on (family, split, index, tag)."""
    rng = _rng("opponent", family, split, index, tag)
    name = f"{split}/{family}-{index}{tag}"
    if family in PARAMETRIC_RANGES:
        ranges = PARAMETRIC_RANGES[family]
        params = ParametricParams(
            tightness=_uniform(rng, ranges["tightness"]),
            aggression=_uniform(rng, ranges["aggression"]),
            bluff=_uniform(rng, ranges["bluff"]),
            tremble=_uniform(rng, TREMBLE_RANGE),
        )
        parameters: dict[str, Any] = {
            "tightness": params.tightness,
            "aggression": params.aggression,
            "bluff": params.bluff,
            "tremble": params.tremble,
        }
        policy = parametric_policy(params, specs)
    elif family == "balanced":
        epsilon = _uniform(rng, BALANCED_EPSILON_RANGE)
        parameters = {"uniform_mix": epsilon}
        policy = mix_with_uniform(base, epsilon)
    elif family == "perturbed_cfr":
        concentration = _uniform(rng, PERTURBATION_CONCENTRATION_RANGE)
        parameters = {"concentration": concentration}
        policy = perturbed_policy(base, concentration, rng)
    else:
        raise ValueError(f"unknown family {family!r}")
    return Opponent(name, family, split, parameters, (policy,), None, (family,))


def make_switching(
    split: str,
    index: int,
    base: GameStrategy,
    specs: Mapping[str, GameInfoSet],
    families: Sequence[str] = STATIONARY_FAMILIES,
) -> Opponent:
    rng = _rng("switching", split, index)
    first, second = (str(f) for f in rng.choice(list(families), size=2, replace=False))
    switch = int(SWITCH_HANDS[index % len(SWITCH_HANDS)])
    a = make_stationary(first, split, index, base, specs, tag=f"-switchA{index}")
    b = make_stationary(second, split, index, base, specs, tag=f"-switchB{index}")
    return Opponent(
        name=f"{split}/switching-{index}:{first}->{second}@{switch}",
        family="switching",
        split=split,
        parameters={"first": dict(a.parameters), "second": dict(b.parameters)},
        policies=(a.policies[0], b.policies[0]),
        switch_hand=switch,
        phase_families=(first, second),
    )


def build_population(
    game: LeducGame,
    base: GameStrategy,
    per_family: int = 4,
    switching_per_split: int = 4,
    families: Sequence[str] = STATIONARY_FAMILIES,
) -> dict[str, list[Opponent]]:
    """Development and held-out opponents; every policy is validated."""
    specs = information_sets(game)
    validate_game_strategy(game, base, infosets=specs)
    population: dict[str, list[Opponent]] = {}
    for split in SPLITS:
        opponents = [
            make_stationary(family, split, i, base, specs)
            for family in families
            for i in range(per_family)
        ]
        opponents += [make_switching(split, i, base, specs) for i in range(switching_per_split)]
        for opponent in opponents:
            for policy in opponent.policies:
                validate_game_strategy(game, policy, infosets=specs)
        population[split] = opponents
    return population
