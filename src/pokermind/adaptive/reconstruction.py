"""Card-conditional opponent-policy reconstruction from observed hands (M6).

M5 estimated the *public-context marginal* ``p(a | context)``. A best response
needs the opponent's *card-conditional* policy ``π(a | card, context)``, which
the marginal does not determine (``docs/design/adaptive_strategy.md`` §2). This
module infers it with an explicit latent-variable model:

* **Parameters.** ``θ[k]``: a distribution over legal actions for every
  canonical (suit-free) opponent information set ``k = (own rank, public rank,
  betting history)``. 288 parameter vectors. Suits are tied, an assumption.
* **Prior.** ``θ[k] ~ Dirichlet(s · m[k])`` with prior mean ``m`` (the
  development population's average card-conditional policy, or the CFR
  strategy) and strength ``s``.
* **Likelihood of one hand.** The opponent's card ``c`` is latent, uniformly
  distributed over the cards hero can't see (not hero's card, not the public
  card if dealt). Given ``c``, the opponent's observed actions have probability
  ``Π_t θ[k(c, context_t)][a_t]``. At a showdown ``c`` is revealed. Hands that
  end with a fold still carry information, through the marginal over ``c``.
  This is the full likelihood, so showdown hands are not over-weighted (no
  selection bias).
* **Estimates.** A penalized EM (MAP-style smoothing) gives a point estimate
  ``θ̂``; data-augmentation Gibbs sampling (alternately sampling each hand's
  card and ``θ``) gives approximate posterior samples.

Hero's own card is legitimately known to hero and is used for card removal.
The true opponent policy is never an input.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from pokermind.evaluation.game_exact import GameInfoSet
from pokermind.games.leduc import RANK_NAMES, LeducGame, LeducState, rank
from pokermind.opponents.context import PublicHandRecord, public_context, public_record
from pokermind.strategy import GameStrategy

Vector = npt.NDArray[np.float64]
Theta = dict[str, Vector]
Policy = dict[str, dict[int, float]]
MIN_PRIOR: Final = 1e-3


def canonical_infoset(key: str) -> str:
    """``"K1|J0|rc|r"`` -> ``"K|J|rc|r"``; ``"K1|cr"`` -> ``"K|cr"``."""
    parts = key.split("|")
    parts[0] = parts[0][0]
    if len(parts) == 4:
        parts[1] = parts[1][0]
    return "|".join(parts)


def key_for(card: int, context: str) -> str:
    """Canonical opponent information set for holding ``card`` in public ``context``."""
    parts = context.split(":")
    own = RANK_NAMES[rank(card)]
    if parts[0] == "R1":
        return f"{own}|{parts[1]}"
    return f"{own}|{parts[1]}|{parts[2]}|{parts[3]}"


@dataclass(frozen=True, slots=True)
class ObserverHandRecord:
    """What hero knows after a hand: the public record plus hero's own card and the
    exact public card (if dealt). Never the opponent's card unless shown down."""

    public: PublicHandRecord
    hero_card: int
    public_card: int | None


def observer_record(
    game: LeducGame, moves: tuple[int, ...], opponent_seat: int, hand_index: int = 0
) -> ObserverHandRecord:
    record = public_record(game, moves, opponent_seat, hand_index)
    state: LeducState = game.state_from_moves(moves)
    return ObserverHandRecord(record, state.private[1 - opponent_seat], state.public)


@dataclass(frozen=True, slots=True)
class HandEvidence:
    candidates: tuple[int, ...]
    """Opponent cards consistent with what hero saw (one card after a showdown)."""
    steps: tuple[tuple[str, tuple[int, ...], int], ...]
    """(context, legal actions, action) for each opponent decision."""


def evidence(record: ObserverHandRecord) -> HandEvidence:
    pub = record.public
    if pub.showdown_card is not None:
        candidates: tuple[int, ...] = (pub.showdown_card,)
    else:
        seen = {record.hero_card} | (
            {record.public_card} if record.public_card is not None else set()
        )
        candidates = tuple(c for c in range(6) if c not in seen)
    steps = tuple((d.context, d.legal_actions, d.action) for d in pub.decisions)
    return HandEvidence(candidates, steps)


def canonical_actions(specs: Mapping[str, GameInfoSet]) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for key, spec in specs.items():
        out.setdefault(canonical_infoset(key), spec.actions)
    return dict(sorted(out.items()))


def canonical_mean(policies: Iterable[GameStrategy], specs: Mapping[str, GameInfoSet]) -> Theta:
    """Average distribution per canonical information set over suits and policies."""
    actions = canonical_actions(specs)
    sums = {k: np.zeros(len(a)) for k, a in actions.items()}
    counts = dict.fromkeys(actions, 0)
    for policy in policies:
        for key, spec in specs.items():
            ck = canonical_infoset(key)
            sums[ck] += np.array([policy[key][a] for a in spec.actions])
            counts[ck] += 1
    out = {}
    for k, total in sums.items():
        mean = np.maximum(total / max(counts[k], 1), MIN_PRIOR)
        out[k] = mean / mean.sum()
    return out


def expand(theta: Mapping[str, Vector], specs: Mapping[str, GameInfoSet]) -> Policy:
    """A canonical parameter table as a full 936-key strategy (suits tied)."""
    return {
        key: {a: float(p) for a, p in zip(spec.actions, theta[canonical_infoset(key)], strict=True)}
        for key, spec in specs.items()
    }


@dataclass(slots=True)
class Reconstruction:
    theta: Theta
    expected_counts: Theta
    iterations: int
    log_likelihood: float


class CardConditionalModel:
    """Dirichlet prior over card-conditional policies + latent-card likelihood."""

    def __init__(self, prior_mean: Mapping[str, Vector], strength: float) -> None:
        if strength <= 0:
            raise ValueError(f"prior strength must be positive, got {strength}")
        self.prior_mean = {k: np.asarray(v, dtype=float) for k, v in prior_mean.items()}
        self.strength = strength

    def _responsibilities(
        self, hand: HandEvidence, theta: Mapping[str, Vector]
    ) -> tuple[Vector, float]:
        lik = np.ones(len(hand.candidates))
        for i, card in enumerate(hand.candidates):
            for context, legal, action in hand.steps:
                lik[i] *= theta[key_for(card, context)][legal.index(action)]
        total = float(lik.sum())
        if total <= 0:
            raise ArithmeticError("observed actions have zero probability under the model")
        return lik / total, total / len(hand.candidates)

    def _counts(self, hands: Sequence[HandEvidence], weights: Sequence[Vector]) -> Theta:
        counts = {k: np.zeros(len(v)) for k, v in self.prior_mean.items()}
        for hand, w in zip(hands, weights, strict=True):
            for i, card in enumerate(hand.candidates):
                if w[i] == 0.0:
                    continue
                for context, legal, action in hand.steps:
                    counts[key_for(card, context)][legal.index(action)] += w[i]
        return counts

    def _posterior_mean(self, counts: Mapping[str, Vector]) -> Theta:
        s = self.strength
        return {k: (counts[k] + s * m) / (counts[k].sum() + s) for k, m in self.prior_mean.items()}

    def fit(
        self, hands: Sequence[HandEvidence], max_iterations: int = 200, tolerance: float = 1e-9
    ) -> Reconstruction:
        """Penalized EM: E-step card responsibilities, M-step posterior-mean update."""
        theta = {k: v.copy() for k, v in self.prior_mean.items()}
        counts = {k: np.zeros(len(v)) for k, v in theta.items()}
        loglik = 0.0
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            resp, loglik = [], 0.0
            for hand in hands:
                r, lik = self._responsibilities(hand, theta)
                resp.append(r)
                loglik += float(np.log(lik))
            counts = self._counts(hands, resp)
            new = self._posterior_mean(counts)
            change = max(float(np.abs(new[k] - theta[k]).max()) for k in theta)
            theta = new
            if change < tolerance:
                break
        return Reconstruction(theta, counts, iterations, loglik)

    def posterior_samples(
        self,
        hands: Sequence[HandEvidence],
        num_samples: int,
        rng: np.random.Generator,
        start: Mapping[str, Vector] | None = None,
        burn_in: int = 10,
        thin: int = 2,
        temperature: float = 1.0,
    ) -> list[Theta]:
        """Data-augmentation Gibbs: sample every hand's card, then ``θ`` from its
        Dirichlet posterior. Deterministic given ``rng``.

        ``temperature < 1`` tempers the likelihood of the action counts (M7B):
        ``θ`` is drawn from ``Dirichlet(τ·counts + s·m)``, so the data count as
        ``τ`` times as many hands and the posterior widens. ``τ = 1`` is the
        untempered M6 sampler, unchanged.
        """
        if not 0.0 < temperature <= 1.0:
            raise ValueError(f"temperature must be in (0, 1], got {temperature}")
        theta = {k: v.copy() for k, v in (start or self.prior_mean).items()}
        samples: list[Theta] = []
        sweep = 0
        while len(samples) < num_samples:
            onehots = []
            for hand in hands:
                r, _ = self._responsibilities(hand, theta)
                j = int(rng.choice(len(r), p=r))
                onehot = np.zeros(len(r))
                onehot[j] = 1.0
                onehots.append(onehot)
            counts = self._counts(hands, onehots)
            s = self.strength
            theta = {}
            for k, m in self.prior_mean.items():
                draw = rng.dirichlet(temperature * counts[k] + s * m)
                draw = np.maximum(draw, 1e-12)
                theta[k] = draw / draw.sum()
            sweep += 1
            if sweep > burn_in and (sweep - burn_in) % thin == 0:
                samples.append(theta)
        return samples


def marginal_copy(
    predict: Callable[[str, tuple[int, ...]], Vector],
    specs: Mapping[str, GameInfoSet],
    contexts: Mapping[str, str],
) -> Policy:
    """The *naive* reconstruction: give every card the public-context marginal.

    ``contexts`` maps each information-set key to its public context. This
    ignores that actions reveal cards; it exists only as a cautionary baseline.
    """
    out: Policy = {}
    for key, spec in specs.items():
        q = predict(contexts[key], spec.actions)
        out[key] = {a: float(p) for a, p in zip(spec.actions, q, strict=True)}
    return out


def infoset_contexts(game: LeducGame) -> dict[str, str]:
    """Public context of every information set (the key without the own card)."""
    out: dict[str, str] = {}

    def walk(state: LeducState) -> None:
        if state.is_terminal:
            return
        if state.is_chance_node:
            for card, _ in state.chance_outcomes():
                walk(state.apply(card))
            return
        out.setdefault(state.information_state_key(state.current_player), public_context(state))
        for a in state.legal_actions():
            walk(state.apply(a))

    walk(game.initial_state())
    return out


def opponent_infoset_weights(
    game: LeducGame, hero: GameStrategy, opponent: GameStrategy
) -> dict[str, float]:
    """Expected visits per hand to each opponent information set (seats alternate)."""
    weights: dict[str, float] = {}

    def walk(state: LeducState, reach: float, opp_seat: int) -> None:
        if state.is_terminal or reach == 0.0:
            return
        if state.is_chance_node:
            for card, p in state.chance_outcomes():
                walk(state.apply(card), reach * p, opp_seat)
            return
        player = state.current_player
        key = state.information_state_key(player)
        table = opponent if player == opp_seat else hero
        if player == opp_seat:
            weights[key] = weights.get(key, 0.0) + reach
        for a in state.legal_actions():
            walk(state.apply(a), reach * table[key][a], opp_seat)

    for seat in (0, 1):
        walk(game.initial_state(), 0.5, seat)
    return weights


def reconstruction_error(
    estimate: GameStrategy, truth: GameStrategy, weights: Mapping[str, float]
) -> float:
    """Reach-weighted mean total-variation distance over opponent information sets."""
    num = den = 0.0
    for key, w in weights.items():
        tv = 0.5 * sum(abs(estimate[key][a] - truth[key][a]) for a in truth[key])
        num += w * tv
        den += w
    return num / den
