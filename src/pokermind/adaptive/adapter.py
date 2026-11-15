"""From observed hands to adaptive hero policies (M6).

``Adapter.snapshot(records)`` turns hero's observations of one opponent into
a set of candidate hero policies. It sees only ``ObserverHandRecord``s
(public actions, hero's own card, the public card, showdown cards), the CFR
baseline and priors built from development opponents. It never receives the
true opponent policy.

Pipeline (``docs/design/adaptive_strategy.md`` §3-4):

1. **Prior selection.** Candidate priors (population or CFR mean, several
   strengths) are compared by 2-fold cross-validated likelihood of the
   observed actions (even vs odd hands, card marginalized). With fewer than
   ``min_hands_for_selection`` hands the default prior is used.
2. **Reconstruction.** Penalized EM gives ``θ̂`` and expected counts per
   canonical information set.
3. **Responses.**
   * ``br_map``: best response to ``θ̂``;
   * ``br_confidence``: best response to ``w·θ̂ + (1 − w)·π_CFR`` with
     ``w = N / (N + k)`` per information set (trust the model only where it
     has data);
   * ``br_naive``: best response to the public marginal copied to every card
     (cautionary baseline).
4. **Mixtures** ``(1 − α)·CFR + α·response`` (realization weights), with
   ``α`` limited by an exact exploitability budget.
5. **Gate.** Posterior samples (Gibbs) estimate the gain over CFR of each
   budgeted candidate (``budget_max`` and ``confidence_budget``). The candidate
   with the largest ``mean − z·sd`` is played if that bound is positive;
   otherwise hero plays CFR.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from pokermind.adaptive.reconstruction import (
    CardConditionalModel,
    HandEvidence,
    ObserverHandRecord,
    Theta,
    canonical_mean,
    evidence,
    expand,
    key_for,
    marginal_copy,
)
from pokermind.adaptive.response import (
    BudgetSearch,
    Mixer,
    Policy,
    exploitability,
    hero_best_response,
    hero_value,
)
from pokermind.evaluation.game_exact import GameInfoSet
from pokermind.games.leduc import LeducGame
from pokermind.opponents.bayes import (
    ChangeDetectingModel,
    DetectorConfig,
    ModelConfig,
    OpponentModel,
    PopulationPrior,
)
from pokermind.strategy import GameStrategy

M5_MODEL: Final = ModelConfig(alpha_fine=16.0, alpha_coarse=64.0)
"""The M5 selected public-context model (used by the naive baseline)."""
M5_DETECTOR: Final = DetectorConfig(short_decay=0.8, threshold=5.0, drift=0.0)
"""The M5 selected change detector."""


@dataclass(frozen=True, slots=True)
class AdaptationSettings:
    budget: float = 0.1
    """Allowed exploitability increase over the CFR baseline (chips/hand)."""
    frontier_budgets: tuple[float, ...] = (0.02, 0.05, 0.1, 0.25, 0.5)
    prior_strengths: tuple[float, ...] = (2.0, 8.0)
    default_prior: tuple[str, float] = ("population", 2.0)
    min_hands_for_selection: int = 10
    confidence_k: tuple[float, ...] = (10.0,)
    """Pseudo-count k of the confidence weight N / (N + k); the first is the default."""
    fixed_alpha: float = 0.25
    evidence_k: float = 100.0
    gate_z: float = 1.0
    posterior_samples: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.budget < 0 or any(b < 0 for b in self.frontier_budgets):
            raise ValueError("budgets must be non-negative")
        if not 0 <= self.fixed_alpha <= 1:
            raise ValueError("fixed_alpha must be in [0, 1]")
        if self.evidence_k <= 0 or any(k <= 0 for k in self.confidence_k):
            raise ValueError("pseudo-counts must be positive")
        if self.posterior_samples < 2:
            raise ValueError("need at least 2 posterior samples")


@dataclass(slots=True)
class ResponseResult:
    name: str
    response: Policy
    mixer: Mixer
    budget: BudgetSearch
    frontier: dict[float, BudgetSearch]


@dataclass(slots=True)
class Snapshot:
    hands: int
    prior: tuple[str, float]
    prior_scores: dict[str, float]
    estimate: Policy
    theta: Theta
    expected_counts: Theta
    responses: dict[str, ResponseResult]
    policies: dict[str, Policy]
    """Hero policies by method name (all built from observations only)."""
    alphas: dict[str, float]
    gate: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Priors:
    """Everything the adapter knows before seeing this opponent."""

    cfr: GameStrategy
    cfr_exploitability: float
    population_theta: Theta
    cfr_theta: Theta
    public_prior: PopulationPrior
    infoset_contexts: Mapping[str, str]

    def theta(self, kind: str) -> Theta:
        if kind == "population":
            return self.population_theta
        if kind == "cfr":
            return self.cfr_theta
        raise ValueError(f"unknown prior {kind!r}")


def build_priors(
    game: LeducGame,
    specs: dict[str, GameInfoSet],
    cfr: GameStrategy,
    development_policies: Sequence[GameStrategy],
    public_prior: PopulationPrior,
    infoset_contexts: Mapping[str, str],
) -> Priors:
    return Priors(
        cfr=cfr,
        cfr_exploitability=exploitability(game, cfr, specs),
        population_theta=canonical_mean(development_policies, specs),
        cfr_theta=canonical_mean([cfr], specs),
        public_prior=public_prior,
        infoset_contexts=infoset_contexts,
    )


def _cv_loglik(model: CardConditionalModel, hands: Sequence[HandEvidence]) -> float:
    """2-fold (even/odd hands) held-out log-likelihood of the observed actions."""
    folds = (hands[0::2], hands[1::2])
    total = 0.0
    for fit_on, score_on in ((folds[0], folds[1]), (folds[1], folds[0])):
        theta = model.fit(fit_on).theta
        for hand in score_on:
            lik = 0.0
            for card in hand.candidates:
                p = 1.0
                for context, legal, action in hand.steps:
                    p *= float(theta[key_for(card, context)][legal.index(action)])
                lik += p / len(hand.candidates)
            total += float(np.log(max(lik, 1e-300)))
    return total


class Adapter:
    def __init__(
        self,
        game: LeducGame,
        specs: dict[str, GameInfoSet],
        priors: Priors,
        settings: AdaptationSettings,
    ) -> None:
        self.game, self.specs, self.priors, self.settings = game, specs, priors, settings

    # -- reconstruction ------------------------------------------------------------------

    def select_prior(
        self, hands: Sequence[HandEvidence]
    ) -> tuple[tuple[str, float], dict[str, float]]:
        s = self.settings
        if len(hands) < s.min_hands_for_selection:
            return s.default_prior, {}
        scores = {}
        for kind in ("population", "cfr"):
            for strength in s.prior_strengths:
                model = CardConditionalModel(self.priors.theta(kind), strength)
                scores[f"{kind}:{strength:g}"] = _cv_loglik(model, hands)
        best = max(scores, key=lambda k: scores[k])
        best_kind, best_strength = best.split(":")
        return (best_kind, float(best_strength)), scores

    def confidence_theta(self, theta: Theta, counts: Theta, k: float) -> Theta:
        out = {}
        for key, est in theta.items():
            n = float(counts[key].sum())
            w = n / (n + k)
            out[key] = w * est + (1.0 - w) * self.priors.cfr_theta[key]
        return out

    # -- the snapshot ----------------------------------------------------------------------

    def snapshot(
        self,
        records: Sequence[ObserverHandRecord],
        public_model: OpponentModel | None = None,
        extra_confidence: bool = False,
    ) -> Snapshot:
        """Candidate hero policies after observing ``records`` (oldest first)."""
        s, game, specs, pri = self.settings, self.game, self.specs, self.priors
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        hands = [evidence(r) for r in records]
        prior, prior_scores = self.select_prior(hands)
        model = CardConditionalModel(pri.theta(prior[0]), prior[1])
        fit = model.fit(hands)
        estimate = expand(fit.theta, specs)
        timings["reconstruction"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if public_model is None:
            public_model = OpponentModel(M5_MODEL, pri.public_prior)
            for r in records:
                public_model.observe_hand(r.public)
        naive = marginal_copy(public_model.predict, specs, pri.infoset_contexts)
        targets: dict[str, GameStrategy] = {"br_map": estimate, "br_naive": naive}
        ks = s.confidence_k if extra_confidence else s.confidence_k[:1]
        for i, k in enumerate(ks):
            name = "br_confidence" if i == 0 else f"br_confidence_k{k:g}"
            targets[name] = expand(self.confidence_theta(fit.theta, fit.expected_counts, k), specs)
        responses: dict[str, ResponseResult] = {}
        base_budget = pri.cfr_exploitability + s.budget
        for name, target in targets.items():
            br = hero_best_response(game, target, specs)
            mixer = Mixer(game, specs, pri.cfr, br, pri.cfr_exploitability)
            if name == "br_naive":
                search = BudgetSearch(1.0, mixer.exploitability(1.0))
                frontier: dict[float, BudgetSearch] = {}
            else:
                search = mixer.largest_alpha(base_budget)
                frontier = {
                    b: mixer.largest_alpha(pri.cfr_exploitability + b, refine=3)
                    for b in s.frontier_budgets
                }
            responses[name] = ResponseResult(name, br, mixer, search, frontier)
        timings["responses_and_budgets"] = time.perf_counter() - t0

        # Gate: posterior gain over CFR of each budgeted candidate; keep the best
        # lower bound, adapt only if it is positive.
        t0 = time.perf_counter()
        mp = responses["br_map"]
        conf = responses["br_confidence"]
        rng = np.random.default_rng(np.random.SeedSequence((s.seed, len(records))))
        samples = [
            expand(th, specs)
            for th in model.posterior_samples(hands, s.posterior_samples, rng, start=fit.theta)
        ]
        cfr_values = [hero_value(game, pri.cfr, sample, specs) for sample in samples]
        candidates = {
            "budget_max": mp.mixer.policy(mp.budget.alpha),
            "confidence_budget": conf.mixer.policy(conf.budget.alpha),
        }
        for name, result in responses.items():
            if name.startswith("br_confidence_k"):
                suffix = name.removeprefix("br_confidence")
                candidates[f"confidence_budget{suffix}"] = result.mixer.policy(result.budget.alpha)
        gate: dict[str, Any] = {"candidates": {}}
        for name, policy in candidates.items():
            gains = [
                hero_value(game, policy, sample, specs) - v
                for sample, v in zip(samples, cfr_values, strict=True)
            ]
            mean, sd = float(np.mean(gains)), float(np.std(gains, ddof=1))
            gate["candidates"][name] = {
                "gains": gains,
                "mean": mean,
                "sd": sd,
                "lcb": mean - s.gate_z * sd,
            }
        chosen = max(
            ("budget_max", "confidence_budget"), key=lambda k: gate["candidates"][k]["lcb"]
        )
        gate["chosen"] = chosen
        gate["open"] = gate["candidates"][chosen]["lcb"] > 0
        timings["gate"] = time.perf_counter() - t0

        n = len(records)
        alphas = {
            "cfr": 0.0,
            "br_naive": 1.0,
            "br_map": 1.0,
            "budget_max": mp.budget.alpha,
            "fixed_mix": min(s.fixed_alpha, mp.budget.alpha),
            "evidence_mix": min(n / (n + s.evidence_k), mp.budget.alpha),
            "confidence_budget": conf.budget.alpha,
            "gated": (mp.budget.alpha if chosen == "budget_max" else conf.budget.alpha)
            if gate["open"]
            else 0.0,
        }
        policies: dict[str, Policy] = {
            "cfr": {k: dict(v) for k, v in pri.cfr.items()},
            "br_naive": responses["br_naive"].response,
            "br_map": mp.response,
            "budget_max": candidates["budget_max"],
            "fixed_mix": mp.mixer.policy(alphas["fixed_mix"]),
            "evidence_mix": mp.mixer.policy(alphas["evidence_mix"]),
            "confidence_budget": candidates["confidence_budget"],
            "gated": (
                candidates[chosen] if gate["open"] else {k: dict(v) for k, v in pri.cfr.items()}
            ),
        }
        return Snapshot(
            hands=n,
            prior=prior,
            prior_scores=prior_scores,
            estimate=estimate,
            theta=fit.theta,
            expected_counts=fit.expected_counts,
            responses=responses,
            policies=policies,
            alphas=alphas,
            gate=gate,
            timings=timings,
        )


def gain_given_alpha(value_cfr: float, value_response: float, alpha: float) -> float:
    """Payoff gain of the realization mixture over CFR (linear in α)."""
    return alpha * (value_response - value_cfr)


class ChangeAwareObserver:
    """Hero's observation history with M5's CUSUM detector (M6).

    After every hand the public record goes to the detector. When it raises an
    alarm, only the last ``retain`` hands are kept, and the detector's
    long-memory public model is rebuilt from exactly those hands, so the
    public model, the reconstruction data and the adaptive policy all restart
    from the same evidence.
    """

    def __init__(
        self,
        public_prior: PopulationPrior,
        detector: DetectorConfig = M5_DETECTOR,
        retain: int = 10,
        detect: bool = True,
    ) -> None:
        self.records: list[ObserverHandRecord] = []
        self.retained_from = 0
        self.retain = retain
        self.detect = detect
        self.public_prior = public_prior
        self.detector = ChangeDetectingModel(M5_MODEL, detector, public_prior)
        self.resets: list[int] = []

    def observe(self, record: ObserverHandRecord) -> None:
        self.records.append(record)
        alarms = len(self.detector.alarms)
        self.detector.observe_hand(record.public)
        if self.detect and len(self.detector.alarms) > alarms:
            self.retained_from = max(0, len(self.records) - self.retain)
            self.resets.append(record.public.hand_index)
            rebuilt = OpponentModel(M5_MODEL, self.public_prior)
            for r in self.retained():
                rebuilt.observe_hand(r.public)
            self.detector.long = rebuilt
            self.detector.statistic = 0.0

    def retained(self) -> list[ObserverHandRecord]:
        return self.records[self.retained_from :]

    def public_model(self) -> OpponentModel:
        """The public model matching :meth:`retained` (cumulative if detection is off)."""
        if self.detect:
            return self.detector.long
        model = OpponentModel(M5_MODEL, self.public_prior)
        for r in self.records:
            model.observe_hand(r.public)
        return model
