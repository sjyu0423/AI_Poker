"""Hierarchical Dirichlet-categorical opponent model (M5).

See ``docs/design/opponent_modeling.md`` §4 for the derivation. In short, for
an opponent decision in context ``c`` with legal actions ``A``, the model keeps
action counts at four levels (``context_levels``) and predicts

    q_3 = (n_3 + α_c b_3) / (N_3 + α_c),          b_3 = pop_3
    q_ℓ = (n_ℓ + α_ℓ b_ℓ) / (N_ℓ + α_ℓ),          b_ℓ = normalize(pop_ℓ · q_{ℓ+1} / pop_{ℓ+1})

with ``α_0 = alpha_fine`` and ``α_ℓ = alpha_coarse`` for ℓ ≥ 1. ``pop`` is the
population prior (uniform, or the average of development opponents). The
posterior at the finest level is ``Dirichlet(n_0 + α_0 b_0)``; each action's
marginal is ``Beta(a_i, Σa − a_i)``.

The model receives only :class:`~pokermind.opponents.context.PublicHandRecord`
objects. It never reads the showdown card.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
import numpy.typing as npt

from pokermind.opponents.context import NUM_LEVELS, PublicHandRecord, context_levels
from pokermind.opponents.exact import ContextStats

Vector = npt.NDArray[np.float64]
PRIOR_FLOOR: Final = 1e-4


# -- Beta distribution helpers (no SciPy dependency) ------------------------------------


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz's method for the incomplete-beta continued fraction (Numerical Recipes 6.4)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError(f"incomplete beta did not converge for a={a}, b={b}, x={x}")


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``: the CDF of ``Beta(a, b)`` at ``x``."""
    if a <= 0 or b <= 0:
        raise ValueError(f"Beta parameters must be positive, got {a}, {b}")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """Quantile of ``Beta(a, b)`` by bisection on :func:`beta_cdf` (accurate to ~1e-12)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# -- priors -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PopulationPrior:
    """Mean action distribution per level key; missing keys fall back to uniform."""

    levels: tuple[Mapping[str, Vector], ...] = field(
        default_factory=lambda: tuple({} for _ in range(NUM_LEVELS))
    )
    source: str = "uniform"

    def mean(self, level: int, key: str, size: int) -> Vector:
        found = self.levels[level].get(key)
        if found is None or len(found) != size:
            return np.full(size, 1.0 / size)
        return found

    @classmethod
    def uniform(cls) -> PopulationPrior:
        return cls()

    @classmethod
    def from_opponents(cls, stats: Iterable[ContextStats], source: str) -> PopulationPrior:
        """Average, over opponents, of each opponent's reach-weighted action
        distribution at every level key. Each opponent counts equally."""
        sums: list[dict[str, Vector]] = [{} for _ in range(NUM_LEVELS)]
        counts: list[dict[str, int]] = [{} for _ in range(NUM_LEVELS)]
        for s in stats:
            per_level: list[dict[str, tuple[Vector, float]]] = [{} for _ in range(NUM_LEVELS)]
            for ctx, w in s.weight.items():
                for level, key in enumerate(context_levels(ctx, s.legal[ctx])):
                    vec, tot = per_level[level].get(key, (np.zeros(len(s.legal[ctx])), 0.0))
                    per_level[level][key] = (vec + w * s.marginal[ctx], tot + w)
            for level in range(NUM_LEVELS):
                for key, (vec, tot) in per_level[level].items():
                    if tot <= 0:
                        continue
                    sums[level][key] = sums[level].get(key, np.zeros(len(vec))) + vec / tot
                    counts[level][key] = counts[level].get(key, 0) + 1
        levels = []
        for level in range(NUM_LEVELS):
            means = {}
            for key, total in sums[level].items():
                mean = np.maximum(total / counts[level][key], PRIOR_FLOOR)
                means[key] = mean / mean.sum()
            levels.append(means)
        return cls(tuple(levels), source)


# -- the model ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelConfig:
    alpha_fine: float = 4.0
    alpha_coarse: float = 4.0
    hierarchical: bool = True
    """False: each context on its own, ``Dirichlet(α_0 · pop_0)`` prior (no back-off)."""
    decay: float = 1.0
    """Multiply every count by this before each new hand (1 = cumulative)."""

    def __post_init__(self) -> None:
        if self.alpha_fine <= 0 or self.alpha_coarse <= 0:
            raise ValueError("concentrations must be positive")
        if not 0 < self.decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")

    @property
    def name(self) -> str:
        kind = "hier" if self.hierarchical else "flat"
        decay = "" if self.decay == 1 else f"_decay{self.decay:g}"
        return f"{kind}_a{self.alpha_fine:g}_c{self.alpha_coarse:g}{decay}"


class OpponentModel:
    """Sequential Bayesian estimates of an opponent's public-context action probabilities."""

    def __init__(self, config: ModelConfig, prior: PopulationPrior | None = None) -> None:
        self.config = config
        self.prior = prior or PopulationPrior.uniform()
        self.counts: list[dict[str, Vector]] = [{} for _ in range(NUM_LEVELS)]
        self.hands_observed = 0
        self.decisions_observed = 0

    # -- updates ----------------------------------------------------------------------------

    def observe_hand(self, record: PublicHandRecord) -> None:
        """Start a new hand (apply decay), then add each opponent decision."""
        if self.config.decay != 1.0:
            for level in self.counts:
                for vec in level.values():
                    vec *= self.config.decay
        for decision in record.decisions:
            self.update(decision.context, decision.legal_actions, decision.action)
        self.hands_observed += 1

    def update(self, context: str, legal_actions: tuple[int, ...], action: int) -> None:
        if action not in legal_actions:
            raise ValueError(f"action {action!r} is not legal in {context!r}: {legal_actions}")
        index = legal_actions.index(action)
        for level, key in enumerate(context_levels(context, legal_actions)):
            vec = self.counts[level].get(key)
            if vec is None:
                vec = self.counts[level][key] = np.zeros(len(legal_actions))
            vec[index] += 1.0
        self.decisions_observed += 1

    # -- predictions ------------------------------------------------------------------------

    def _counts(self, level: int, key: str, size: int) -> Vector:
        vec = self.counts[level].get(key)
        return vec if vec is not None else np.zeros(size)

    def posterior_parameters(self, context: str, legal_actions: tuple[int, ...]) -> Vector:
        """Dirichlet parameters ``n_0 + α_0 b_0`` at the finest level (in legal order)."""
        keys = context_levels(context, legal_actions)
        size = len(legal_actions)
        cfg, prior = self.config, self.prior
        if not cfg.hierarchical:
            return self._counts(0, keys[0], size) + cfg.alpha_fine * prior.mean(0, keys[0], size)
        q = prior.mean(NUM_LEVELS - 1, keys[-1], size)
        for level in range(NUM_LEVELS - 1, -1, -1):
            if level == NUM_LEVELS - 1:
                base = q
            else:
                ratio = prior.mean(level, keys[level], size) * q
                ratio = ratio / prior.mean(level + 1, keys[level + 1], size)
                base = ratio / ratio.sum()
            alpha = cfg.alpha_fine if level == 0 else cfg.alpha_coarse
            params = self._counts(level, keys[level], size) + alpha * base
            if level == 0:
                return params
            q = params / params.sum()
        raise AssertionError("unreachable")

    def predict(self, context: str, legal_actions: tuple[int, ...]) -> Vector:
        params = self.posterior_parameters(context, legal_actions)
        return params / params.sum()

    def posterior_std(self, context: str, legal_actions: tuple[int, ...]) -> Vector:
        a = self.posterior_parameters(context, legal_actions)
        a0 = a.sum()
        return np.sqrt(a * (a0 - a) / (a0 * a0 * (a0 + 1.0)))

    def credible_interval(
        self, context: str, legal_actions: tuple[int, ...], mass: float = 0.9
    ) -> tuple[Vector, Vector]:
        """Central credible interval of each action probability (Beta marginals)."""
        a = self.posterior_parameters(context, legal_actions)
        a0 = a.sum()
        tail = (1.0 - mass) / 2
        lo = np.array([beta_ppf(tail, ai, a0 - ai) for ai in a])
        hi = np.array([beta_ppf(1.0 - tail, ai, a0 - ai) for ai in a])
        return lo, hi

    def interval_contains(
        self, context: str, legal_actions: tuple[int, ...], truth: Vector, mass: float = 0.9
    ) -> npt.NDArray[np.bool_]:
        """Whether each true probability lies in the central ``mass`` interval (one CDF each)."""
        a = self.posterior_parameters(context, legal_actions)
        a0 = a.sum()
        tail = (1.0 - mass) / 2
        cdf = np.array([beta_cdf(float(t), ai, a0 - ai) for t, ai in zip(truth, a, strict=True)])
        return (cdf >= tail) & (cdf <= 1.0 - tail)

    def opportunities(self, context: str) -> float:
        """(Decayed) number of observed decisions in ``context``."""
        vec = self.counts[0].get(context)
        return 0.0 if vec is None else float(vec.sum())

    def snapshot(self) -> OpponentModel:
        return copy.deepcopy(self)

    def with_config(
        self, config: ModelConfig, prior: PopulationPrior | None = None
    ) -> OpponentModel:
        """The same counts under another prediction rule (decay must match)."""
        if config.decay != self.config.decay:
            raise ValueError("counts were accumulated with a different decay")
        other = copy.copy(self)
        other.config = config
        other.prior = prior if prior is not None else self.prior
        return other


def raw_frequency(model: OpponentModel, context: str, legal_actions: tuple[int, ...]) -> Vector:
    """Unsmoothed empirical frequencies in the exact context; uniform if never seen."""
    vec = model.counts[0].get(context)
    if vec is None or vec.sum() == 0:
        return np.full(len(legal_actions), 1.0 / len(legal_actions))
    out: Vector = vec / vec.sum()
    return out


# -- change detection ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    short_decay: float = 0.9
    threshold: float = 4.0
    drift: float = 0.0

    @property
    def name(self) -> str:
        return f"cusum_s{self.short_decay:g}_h{self.threshold:g}_d{self.drift:g}"


class ChangeDetectingModel:
    """Cumulative model + CUSUM change detector with reset.

    Each observed action scores ``log q_short(a) − log q_long(a) − drift``, both
    predicted *before* the update. The CUSUM ``S = max(0, S + score)`` grows
    when a short-memory model explains recent actions better than the
    long-memory one. When ``S`` exceeds ``threshold`` after a hand, an alarm is
    recorded and the long-memory counts are replaced by the short-memory
    counts (the recent evidence), and ``S`` is reset.
    """

    def __init__(
        self, config: ModelConfig, detector: DetectorConfig, prior: PopulationPrior | None = None
    ) -> None:
        if config.decay != 1.0:
            raise ValueError("the long-memory model must be cumulative (decay 1)")
        self.long = OpponentModel(config, prior)
        self.short = OpponentModel(replace(config, decay=detector.short_decay), prior)
        self.detector = detector
        self.statistic = 0.0
        self.alarms: list[int] = []

    @property
    def config(self) -> ModelConfig:
        return self.long.config

    def observe_hand(self, record: PublicHandRecord) -> None:
        for vec_level in self.short.counts:
            for vec in vec_level.values():
                vec *= self.detector.short_decay
        for d in record.decisions:
            i = d.legal_actions.index(d.action)
            q_long = self.long.predict(d.context, d.legal_actions)[i]
            q_short = self.short.predict(d.context, d.legal_actions)[i]
            score = math.log(q_short) - math.log(q_long) - self.detector.drift
            self.statistic = max(0.0, self.statistic + score)
            self.long.update(d.context, d.legal_actions, d.action)
            self.short.update(d.context, d.legal_actions, d.action)
        self.long.hands_observed += 1
        self.short.hands_observed += 1
        if self.statistic > self.detector.threshold:
            self.alarms.append(record.hand_index)
            self.long.counts = copy.deepcopy(self.short.counts)
            self.statistic = 0.0

    def predict(self, context: str, legal_actions: tuple[int, ...]) -> Vector:
        return self.long.predict(context, legal_actions)
