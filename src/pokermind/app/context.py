"""Everything the application loads once: the frozen research baseline (M8A).

:func:`load_app_context` reads ``configs/pokermind-research-v1.json``,
verifies the digest of the CFR hero file named there, and recomputes the
hero's exact exploitability, which must match the frozen value. It then builds
the same development-population priors that the M6-M7B studies used. All
research hyperparameters come from the configuration; none are declared here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pokermind.adaptive.adapter import AdaptationSettings, Adapter, Priors, build_priors
from pokermind.adaptive.reconstruction import infoset_contexts
from pokermind.adaptive.response import exploitability
from pokermind.evaluation.game_exact import GameInfoSet, information_sets
from pokermind.games.leduc import LeducGame
from pokermind.opponents.bayes import DetectorConfig, PopulationPrior
from pokermind.opponents.exact import context_stats
from pokermind.opponents.population import build_population
from pokermind.research import DEFAULT_CONFIG_PATH, ResearchConfig, load_research_config
from pokermind.robust.selection import SelectionRule, Selector
from pokermind.robust.tree import CompiledTree
from pokermind.strategy import GameStrategy, game_strategy_digest, load_game_strategy

EXPLOITABILITY_MATCH: Final = 1e-9
"""Allowed difference between the recomputed and the frozen hero exploitability."""


class ContextError(RuntimeError):
    """The frozen baseline cannot be loaded as configured."""


@dataclass(frozen=True, slots=True)
class AppContext:
    research: ResearchConfig
    config_digest: str
    game: LeducGame
    specs: dict[str, GameInfoSet]
    cfr: GameStrategy
    cfr_digest: str
    cfr_exploitability: float
    """Recomputed by the M3 exact evaluator at load time."""
    priors: Priors
    adapter: Adapter
    tree: CompiledTree
    selector: Selector

    @property
    def budget(self) -> float:
        """Exploitability limit of every strategy the AI may play (chips/hand)."""
        return self.cfr_exploitability + self.research.selection.budget_delta

    @property
    def candidate_stop(self) -> float:
        """The stop level of the candidate grid in the frozen evaluation (largest reported δ)."""
        return self.cfr_exploitability + max(self.research.selection.reported_deltas)

    @property
    def rule(self) -> SelectionRule:
        s = self.research.selection
        return SelectionRule(s.rule, s.z, s.inflation)

    @property
    def detector(self) -> DetectorConfig:
        cd = self.research.change_detection
        return DetectorConfig(short_decay=cd.short_decay, threshold=cd.threshold, drift=cd.drift)


def adaptation_settings(research: ResearchConfig) -> AdaptationSettings:
    """The prior-selection settings of the frozen reconstruction (only these are used)."""
    rec = research.reconstruction
    return AdaptationSettings(
        budget=research.selection.budget_delta,
        prior_strengths=rec.prior_strengths,
        default_prior=rec.default_prior,
        min_hands_for_selection=rec.min_hands_for_selection,
        posterior_samples=rec.posterior_samples,
    )


def development_priors(
    game: LeducGame, specs: dict[str, GameInfoSet], cfr: GameStrategy, research: ResearchConfig
) -> Priors:
    """The development-population priors of ``adaptive_study.build_world`` (no held-out data)."""
    population = build_population(
        game, cfr, research.world.per_family, research.world.switching_per_split
    )
    dev = [o for o in population["development"] if not o.is_switching]
    public_prior = PopulationPrior.from_opponents(
        [context_stats(game, cfr, o.policies[0]) for o in dev], "development stationary"
    )
    return build_priors(
        game, specs, cfr, [o.policies[0] for o in dev], public_prior, infoset_contexts(game)
    )


def load_app_context(
    config_path: Path = DEFAULT_CONFIG_PATH, hero_path: Path | None = None
) -> AppContext:
    try:
        research = load_research_config(config_path)
    except (OSError, ValueError, KeyError) as exc:
        raise ContextError(f"cannot load research configuration {config_path}: {exc}") from exc
    if research.game != "leduc_poker":
        raise ContextError(f"unsupported game {research.game!r}")
    game = LeducGame()
    path = hero_path if hero_path is not None else Path(research.hero.path)
    if not path.exists():
        raise ContextError(
            f"CFR hero {path} not found; train it with `pokermind train-cfr --game leduc "
            "--checkpoints 1000 10000 20000 50000 100000 --output-dir runs/m4-leduc-teacher`"
        )
    cfr = load_game_strategy(path, game.name, game.action_codes)
    digest = game_strategy_digest(cfr, game.action_codes)
    if digest != research.hero.digest:
        raise ContextError(f"hero digest {digest[:12]}… does not match the frozen configuration")
    specs = information_sets(game)
    expl = exploitability(game, cfr, specs)
    if abs(expl - research.hero.exploitability) > EXPLOITABILITY_MATCH:
        raise ContextError(f"hero exploitability {expl} differs from the frozen value")
    priors = development_priors(game, specs, cfr, research)
    tree = CompiledTree(game)
    return AppContext(
        research=research,
        config_digest=research.digest(),
        game=game,
        specs=specs,
        cfr=cfr,
        cfr_digest=digest,
        cfr_exploitability=expl,
        priors=priors,
        adapter=Adapter(game, specs, priors, adaptation_settings(research)),
        tree=tree,
        selector=Selector(game, specs, tree, cfr, expl),
    )


__all__ = [
    "AppContext",
    "ContextError",
    "adaptation_settings",
    "development_priors",
    "load_app_context",
]
