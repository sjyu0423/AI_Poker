"""The versioned PokerMind research configuration (M7B research freeze).

One JSON file, ``configs/pokermind-research-v1.json``, fixes every choice that
defines the adaptive Leduc baseline:
* the CFR hero and its digest;
* the opponent world;
* the reconstruction and posterior settings;
* the RNR solver;
* the exploitability budget and the selection rule;
* change detection.

The future interactive application must load this file rather than
re-declare the settings. :func:`load_research_config` validates the file and
recomputes its digest, so silent edits are detected.

The selection fields (``rule``, ``z``, ``inflation``, ``temperature``) were
chosen on development opponents only (``pokermind reliability calibrate``)
before any held-out evaluation. ``docs/research/research_freeze.md`` records
how.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

RESEARCH_NAME: Final = "pokermind-research"
RESEARCH_VERSION: Final = "1.0.0"
DEFAULT_CONFIG_PATH: Final = Path("configs/pokermind-research-v1.json")


@dataclass(frozen=True, slots=True)
class HeroConfig:
    path: str = "runs/m4-leduc-teacher/strategies/strategy_100000.json"
    digest: str = "3f1136d492159d56eab73efc2c9a365dfc930eb20d7ae13b105cc994e17874bf"
    """SHA-256 game-strategy digest (``pokermind.strategy.game_strategy_digest``)."""
    exploitability: float = 0.0029968965501324385
    iterations: int = 100_000


@dataclass(frozen=True, slots=True)
class WorldConfig:
    per_family: int = 4
    out_of_population_per_family: int = 2
    switching_per_split: int = 8
    bait_switch_count: int = 24


@dataclass(frozen=True, slots=True)
class ReconstructionConfig:
    prior_strengths: tuple[float, ...] = (2.0, 8.0)
    default_prior: tuple[str, float] = ("population", 2.0)
    min_hands_for_selection: int = 10
    posterior_samples: int = 16
    burn_in: int = 10
    thin: int = 2
    temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class ResponseConfig:
    method: str = "rnr"
    solver: str = "CFR+ (alternating, regrets floored, linear averaging)"
    iterations: int = 500
    p_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    budget_delta: float = 0.1
    """Allowed exploitability above the hero's (chips/hand)."""
    reported_deltas: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.25)
    rule: str = "point"
    z: float = 0.0
    inflation: float = 1.0
    tolerance: float = 1e-12
    certificate: str = "M3 recursive exact evaluator"


@dataclass(frozen=True, slots=True)
class ChangeDetectionConfig:
    short_decay: float = 0.8
    threshold: float = 5.0
    drift: float = 0.0
    retain: int = 10


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    name: str = RESEARCH_NAME
    version: str = RESEARCH_VERSION
    game: str = "leduc_poker"
    hero: HeroConfig = field(default_factory=HeroConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    change_detection: ChangeDetectionConfig = field(default_factory=ChangeDetectionConfig)
    provenance: dict[str, Any] = field(default_factory=dict)
    """How the frozen values were chosen (development-only evidence, not used to act)."""

    def __post_init__(self) -> None:
        if self.selection.rule not in ("point", "lcb"):
            raise ValueError(f"unknown selection rule {self.selection.rule!r}")
        if self.selection.budget_delta < 0 or self.selection.z < 0:
            raise ValueError("budget and z must be non-negative")
        if self.selection.inflation <= 0:
            raise ValueError("inflation must be positive")
        if not 0 < self.reconstruction.temperature <= 1:
            raise ValueError("temperature must be in (0, 1]")
        if self.response.method != "rnr" or self.response.iterations <= 0:
            raise ValueError("the frozen response is RNR with a positive iteration count")
        grid = self.response.p_grid
        if list(grid) != sorted(grid) or grid[0] != 0.0 or grid[-1] > 1.0:
            raise ValueError("p_grid must ascend from 0 to at most 1")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(asdict(self))
        return out

    def digest(self) -> str:
        """SHA-256 of the canonical JSON of every field except provenance."""
        body = self.to_dict()
        body.pop("provenance")
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def research_config_from_dict(data: dict[str, Any]) -> ResearchConfig:
    def tup(d: dict[str, Any], *keys: str) -> dict[str, Any]:
        out = dict(d)
        for k in keys:
            out[k] = tuple(out[k])
        return out

    rec = tup(data["reconstruction"], "prior_strengths", "default_prior")
    rec["default_prior"] = (str(rec["default_prior"][0]), float(rec["default_prior"][1]))
    return ResearchConfig(
        name=data["name"],
        version=data["version"],
        game=data["game"],
        hero=HeroConfig(**data["hero"]),
        world=WorldConfig(**data["world"]),
        reconstruction=ReconstructionConfig(**rec),
        response=ResponseConfig(**tup(data["response"], "p_grid")),
        selection=SelectionConfig(**tup(data["selection"], "reported_deltas")),
        change_detection=ChangeDetectionConfig(**data["change_detection"]),
        provenance=dict(data.get("provenance", {})),
    )


def save_research_config(config: ResearchConfig, path: Path) -> None:
    doc = config.to_dict()
    doc["digest"] = config.digest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")


def load_research_config(path: Path = DEFAULT_CONFIG_PATH) -> ResearchConfig:
    data = json.loads(path.read_text())
    recorded = data.pop("digest", None)
    config = research_config_from_dict(data)
    if recorded is not None and recorded != config.digest():
        raise ValueError(f"{path}: digest mismatch (file edited without re-freezing)")
    if config.name != RESEARCH_NAME:
        raise ValueError(f"{path}: not a {RESEARCH_NAME} configuration")
    return config


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "RESEARCH_NAME",
    "RESEARCH_VERSION",
    "ChangeDetectionConfig",
    "HeroConfig",
    "ReconstructionConfig",
    "ResearchConfig",
    "ResponseConfig",
    "SelectionConfig",
    "WorldConfig",
    "load_research_config",
    "research_config_from_dict",
    "save_research_config",
]
