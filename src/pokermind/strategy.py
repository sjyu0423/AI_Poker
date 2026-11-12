"""Tabular strategies: the data format shared by learning, evaluation and agents.

A :data:`Strategy` maps an information-set key (``InfoState.key``, e.g.
``"Q|xb"``) to a probability distribution over the legal actions at that
information set. A single table can cover both players, because Kuhn Poker
information-set keys never overlap between seats.

This module only defines the type and its JSON file format. It depends on
the game's public types and nothing else. Validating a strategy against the
game tree belongs to :mod:`pokermind.evaluation.exact`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from pokermind.games.kuhn import Action

Strategy = Mapping[str, Mapping[Action, float]]
"""Information-set key -> {legal action: probability}."""

STRATEGY_FORMAT: Final = "pokermind.strategy.v1"


class StrategyFormatError(ValueError):
    """Raised when a strategy file cannot be parsed."""


def strategy_to_jsonable(strategy: Strategy) -> dict[str, dict[str, float]]:
    """Plain-JSON form: keys sorted, actions written as one-letter codes."""
    return {
        key: {action.code: float(prob) for action, prob in strategy[key].items()}
        for key in sorted(strategy)
    }


def strategy_from_jsonable(data: object) -> dict[str, dict[Action, float]]:
    """Inverse of :func:`strategy_to_jsonable`. Checks structure, not probabilities."""
    if not isinstance(data, dict):
        raise StrategyFormatError("strategy must be a JSON object")
    by_code = {a.code: a for a in Action}
    result: dict[str, dict[Action, float]] = {}
    for key, dist in data.items():
        if not isinstance(key, str) or not isinstance(dist, dict):
            raise StrategyFormatError(f"malformed entry for {key!r}")
        parsed: dict[Action, float] = {}
        for code, prob in dist.items():
            if code not in by_code:
                raise StrategyFormatError(f"unknown action code {code!r} at {key!r}")
            if isinstance(prob, bool) or not isinstance(prob, int | float):
                raise StrategyFormatError(f"probability must be a number at {key!r}/{code}")
            parsed[by_code[code]] = float(prob)
        result[key] = parsed
    return result


def strategy_digest(strategy: Strategy) -> str:
    """SHA-256 of the canonical JSON form. Equal digests mean bit-identical tables."""
    canonical = json.dumps(strategy_to_jsonable(strategy), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_strategy(
    path: Path, strategy: Strategy, metadata: Mapping[str, Any] | None = None
) -> None:
    """Write a strategy file. Python's float repr round-trips exactly through JSON."""
    document = {
        "format": STRATEGY_FORMAT,
        "game": "kuhn",
        "digest": strategy_digest(strategy),
        "metadata": dict(metadata or {}),
        "strategy": strategy_to_jsonable(strategy),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


def load_strategy(path: Path) -> dict[str, dict[Action, float]]:
    """Read a file written by :func:`save_strategy` and check its format and digest."""
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise StrategyFormatError(f"{path}: not valid JSON ({exc})") from None
    if not isinstance(document, dict) or document.get("format") != STRATEGY_FORMAT:
        raise StrategyFormatError(f"{path}: expected format {STRATEGY_FORMAT!r}")
    if document.get("game") != "kuhn":
        raise StrategyFormatError(f"{path}: expected game 'kuhn', got {document.get('game')!r}")
    strategy = strategy_from_jsonable(document.get("strategy"))
    if document.get("digest") != strategy_digest(strategy):
        raise StrategyFormatError(f"{path}: digest mismatch (file edited or corrupted)")
    return strategy
