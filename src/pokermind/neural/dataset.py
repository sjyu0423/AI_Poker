"""Supervised teacher dataset built from a CFR average strategy.

One example per information set, in sorted key order: 12 examples, no
duplicates. The 12 examples are the entire input space of Kuhn Poker, so this
dataset can measure how well a model *represents* the teacher, not how well
it generalizes (see ``docs/concepts/neural_policy.md`` §10).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pokermind.evaluation import validate_strategy
from pokermind.games.kuhn import InfoState, KuhnState, all_deals, initial_state
from pokermind.learning import train_cfr
from pokermind.neural.encoding import (
    ACTIONS,
    FEATURE_NAMES,
    NUM_ACTIONS,
    action_index,
    encode_info_state,
    legal_action_mask,
)
from pokermind.strategy import Strategy, load_strategy, strategy_digest


@dataclass(frozen=True, slots=True)
class TeacherExample:
    key: str
    info_state: InfoState
    features: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    target: npt.NDArray[np.float64]
    """Teacher probabilities over all 4 actions; 0 for illegal actions."""


@dataclass(frozen=True, slots=True)
class TeacherDataset:
    examples: tuple[TeacherExample, ...]
    teacher: Mapping[str, Any] = field(default_factory=dict)
    """Teacher provenance: digest, source, iterations."""

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(e.key for e in self.examples)

    def features(self) -> npt.NDArray[np.float32]:
        return np.stack([e.features for e in self.examples])

    def masks(self) -> npt.NDArray[np.bool_]:
        return np.stack([e.mask for e in self.examples])

    def targets(self) -> npt.NDArray[np.float64]:
        return np.stack([e.target for e in self.examples])

    def describe(self) -> dict[str, Any]:
        return {
            "num_examples": len(self.examples),
            "keys": list(self.keys),
            "feature_names": list(FEATURE_NAMES),
            "actions": [a.name for a in ACTIONS],
            "teacher": dict(self.teacher),
        }


def _walk(state: KuhnState) -> Iterator[KuhnState]:
    yield state
    for action in state.legal_actions():
        yield from _walk(state.apply(action))


def decision_info_states() -> dict[str, tuple[InfoState, tuple[Any, ...]]]:
    """Each information set's InfoState and legal actions, from the engine."""
    found: dict[str, tuple[InfoState, tuple[Any, ...]]] = {}
    for cards in all_deals():
        for state in _walk(initial_state(cards)):
            if state.is_terminal:
                continue
            info = state.information_state(state.current_player)
            found.setdefault(info.key, (info, state.legal_actions()))
    return dict(sorted(found.items()))


def build_teacher_dataset(
    strategy: Strategy, teacher_metadata: Mapping[str, Any] | None = None
) -> TeacherDataset:
    """One example per information set. The strategy is validated first."""
    validate_strategy(strategy)
    examples = []
    for key, (info, legal) in decision_info_states().items():
        target = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for action in legal:
            target[action_index(action)] = strategy[key][action]
        examples.append(
            TeacherExample(
                key=key,
                info_state=info,
                features=encode_info_state(info),
                mask=legal_action_mask(legal),
                target=target,
            )
        )
    metadata = {"digest": strategy_digest(strategy), **dict(teacher_metadata or {})}
    return TeacherDataset(examples=tuple(examples), teacher=metadata)


def load_teacher(path: Path | None = None, iterations: int = 50_000) -> TeacherDataset:
    """The teacher dataset from a strategy file, or by re-running M1's CFR.

    Vanilla CFR is deterministic, so ``iterations=50_000`` reproduces M1's
    reported strategy bit for bit (digest ``b8217a0c…``).
    """
    if path is not None:
        return build_teacher_dataset(load_strategy(path), {"source": str(path)})
    strategy = train_cfr(iterations).average_strategy()
    return build_teacher_dataset(
        strategy, {"source": "vanilla-cfr", "algorithm": "vanilla-cfr", "iterations": iterations}
    )
