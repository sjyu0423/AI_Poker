"""Leduc teacher dataset and deterministic generalization splits (M4).

**Examples.** One example per *canonical* (suit-free) information state: 288
examples covering all 936 information sets. Each example lists the
suit-specific keys it stands for. Its target is the teacher's distribution,
which must agree across those keys (CFR on Leduc is exactly suit-symmetric;
the builder checks this).

**Splits.** Every split partitions the 288 canonical states into ``train``,
``validation`` and ``test``. The unit of splitting is the canonical state, so
two information sets with the same feature vector can never be on different
sides. ``validation`` is drawn at random from the non-test states and is the
only data used for checkpoint selection; ``test`` is never used for training
or selection.

* ``full``: all 288 states for training, no validation or test (the
  representation question; checkpoint selected on the training KL, as in M2B).
* ``random``: a seeded 20 % of the states held out.
* ``public``: every round-2 state whose (own rank, public rank) combination is
  (Q, Q) or (J, K): one pair and one non-pair combination never seen.
* ``history``: every state whose round-1 betting starts with raise-reraise
  (``rr``, ``rrc``): a betting line never seen in either round.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from pokermind.evaluation.game_exact import validate_game_strategy
from pokermind.games.base import GameState
from pokermind.games.leduc import RANK_NAMES, LeducGame, LeducObservation, LeducState, rank
from pokermind.neural.leduc_encoding import (
    BINARY_FEATURES,
    NUM_ACTIONS,
    canonical_key,
    encode_observation,
    legal_action_mask,
)
from pokermind.strategy import GameStrategy

SPLIT_NAMES: Final = ("full", "random", "public", "history")
HELD_OUT_PUBLIC_COMBOS: Final = (("Q", "Q"), ("J", "K"))
"""(own rank, public rank) combinations held out by the ``public`` split."""
HELD_OUT_ROUND1_PREFIX: Final = "rr"
TEST_FRACTION: Final = 0.2
VALIDATION_FRACTION: Final = 0.1
SYMMETRY_TOLERANCE: Final = 1e-12


@dataclass(frozen=True, slots=True)
class LeducExample:
    key: str
    """Canonical key, e.g. ``"K|J|rrc|c"``."""
    members: tuple[str, ...]
    """Suit-specific information-set keys with this canonical state."""
    observation: LeducObservation
    """One representative observation (the first member)."""
    legal_actions: tuple[int, ...]
    features: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    target: npt.NDArray[np.float64]
    """Teacher probabilities over (fold, call, raise); 0 for illegal actions."""

    @property
    def round(self) -> int:
        return 1 if self.observation.public_card is None else 2

    @property
    def own_rank(self) -> str:
        return RANK_NAMES[rank(self.observation.private_card)]

    @property
    def public_rank(self) -> str | None:
        card = self.observation.public_card
        return None if card is None else RANK_NAMES[rank(card)]

    @property
    def histories(self) -> tuple[str, str | None]:
        parts = self.key.split("|")
        return (parts[1], None) if len(parts) == 2 else (parts[2], parts[3])


@dataclass(frozen=True, slots=True)
class LeducDataset:
    examples: tuple[LeducExample, ...]
    teacher: Mapping[str, Any] = field(default_factory=dict)
    max_symmetry_deviation: float = 0.0
    """Largest difference between the teacher's probabilities on suit twins."""

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(e.key for e in self.examples)

    def features(self, idx: Sequence[int] | None = None) -> npt.NDArray[np.float32]:
        return np.stack([self.examples[i].features for i in self._idx(idx)])

    def masks(self, idx: Sequence[int] | None = None) -> npt.NDArray[np.bool_]:
        return np.stack([self.examples[i].mask for i in self._idx(idx)])

    def targets(self, idx: Sequence[int] | None = None) -> npt.NDArray[np.float64]:
        return np.stack([self.examples[i].target for i in self._idx(idx)])

    def _idx(self, idx: Sequence[int] | None) -> Sequence[int]:
        return range(len(self.examples)) if idx is None else idx

    def expand(self, table: Mapping[str, Mapping[int, float]]) -> dict[str, dict[int, float]]:
        """A canonical-key table as a full 936-key strategy."""
        return {m: dict(table[e.key]) for e in self.examples for m in e.members}


def _walk(state: GameState) -> Iterator[GameState]:
    yield state
    if state.is_terminal:
        return
    moves = (
        [o for o, _ in state.chance_outcomes()] if state.is_chance_node else state.legal_actions()
    )
    for move in moves:
        yield from _walk(state.apply(move))


def canonical_states(
    game: LeducGame | None = None,
) -> dict[str, tuple[LeducObservation, tuple[int, ...], tuple[str, ...]]]:
    """Canonical key -> (representative observation, legal actions, member keys)."""
    g = game or LeducGame()
    found: dict[str, tuple[LeducObservation, tuple[int, ...], list[str]]] = {}
    for state in _walk(g.initial_state()):
        if state.is_terminal or state.is_chance_node:
            continue
        assert isinstance(state, LeducState)
        obs = state.observation(state.current_player)
        ckey = canonical_key(obs)
        entry = found.setdefault(ckey, (obs, state.legal_actions(), []))
        if entry[1] != state.legal_actions():
            raise AssertionError(f"legal actions differ within {ckey!r}")
        if obs.key not in entry[2]:
            entry[2].append(obs.key)
    return {k: (o, a, tuple(sorted(m))) for k, (o, a, m) in sorted(found.items())}


def build_leduc_dataset(
    strategy: GameStrategy,
    teacher_metadata: Mapping[str, Any] | None = None,
    game: LeducGame | None = None,
    symmetry_tolerance: float = SYMMETRY_TOLERANCE,
) -> LeducDataset:
    """One example per canonical state. The strategy is validated first."""
    g = game or LeducGame()
    validate_game_strategy(g, strategy)
    examples = []
    deviation = 0.0
    for ckey, (obs, legal, members) in canonical_states(g).items():
        rows = np.array([[strategy[m].get(a, 0.0) for a in range(NUM_ACTIONS)] for m in members])
        deviation = max(deviation, float((rows.max(axis=0) - rows.min(axis=0)).max()))
        examples.append(
            LeducExample(
                key=ckey,
                members=members,
                observation=obs,
                legal_actions=legal,
                features=encode_observation(obs, g.config),
                mask=legal_action_mask(legal),
                target=rows.mean(axis=0),
            )
        )
    if deviation > symmetry_tolerance:
        raise ValueError(
            f"teacher is not suit-symmetric: twins differ by {deviation:.3g} "
            f"(tolerance {symmetry_tolerance:g})"
        )
    return LeducDataset(tuple(examples), dict(teacher_metadata or {}), deviation)


# -- splits -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Split:
    name: str
    description: str
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        parts = (set(self.train), set(self.validation), set(self.test))
        if any(
            len(p) != len(t)
            for p, t in zip(parts, (self.train, self.validation, self.test), strict=True)
        ):
            raise ValueError(f"{self.name}: duplicate indices")
        for a, b in itertools.combinations(parts, 2):
            if a & b:
                raise ValueError(f"{self.name}: train/validation/test overlap")
        if not self.train:
            raise ValueError(f"{self.name}: empty training set")


def _validation_from(pool: Sequence[int], seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    order = [pool[i] for i in rng.permutation(len(pool))]
    n_val = round(VALIDATION_FRACTION * len(pool))
    return tuple(sorted(order[n_val:])), tuple(sorted(order[:n_val]))


def make_split(dataset: LeducDataset, name: str, seed: int = 0) -> Split:
    """Deterministic split ``name`` (see the module docstring)."""
    n = len(dataset)
    examples = dataset.examples
    if name == "full":
        return Split(name, "all 288 canonical states for training", tuple(range(n)), (), ())
    if name == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        test = tuple(sorted(int(i) for i in order[: round(TEST_FRACTION * n)]))
        description = f"seeded random {TEST_FRACTION:.0%} of canonical states (seed {seed})"
    elif name == "public":
        test = tuple(
            i
            for i, e in enumerate(examples)
            if e.round == 2 and (e.own_rank, e.public_rank) in HELD_OUT_PUBLIC_COMBOS
        )
        combos = ", ".join(f"({o},{p})" for o, p in HELD_OUT_PUBLIC_COMBOS)
        description = f"round-2 states with (own, public) rank in {combos}"
    elif name == "history":
        test = tuple(
            i for i, e in enumerate(examples) if e.histories[0].startswith(HELD_OUT_ROUND1_PREFIX)
        )
        description = f"states whose round-1 betting starts with {HELD_OUT_ROUND1_PREFIX!r}"
    else:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
    held = set(test)
    train, validation = _validation_from([i for i in range(n) if i not in held], seed + 1)
    return Split(name, description, train, validation, test)


def split_coverage(dataset: LeducDataset, split: Split) -> dict[str, Any]:
    """Sizes, overlap checks and how out-of-distribution the test states are."""
    ex = dataset.examples
    seen = set(split.train) | set(split.validation)
    feats = dataset.features()
    fit = feats[sorted(seen)]
    fit_rows = {r.tobytes() for r in fit}
    active_fit = (fit[:, list(BINARY_FEATURES)] > 0).any(axis=0)

    def by_round(idx: Sequence[int]) -> dict[str, int]:
        return {f"round{r}": sum(1 for i in idx if ex[i].round == r) for r in (1, 2)}

    combos = {(ex[i].own_rank, ex[i].public_rank) for i in seen}
    hists = {ex[i].histories for i in seen}
    r1_hists = {ex[i].histories[0] for i in seen}
    test_rows = []
    for i in split.test:
        unseen = [
            int(BINARY_FEATURES[j])
            for j in range(len(BINARY_FEATURES))
            if feats[i, BINARY_FEATURES[j]] > 0 and not active_fit[j]
        ]
        hamming = int(np.min((feats[sorted(seen)] != feats[i]).sum(axis=1)))
        test_rows.append(
            {
                "key": ex[i].key,
                "identical_features_in_fit": feats[i].tobytes() in fit_rows,
                "unseen_active_features": unseen,
                "card_combo_seen": (ex[i].own_rank, ex[i].public_rank) in combos,
                "history_seen": ex[i].histories in hists,
                "round1_history_seen": ex[i].histories[0] in r1_hists,
                "nearest_fit_hamming": hamming,
            }
        )
    n_test = len(test_rows)

    def frac(field_name: str) -> float:
        return sum(bool(r[field_name]) for r in test_rows) / n_test if n_test else 0.0

    members = sum(len(ex[i].members) for i in split.test)
    return {
        "name": split.name,
        "description": split.description,
        "sizes": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "test_information_sets": members,
        },
        "by_round": {
            "train": by_round(split.train),
            "validation": by_round(split.validation),
            "test": by_round(split.test),
        },
        "overlap": {
            "train_validation": len(set(split.train) & set(split.validation)),
            "train_test": len(set(split.train) & set(split.test)),
            "validation_test": len(set(split.validation) & set(split.test)),
            "test_feature_vectors_in_fit": sum(r["identical_features_in_fit"] for r in test_rows),
        },
        "ood": {
            "test_with_unseen_active_feature": frac("unseen_active_features"),
            "test_card_combo_seen": frac("card_combo_seen"),
            "test_history_seen": frac("history_seen"),
            "test_round1_history_seen": frac("round1_history_seen"),
            "mean_nearest_fit_hamming": (
                float(np.mean([r["nearest_fit_hamming"] for r in test_rows])) if n_test else 0.0
            ),
        },
        "test_states": test_rows,
    }


__all__ = [
    "HELD_OUT_PUBLIC_COMBOS",
    "HELD_OUT_ROUND1_PREFIX",
    "SPLIT_NAMES",
    "LeducDataset",
    "LeducExample",
    "Split",
    "build_leduc_dataset",
    "canonical_states",
    "make_split",
    "split_coverage",
]
