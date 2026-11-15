"""Numeric encoding of Leduc Hold'em observations (M4).

NumPy only. The encoder accepts a :class:`~pokermind.games.leduc.LeducObservation`,
which has no field for the opponent's private card, so hidden information
cannot reach a model (``docs/design/leduc_encoding.md``).

Features (30, float32):

    0-2    own rank              one-hot J, Q, K
    3-5    public rank           one-hot J, Q, K (all zero before the flop)
    6      public hidden         1 in round 1
    7      pair                  own rank == public rank
    8      round 2               1 in round 2
    9-16   round-1 actions       position 0..3 x {call, raise}
    17-24  round-2 actions       position 0..3 x {call, raise}
    25     acting seat           1 if seat 1 is to act
    26     facing a bet          contributions differ (fold is legal)
    27     raises this round     / max_raises
    28     own contribution      / maximum contribution
    29     opponent contribution / maximum contribution

**Suits are deliberately dropped.** Leduc's rules depend on ranks only, so
the two suits are strategically identical: the encoding maps the 936
suit-specific information sets onto 288 *canonical* (rank-only) states, and
it is injective on those. Features 7 and 25-29 are functions of the others;
they are given explicitly so that small models don't have to learn them.

Actions (3): fold, call, raise, as in the engine. Fold never appears inside a
non-terminal history, so the history slots need only call and raise.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from pokermind.games.leduc import (
    ACTION_NAMES,
    CALL,
    FOLD,
    RAISE,
    RANK_NAMES,
    LeducConfig,
    LeducObservation,
    rank,
)

NUM_ACTIONS: Final = len(ACTION_NAMES)
MAX_ROUND_ACTIONS: Final = 4
"""Longest betting sequence in one round: call, raise, raise, call."""

_HISTORY_CODES: Final = ("c", "r")
FEATURE_NAMES: Final[tuple[str, ...]] = (
    *(f"own={r}" for r in RANK_NAMES),
    *(f"public={r}" for r in RANK_NAMES),
    "public=hidden",
    "pair",
    "round=2",
    *(
        f"r{rnd}.a{pos}={code}"
        for rnd in (1, 2)
        for pos in range(MAX_ROUND_ACTIONS)
        for code in _HISTORY_CODES
    ),
    "acting_seat=1",
    "facing_bet",
    "raises_this_round",
    "own_contribution",
    "opponent_contribution",
)
NUM_FEATURES: Final = len(FEATURE_NAMES)
BINARY_FEATURES: Final = tuple(range(27))
"""Indices of the 0/1 features (everything before the three scaled counts)."""

_OWN, _PUBLIC, _HIDDEN, _PAIR, _ROUND2 = 0, 3, 6, 7, 8
_HISTORY = (9, 17)
_SEAT, _FACING, _RAISES, _OWN_CONTRIB, _OPP_CONTRIB = 25, 26, 27, 28, 29


def max_contribution(config: LeducConfig) -> int:
    """Largest possible contribution of one player (13 with the defaults)."""
    return config.ante + config.max_raises * sum(config.raise_sizes)


def replay_public_betting(
    rounds: tuple[tuple[int, ...], ...], config: LeducConfig
) -> tuple[tuple[int, int], int, int]:
    """Contributions, raises in the current round and the seat to act.

    Uses only the public action history (player 0 opens each round).
    """
    if not 1 <= len(rounds) <= 2:
        raise ValueError(f"expected 1 or 2 rounds, got {len(rounds)}")
    contrib = [config.ante, config.ante]
    raises = 0
    for rnd, actions in enumerate(rounds):
        raises = 0
        if len(actions) > MAX_ROUND_ACTIONS:
            raise ValueError(f"round {rnd + 1} has too many actions: {actions}")
        for i, action in enumerate(actions):
            player = i % 2
            if action == RAISE:
                if raises >= config.max_raises:
                    raise ValueError(f"too many raises in round {rnd + 1}: {actions}")
                contrib[player] = max(contrib) + config.raise_sizes[rnd]
                raises += 1
            elif action == CALL:
                contrib[player] = max(contrib)
            elif action == FOLD:
                raise ValueError("a fold ends the hand; it can't be in a decision history")
            else:
                raise ValueError(f"unknown action {action!r}")
    return (contrib[0], contrib[1]), raises, len(rounds[-1]) % 2


def encode_observation(
    observation: LeducObservation, config: LeducConfig | None = None
) -> npt.NDArray[np.float32]:
    """The 30-dimensional feature vector of a decision-point observation."""
    if not isinstance(observation, LeducObservation):
        raise TypeError(f"expected LeducObservation, got {type(observation).__name__}")
    cfg = config or LeducConfig()
    rounds = observation.rounds
    in_round2 = observation.public_card is not None
    if in_round2 != (len(rounds) == 2):
        raise ValueError(f"public card and number of rounds disagree: {observation!r}")
    contrib, raises, seat = replay_public_betting(rounds, cfg)
    if seat != observation.player:
        raise ValueError(f"player {observation.player} is not the one to act: {observation!r}")

    x = np.zeros(NUM_FEATURES, dtype=np.float32)
    own = rank(observation.private_card)
    x[_OWN + own] = 1.0
    if observation.public_card is None:
        x[_HIDDEN] = 1.0
    else:
        public = rank(observation.public_card)
        x[_PUBLIC + public] = 1.0
        x[_PAIR] = float(own == public)
        x[_ROUND2] = 1.0
    for rnd, actions in enumerate(rounds):
        for pos, action in enumerate(actions):
            x[_HISTORY[rnd] + 2 * pos + _HISTORY_CODES.index("cr"[action - CALL])] = 1.0
    top = max_contribution(cfg)
    x[_SEAT] = float(seat)
    x[_FACING] = float(contrib[0] != contrib[1])
    x[_RAISES] = raises / cfg.max_raises
    x[_OWN_CONTRIB] = contrib[seat] / top
    x[_OPP_CONTRIB] = contrib[1 - seat] / top
    return x


def legal_action_mask(legal_actions: tuple[int, ...]) -> npt.NDArray[np.bool_]:
    """Boolean mask over (fold, call, raise); True for legal actions."""
    if not legal_actions:
        raise ValueError("no legal actions")
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for action in legal_actions:
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        mask[action] = True
    return mask


def canonical_key(observation: LeducObservation) -> str:
    """Suit-free key: ``"K|cr"`` in round 1, ``"K|J|rrc|c"`` in round 2."""
    parts = observation.key.split("|")
    parts[0] = parts[0][0]
    if observation.public_card is not None:
        parts[1] = parts[1][0]
    return "|".join(parts)
