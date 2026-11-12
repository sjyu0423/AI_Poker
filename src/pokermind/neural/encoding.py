"""Explicit numeric encoding of Kuhn Poker information states.

NumPy only (no PyTorch), so the encoding can be tested and reused on its own.

Features (7, float32):

    0-2  own card       one-hot J, Q, K
    3-6  public history one-hot "", "x", "b", "xb"

Actions (4) follow M0's ``Action`` enum order: CHECK, BET, CALL, FOLD. The
legal-action mask marks the legal ones.

The encoder accepts only an :class:`~pokermind.games.kuhn.InfoState`, which has
no field for the opponent's card, so hidden information can't reach a model.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from pokermind.games.kuhn import Action, Card, History, InfoState

ACTIONS: Final[tuple[Action, ...]] = tuple(Action)
DECISION_HISTORIES: Final[tuple[History, ...]] = (
    (),
    (Action.CHECK,),
    (Action.BET,),
    (Action.CHECK, Action.BET),
)
CARDS: Final[tuple[Card, ...]] = tuple(Card)

NUM_FEATURES: Final = len(CARDS) + len(DECISION_HISTORIES)
NUM_ACTIONS: Final = len(ACTIONS)
FEATURE_NAMES: Final[tuple[str, ...]] = (
    *(f"card={c.name}" for c in CARDS),
    *(f"history={''.join(a.code for a in h) or '-'}" for h in DECISION_HISTORIES),
)

_ACTION_INDEX: Final = {a: i for i, a in enumerate(ACTIONS)}
_HISTORY_INDEX: Final = {h: i for i, h in enumerate(DECISION_HISTORIES)}


def action_index(action: Action) -> int:
    return _ACTION_INDEX[action]


def encode_info_state(info_state: InfoState) -> npt.NDArray[np.float32]:
    """The 7-dimensional feature vector of an information state."""
    if not isinstance(info_state, InfoState):
        raise TypeError(f"expected InfoState, got {type(info_state).__name__}")
    try:
        history_idx = _HISTORY_INDEX[info_state.history]
    except KeyError:
        raise ValueError(f"not a decision history: {info_state.key!r}") from None
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    features[CARDS.index(info_state.private_card)] = 1.0
    features[len(CARDS) + history_idx] = 1.0
    return features


def legal_action_mask(legal_actions: tuple[Action, ...]) -> npt.NDArray[np.bool_]:
    """Boolean mask over :data:`ACTIONS`; True for legal actions."""
    if not legal_actions:
        raise ValueError("no legal actions")
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for action in legal_actions:
        mask[_ACTION_INDEX[action]] = True
    return mask
