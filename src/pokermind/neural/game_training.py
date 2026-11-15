"""Supervised distillation with a validation set (M4).

The M2 procedure (``neural/training.py``) on any ``(features, mask, target)``
arrays: full-batch Adam on the masked soft-target cross-entropy
``H(p, q_θ) = -mean Σ_a p log q_θ``, cosine learning-rate decay and
best-checkpoint selection. The difference: when a validation set is given,
the checkpoint is selected by the **validation** KL, so selection never
looks at test data and does not reward memorizing the training set. Without
a validation set it falls back to the training KL, exactly as M2B.

Training is on the CPU in float32 with deterministic algorithms, fixed data
order and seeded initialization, so a configuration reproduces bit for bit.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from pokermind.neural.game_model import GamePolicyConfig, GamePolicyNetwork
from pokermind.neural.model import masked_log_softmax
from pokermind.neural.training import (
    TrainingConfig,
    backward,
    learning_rate_at,
    teacher_entropy,
)


@dataclass(frozen=True, slots=True)
class Arrays:
    features: npt.NDArray[np.float32]
    masks: npt.NDArray[np.bool_]
    targets: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        n = len(self.features)
        if len(self.masks) != n or len(self.targets) != n or n == 0:
            raise ValueError("features, masks and targets need the same non-zero length")
        if bool((self.targets[~self.masks] != 0).any()):
            raise ValueError("targets put probability on illegal actions")
        if not np.allclose(self.targets.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("targets must sum to 1")

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.ascontiguousarray(self.features)),
            torch.from_numpy(np.ascontiguousarray(self.masks)),
            torch.from_numpy(np.ascontiguousarray(self.targets)).float(),
        )


@dataclass(slots=True)
class GameTrainingHistory:
    """Per-epoch metrics, recorded before that epoch's update (epoch 0 = init)."""

    epoch: list[int] = field(default_factory=list)
    train_cross_entropy: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    validation_kl: list[float] = field(default_factory=list)
    selected_epoch: int = 0
    selection: str = "train_kl"
    seconds: float = 0.0

    def to_dict(self, every: int = 1) -> dict[str, Any]:
        sl = slice(None, None, every)
        return {
            "epoch": self.epoch[sl],
            "train_cross_entropy": self.train_cross_entropy[sl],
            "train_kl": self.train_kl[sl],
            "validation_kl": self.validation_kl[sl],
            "selected_epoch": self.selected_epoch,
            "selection": self.selection,
            "seconds": self.seconds,
        }


def train_game_policy(
    train: Arrays,
    policy: GamePolicyConfig,
    training: TrainingConfig,
    validation: Arrays | None = None,
    weight_decay: float = 0.0,
) -> tuple[GamePolicyNetwork, GameTrainingHistory]:
    """Train; ``weight_decay`` is Adam's L2 penalty (0 = the M2B procedure)."""
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if training.checkpoint_epochs or training.early_stopping_patience is not None:
        raise ValueError("checkpoint_epochs and early stopping are not supported here")
    x, mask, p = train.tensors()
    h_p = teacher_entropy(p)
    val = validation.tensors() if validation is not None else None
    h_val = teacher_entropy(val[2]) if val is not None else None

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = GamePolicyNetwork(policy)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=training.learning_rate, weight_decay=weight_decay
        )
        history = GameTrainingHistory(selection="validation_kl" if val else "train_kl")
        best: tuple[float, int, dict[str, torch.Tensor]] | None = None
        start = time.perf_counter()

        def record(epoch: int, ce: torch.Tensor) -> None:
            nonlocal best
            with torch.no_grad():
                train_kl = float(ce - h_p)
                if val is not None and h_val is not None:
                    log_qv = masked_log_softmax(model(val[0]), val[1])
                    val_kl = float(-(val[2] * log_qv).sum(dim=-1).mean() - h_val)
                else:
                    val_kl = math.nan
            history.epoch.append(epoch)
            history.train_cross_entropy.append(float(ce))
            history.train_kl.append(train_kl)
            history.validation_kl.append(val_kl)
            score = val_kl if val is not None else train_kl
            if training.keep_best and (best is None or score < best[0]):
                best = (score, epoch, copy.deepcopy(model.state_dict()))

        for epoch in range(training.epochs):
            log_q = masked_log_softmax(model(x), mask)
            ce = -(p * log_q).sum(dim=-1).mean()
            record(epoch, ce.detach())
            lr = learning_rate_at(training, epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            backward(ce)
            optimizer.step()
        with torch.no_grad():
            log_q = masked_log_softmax(model(x), mask)
            record(training.epochs, -(p * log_q).sum(dim=-1).mean())
        history.selected_epoch = training.epochs
        if best is not None:
            history.selected_epoch = best[1]
            model.load_state_dict(best[2])
        history.seconds = time.perf_counter() - start
    finally:
        torch.use_deterministic_algorithms(previous)
    model.eval()
    return model, history
