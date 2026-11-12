"""Supervised policy distillation: fit a network to the teacher's distributions.

Each epoch is one full-batch step over all 12 information sets
(``docs/concepts/neural_policy.md`` §5-7):

1. forward pass: logits ``z = f_θ(x)``, masked log-softmax ``log q_θ``;
2. loss: soft-target cross-entropy ``H(p, q_θ) = -mean_I Σ_a p log q_θ``,
   which has the same gradient as KL(p ‖ q_θ);
3. backward pass: autograd fills ``∂H/∂θ`` (backpropagation);
4. Adam step: ``θ ← θ - η m̂ / (√v̂ + ε)``.

Training runs on the CPU in float32. The data has a fixed order, there is no
shuffling, and initialization is seeded, so a configuration always produces
bit-identical weights.

Stability options (M2B; all off by default, which reproduces M2A exactly):

* ``lr_schedule="cosine"``: ``η_t = η_min + ½(η − η_min)(1 + cos(π t / T))``
  with ``η_min = min_lr_factor · η``. Smaller late steps stop Adam from
  overshooting once the logits are large.
* ``keep_best=True``: return the parameters with the lowest recorded KL
  instead of the last ones (best-checkpoint selection).
* ``early_stopping_patience=P``: stop when the KL has not improved by a
  factor ``(1 - min_relative_improvement)`` for ``P`` epochs.

Kuhn Poker has no held-out data (all 12 information sets are the training
set), so "best" and "early stopping" are judged on the training KL. Here that
is legitimate because the goal is to *represent* the teacher, but it is not a
validation procedure and says nothing about generalization.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pokermind.neural.dataset import TeacherDataset
from pokermind.neural.model import (
    PolicyConfig,
    PolicyNetwork,
    masked_log_softmax,
    save_checkpoint,
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 3_000
    learning_rate: float = 0.03
    checkpoint_epochs: tuple[int, ...] = ()
    """Epochs at which to save a checkpoint (requires a checkpoint directory)."""
    lr_schedule: str = "constant"
    """``"constant"`` (M2A) or ``"cosine"``."""
    min_lr_factor: float = 0.01
    """Cosine schedule only: final learning rate as a fraction of the initial one."""
    keep_best: bool = False
    early_stopping_patience: int | None = None
    min_relative_improvement: float = 0.01

    def __post_init__(self) -> None:
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}"
            )
        if not 0 < self.min_lr_factor <= 1:
            raise ValueError(f"min_lr_factor must be in (0, 1], got {self.min_lr_factor}")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("min_relative_improvement must be in [0, 1)")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if any(not 0 < e <= self.epochs for e in self.checkpoint_epochs):
            raise ValueError(f"checkpoint epochs must be in [1, epochs]: {self.checkpoint_epochs}")


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch metrics, recorded *before* that epoch's update (epoch 0 = init)."""

    epoch: list[int] = field(default_factory=list)
    cross_entropy: list[float] = field(default_factory=list)
    kl: list[float] = field(default_factory=list)
    max_abs_error: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    """Learning rate used for the update that follows each record (nan after the last)."""
    checkpoints: dict[int, str] = field(default_factory=dict)
    seconds: float = 0.0
    selected_epoch: int = 0
    """Number of updates applied to the returned parameters."""
    stopped_early: bool = False

    @property
    def selected_kl(self) -> float:
        return self.kl[self.epoch.index(self.selected_epoch)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cross_entropy": self.cross_entropy,
            "kl": self.kl,
            "max_abs_error": self.max_abs_error,
            "learning_rate": self.learning_rate,
            "checkpoints": {str(k): v for k, v in self.checkpoints.items()},
            "seconds": self.seconds,
            "selected_epoch": self.selected_epoch,
            "stopped_early": self.stopped_early,
        }


def learning_rate_at(training: TrainingConfig, epoch: int) -> float:
    """Learning rate for update number ``epoch`` (0-based)."""
    if training.lr_schedule == "constant":
        return training.learning_rate
    lr, lr_min = training.learning_rate, training.learning_rate * training.min_lr_factor
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * epoch / training.epochs))


def backward(loss: torch.Tensor) -> None:
    """``loss.backward()``: backpropagation fills ``param.grad`` for every parameter.

    A typed wrapper: PyTorch 2.15's stubs leave ``Tensor.backward`` untyped.
    """
    loss.backward()  # type: ignore[no-untyped-call]


def teacher_entropy(targets: torch.Tensor) -> torch.Tensor:
    """Mean entropy ``H(p)`` of the targets, with ``0 log 0 = 0``."""
    safe = torch.where(targets > 0, targets, torch.ones_like(targets))
    return -(targets * torch.log(safe)).sum(dim=-1).mean()


def train_policy(
    dataset: TeacherDataset,
    policy: PolicyConfig,
    training: TrainingConfig,
    checkpoint_dir: Path | None = None,
) -> tuple[PolicyNetwork, TrainingHistory]:
    if training.checkpoint_epochs and checkpoint_dir is None:
        raise ValueError("checkpoint_epochs requires checkpoint_dir")
    x = torch.from_numpy(dataset.features())
    mask = torch.from_numpy(dataset.masks())
    p = torch.from_numpy(dataset.targets()).float()
    h_p = teacher_entropy(p)

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = PolicyNetwork(policy)
        optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
        history = TrainingHistory()
        start = time.perf_counter()

        best_kl, best_state = math.inf, None
        es_best, es_epoch = math.inf, 0

        def record(epoch: int, log_q: torch.Tensor, ce: torch.Tensor) -> float:
            nonlocal best_kl, best_state
            with torch.no_grad():
                kl = float(ce - h_p)
                history.epoch.append(epoch)
                history.cross_entropy.append(float(ce))
                history.kl.append(kl)
                history.max_abs_error.append(float((log_q.exp() - p).abs().max()))
            if training.keep_best and kl < best_kl:
                best_kl, best_state = kl, (epoch, copy.deepcopy(model.state_dict()))
            return kl

        updates = 0
        for epoch in range(training.epochs):
            log_q = masked_log_softmax(model(x), mask)  # 1. forward
            ce = -(p * log_q).sum(dim=-1).mean()  # 2. loss
            kl = record(epoch, log_q, ce)
            if training.early_stopping_patience is not None:
                if kl < es_best * (1 - training.min_relative_improvement):
                    es_best, es_epoch = kl, epoch
                elif epoch - es_epoch >= training.early_stopping_patience:
                    history.stopped_early = True
                    history.learning_rate.append(math.nan)
                    break
            lr = learning_rate_at(training, epoch)
            history.learning_rate.append(lr)
            if training.lr_schedule != "constant":
                for group in optimizer.param_groups:
                    group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            backward(ce)  # 3. backpropagation
            optimizer.step()  # 4. parameter update
            updates = epoch + 1
            done = epoch + 1
            if checkpoint_dir is not None and done in training.checkpoint_epochs:
                path = checkpoint_dir / f"{policy.name}_seed{policy.seed}_epoch{done:06d}.pt"
                save_checkpoint(path, model, {"epoch": done, "teacher": dict(dataset.teacher)})
                history.checkpoints[done] = str(path)

        if not history.stopped_early:
            with torch.no_grad():
                log_q = masked_log_softmax(model(x), mask)
            record(updates, log_q, -(p * log_q).sum(dim=-1).mean())
            history.learning_rate.append(math.nan)
        history.selected_epoch = history.epoch[-1]
        if training.keep_best and best_state is not None:
            history.selected_epoch = best_state[0]
            model.load_state_dict(best_state[1])
        history.seconds = time.perf_counter() - start
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
    model.eval()
    return model, history


def state_dict_digest(model: PolicyNetwork) -> str:
    """SHA-256 over the raw parameter bytes, in state-dict order."""
    import hashlib

    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return h.hexdigest()
