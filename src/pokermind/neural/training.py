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
"""

from __future__ import annotations

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

    def __post_init__(self) -> None:
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
    checkpoints: dict[int, str] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cross_entropy": self.cross_entropy,
            "kl": self.kl,
            "max_abs_error": self.max_abs_error,
            "checkpoints": {str(k): v for k, v in self.checkpoints.items()},
            "seconds": self.seconds,
        }


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

        def record(epoch: int, log_q: torch.Tensor, ce: torch.Tensor) -> None:
            with torch.no_grad():
                history.epoch.append(epoch)
                history.cross_entropy.append(float(ce))
                history.kl.append(float(ce - h_p))
                history.max_abs_error.append(float((log_q.exp() - p).abs().max()))

        for epoch in range(training.epochs):
            log_q = masked_log_softmax(model(x), mask)  # 1. forward
            ce = -(p * log_q).sum(dim=-1).mean()  # 2. loss
            record(epoch, log_q, ce)
            optimizer.zero_grad(set_to_none=True)
            backward(ce)  # 3. backpropagation
            optimizer.step()  # 4. parameter update
            done = epoch + 1
            if checkpoint_dir is not None and done in training.checkpoint_epochs:
                path = checkpoint_dir / f"{policy.name}_seed{policy.seed}_epoch{done:06d}.pt"
                save_checkpoint(path, model, {"epoch": done, "teacher": dict(dataset.teacher)})
                history.checkpoints[done] = str(path)

        with torch.no_grad():
            log_q = masked_log_softmax(model(x), mask)
            record(training.epochs, log_q, -(p * log_q).sum(dim=-1).mean())
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
