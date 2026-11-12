"""Tiny MLP policies with legal-action masking (PyTorch).

Architecture (``docs/concepts/neural_policy.md`` §3-4)::

    x (7) -> [Linear -> tanh] * len(hidden_sizes) -> Linear -> logits (4)
    q(a|x) = softmax over legal actions (illegal logits masked out)

``hidden_sizes=()`` gives a linear (multinomial-logistic) policy.
"""

from __future__ import annotations

import io
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from pokermind.games.kuhn import Action
from pokermind.neural.encoding import (
    ACTIONS,
    NUM_ACTIONS,
    NUM_FEATURES,
    encode_info_state,
    legal_action_mask,
)

CHECKPOINT_FORMAT: Final = "pokermind.neural-policy.v1"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    hidden_sizes: tuple[int, ...] = (16,)
    seed: int = 0
    """Seed for parameter initialization."""

    def __post_init__(self) -> None:
        if any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden sizes must be positive, got {self.hidden_sizes}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    @property
    def name(self) -> str:
        if not self.hidden_sizes:
            return "linear"
        return "mlp" + "x".join(str(h) for h in self.hidden_sizes)


class PolicyNetwork(nn.Module):
    """Maps features to logits. Use :func:`masked_softmax` for probabilities."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        sizes = (NUM_FEATURES, *config.hidden_sizes, NUM_ACTIONS)
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out) for n_in, n_out in itertools.pairwise(sizes)
        )
        initialize_parameters(self, config.seed)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = features
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        logits: torch.Tensor = self.layers[-1](h)
        return logits


def initialize_parameters(model: PolicyNetwork, seed: int) -> None:
    """PyTorch's default ``Linear`` scheme, but from an explicit generator.

    Each weight and bias is drawn from ``U(-1/√fan_in, 1/√fan_in)``. The global
    RNG is never touched, so initialization doesn't depend on whatever else
    ran first.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.layers:
            assert isinstance(layer, nn.Linear)
            bound = 1.0 / math.sqrt(layer.in_features)
            for param in (layer.weight, layer.bias):
                values = torch.rand(param.shape, generator=generator, dtype=param.dtype)
                param.copy_((values * 2 - 1) * bound)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace illegal logits by the most negative finite value.

    ``exp`` of that is exactly 0 after softmax, so illegal actions get zero
    probability and zero gradient. Using a finite value rather than ``-inf``
    avoids ``0 * -inf = nan`` in the loss.
    """
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a bool tensor")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every row of the mask needs at least one legal action")
    return torch.where(mask, logits, torch.finfo(logits.dtype).min)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.softmax(masked_logits(logits, mask), dim=-1)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(masked_logits(logits, mask), dim=-1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# -- inference in float64 --------------------------------------------------------------


def legal_probabilities(
    model: PolicyNetwork, features: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Masked softmax computed in float64 from the model's float32 logits.

    The network runs in float32. The final softmax is done in float64 so the
    resulting distribution sums to 1 within ~1e-16, which the exact evaluator
    requires (tolerance 1e-9).
    """
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(features))).double().numpy()
    z = np.where(mask, logits, -np.inf)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    probs: npt.NDArray[np.float64] = e / e.sum(axis=-1, keepdims=True)
    return probs


def model_strategy(model: PolicyNetwork) -> dict[str, dict[Action, float]]:
    """The model's policy as a 12-entry table, for the exact evaluator."""
    from pokermind.neural.dataset import decision_info_states

    table: dict[str, dict[Action, float]] = {}
    for key, (info, legal) in decision_info_states().items():
        probs = legal_probabilities(model, encode_info_state(info), legal_action_mask(legal))
        table[key] = {a: float(probs[ACTIONS.index(a)]) for a in legal}
    return table


# -- checkpoints -----------------------------------------------------------------------


def checkpoint_bytes(model: PolicyNetwork, metadata: dict[str, Any] | None = None) -> bytes:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": {"hidden_sizes": list(model.config.hidden_sizes), "seed": model.config.seed},
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def save_checkpoint(
    path: Path, model: PolicyNetwork, metadata: dict[str, Any] | None = None
) -> int:
    """Write a checkpoint; returns its size in bytes."""
    data = checkpoint_bytes(model, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def load_checkpoint(path: Path) -> tuple[PolicyNetwork, dict[str, Any]]:
    """Load with ``weights_only=True``: tensors and plain containers only, no pickled code."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} checkpoint")
    cfg = payload["config"]
    model = PolicyNetwork(
        PolicyConfig(hidden_sizes=tuple(int(h) for h in cfg["hidden_sizes"]), seed=int(cfg["seed"]))
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload["metadata"])


__all__ = [
    "PolicyConfig",
    "PolicyNetwork",
    "checkpoint_bytes",
    "initialize_parameters",
    "legal_probabilities",
    "load_checkpoint",
    "masked_log_softmax",
    "masked_logits",
    "masked_softmax",
    "model_strategy",
    "parameter_count",
    "save_checkpoint",
]
