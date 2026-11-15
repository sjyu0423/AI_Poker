"""Policy networks with configurable input and output sizes (M4).

The same architecture and initialization as :mod:`pokermind.neural.model`
(``[Linear -> tanh] * k -> Linear``, ``U(-1/√fan_in, 1/√fan_in)`` from an explicit
generator, masked softmax), without the Kuhn-specific 7 inputs and 4 outputs.
The Kuhn module is unchanged, so M2 checkpoints and digests are unaffected.
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

CHECKPOINT_FORMAT: Final = "pokermind.game-policy.v1"


@dataclass(frozen=True, slots=True)
class GamePolicyConfig:
    num_features: int
    num_actions: int
    hidden_sizes: tuple[int, ...] = (32,)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_features <= 0 or self.num_actions <= 1:
            raise ValueError(f"invalid sizes {self.num_features} -> {self.num_actions}")
        if any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden sizes must be positive, got {self.hidden_sizes}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    @property
    def name(self) -> str:
        if not self.hidden_sizes:
            return "linear"
        return "mlp" + "x".join(str(h) for h in self.hidden_sizes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_features": self.num_features,
            "num_actions": self.num_actions,
            "hidden_sizes": list(self.hidden_sizes),
            "seed": self.seed,
        }


class GamePolicyNetwork(nn.Module):
    """Maps features to logits; mask and softmax them for probabilities."""

    def __init__(self, config: GamePolicyConfig) -> None:
        super().__init__()
        self.config = config
        sizes = (config.num_features, *config.hidden_sizes, config.num_actions)
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out) for n_in, n_out in itertools.pairwise(sizes)
        )
        generator = torch.Generator().manual_seed(config.seed)
        with torch.no_grad():
            for layer in self.layers:
                assert isinstance(layer, nn.Linear)
                bound = 1.0 / math.sqrt(layer.in_features)
                for param in (layer.weight, layer.bias):
                    values = torch.rand(param.shape, generator=generator, dtype=param.dtype)
                    param.copy_((values * 2 - 1) * bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = features
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        logits: torch.Tensor = self.layers[-1](h)
        return logits


def legal_probabilities(
    model: GamePolicyNetwork,
    features: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Masked softmax in float64 from the float32 logits (rows sum to 1 within ~1e-16).

    Illegal actions get exactly 0; every legal action gets a positive
    probability unless its logit is so far below the others that ``exp``
    underflows in float64.
    """
    if not bool(mask.any(axis=-1).all()):
        raise ValueError("every row of the mask needs at least one legal action")
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(features))).double().numpy()
    z = np.where(mask, logits, -np.inf)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    probs: npt.NDArray[np.float64] = e / e.sum(axis=-1, keepdims=True)
    return probs


def checkpoint_bytes(model: GamePolicyNetwork, metadata: dict[str, Any] | None = None) -> bytes:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def save_checkpoint(
    path: Path, model: GamePolicyNetwork, metadata: dict[str, Any] | None = None
) -> int:
    data = checkpoint_bytes(model, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def load_checkpoint(path: Path) -> tuple[GamePolicyNetwork, dict[str, Any]]:
    """Load with ``weights_only=True``: tensors and plain containers only."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} checkpoint")
    cfg = payload["config"]
    model = GamePolicyNetwork(
        GamePolicyConfig(
            num_features=int(cfg["num_features"]),
            num_actions=int(cfg["num_actions"]),
            hidden_sizes=tuple(int(h) for h in cfg["hidden_sizes"]),
            seed=int(cfg["seed"]),
        )
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload["metadata"])
