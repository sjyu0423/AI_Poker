"""A neural Leduc policy as a ``GameAgent`` (M4).

At each decision it encodes the observation (own card, public card, public
betting; never the opponent's card), runs the network, masks illegal actions
and samples with the shared one-draw CDF sampler.
"""

from __future__ import annotations

import numpy as np

from pokermind.agents.cfr_agent import sample_action
from pokermind.games.base import Observation
from pokermind.games.leduc import LeducConfig, LeducObservation
from pokermind.neural.game_model import GamePolicyNetwork, legal_probabilities
from pokermind.neural.leduc_encoding import encode_observation, legal_action_mask


class LeducNeuralAgent:
    def __init__(
        self, model: GamePolicyNetwork, config: LeducConfig | None = None, name: str | None = None
    ) -> None:
        self.model = model.eval()
        self.config = config or LeducConfig()
        self._name = name or f"neural({model.config.name})"

    @property
    def name(self) -> str:
        return self._name

    def distribution(
        self, observation: LeducObservation, legal_actions: tuple[int, ...]
    ) -> dict[int, float]:
        probs = legal_probabilities(
            self.model,
            encode_observation(observation, self.config)[None, :],
            legal_action_mask(legal_actions)[None, :],
        )[0]
        return {a: float(probs[a]) for a in legal_actions}

    def act(
        self, observation: Observation, legal_actions: tuple[int, ...], rng: np.random.Generator
    ) -> int:
        if not isinstance(observation, LeducObservation):
            raise TypeError(f"expected LeducObservation, got {type(observation).__name__}")
        dist = self.distribution(observation, legal_actions)
        return sample_action(legal_actions, [dist[a] for a in legal_actions], rng, observation.key)
