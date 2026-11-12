"""A neural policy that implements the M0 ``Agent`` interface.

At every decision it encodes the ``InfoState`` (own card and public history
only), runs the network, masks the illegal actions, and samples with the
generator the simulator gives it. Sampling is shared with ``StrategyAgent``,
so a network that outputs exactly the teacher's probabilities makes exactly
the teacher's choices under the same seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pokermind.agents.cfr_agent import sample_action
from pokermind.games.kuhn import Action, InfoState
from pokermind.neural.encoding import ACTIONS, encode_info_state, legal_action_mask
from pokermind.neural.model import PolicyNetwork, legal_probabilities, load_checkpoint


class NeuralPolicyAgent:
    def __init__(self, model: PolicyNetwork, name: str | None = None) -> None:
        self.model = model.eval()
        self._name = name or f"neural({model.config.name})"

    @classmethod
    def from_checkpoint(cls, path: Path) -> NeuralPolicyAgent:
        model, _ = load_checkpoint(path)
        return cls(model, f"neural({path.name})")

    @property
    def name(self) -> str:
        return self._name

    def distribution(
        self, info_state: InfoState, legal_actions: tuple[Action, ...]
    ) -> dict[Action, float]:
        probs = legal_probabilities(
            self.model, encode_info_state(info_state), legal_action_mask(legal_actions)
        )
        return {a: float(probs[ACTIONS.index(a)]) for a in legal_actions}

    def act(
        self,
        info_state: InfoState,
        legal_actions: tuple[Action, ...],
        rng: np.random.Generator,
    ) -> Action:
        dist = self.distribution(info_state, legal_actions)
        return sample_action(legal_actions, [dist[a] for a in legal_actions], rng, info_state.key)
