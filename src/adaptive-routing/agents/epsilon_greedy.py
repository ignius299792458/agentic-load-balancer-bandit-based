"""Epsilon Greedy Bandit Agent."""

import numpy as np

from .base import BanditAgent


class EpsilonGreedyAgent(BanditAgent):

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_actions)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select_action(self, step: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        max_value = np.max(self.q_values)
        candidates = np.flatnonzero(np.isclose(self.q_values, max_value))
        return int(self.rng.choice(candidates))
