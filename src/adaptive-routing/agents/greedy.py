"""Greedy bandit agent."""

import numpy as np

from .base import BanditAgent


class GreedyAgent(BanditAgent):
    """Always exploit the action with the largest estimated value."""

    def select_action(self, step: int) -> int:
        max_value = np.max(self.q_values)
        candidates = np.flatnonzero(np.isclose(self.q_values, max_value))
        return int(np.random.choice(candidates))
