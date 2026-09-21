"""Base class for bandit agents."""

from abc import ABC, abstractmethod

import numpy as np


class BanditAgent(ABC):
    """Common state and learning rule for action-value bandit agents."""

    def __init__(self, n_actions: int) -> None:
        if n_actions < 1:
            raise ValueError("n_actions must be positive.")
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions, dtype=float)
        self.action_counts = np.zeros(n_actions, dtype=int)

    @abstractmethod
    def select_action(self, step: int) -> int:
        """Select an action."""

    def update(self, action: int, reward: float) -> None:
        """Incrementally update Q(action) using the sample-average method."""
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n
