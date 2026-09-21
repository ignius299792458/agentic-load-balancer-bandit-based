"""Upper-Confidence-Bound bandit agent."""

import numpy as np

from .base import BanditAgent


class UCBAgent(BanditAgent):
    """UCB1 agent balancing estimated value and uncertainty."""

    def __init__(
        self,
        n_actions: int,
        c: float = 2.0,
    ) -> None:
        super().__init__(n_actions)
        if c < 0:
            raise ValueError("c must be non-negative.")
        self.c = c

    def select_action(self, step: int) -> int:
        # Force every action to be sampled once.
        untried = np.flatnonzero(self.action_counts == 0)
        if len(untried):
            return int(untried[0])

        exploration_bonus = self.c * np.sqrt(np.log(step + 1) / self.action_counts)
        ucb_values = self.q_values + exploration_bonus
        return int(np.argmax(ucb_values))
