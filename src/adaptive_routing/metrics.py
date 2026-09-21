"""Experiment metrics."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ExperimentResult:
    rewards: np.ndarray
    latencies_ms: np.ndarray
    failures: np.ndarray
    actions: np.ndarray
    q_values: np.ndarray

    @property
    def cumulative_reward(self) -> np.ndarray:
        return np.cumsum(self.rewards)

    @property
    def average_reward(self) -> float:
        return float(np.mean(self.rewards))

    @property
    def failure_rate(self) -> float:
        return float(np.mean(self.failures))

    @property
    def average_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms))


def calculate_true_mean_rewards(environment) -> np.ndarray:
    """Analytical mean reward of each arm for the stationary environment."""
    means = []
    for server in environment.servers:
        expected_latency = server.mean_latency_ms
        success_reward = 1.0 - expected_latency / 200.0
        mean_reward = (
            1.0 - server.failure_rate
        ) * success_reward + server.failure_rate * -1.0
        means.append(mean_reward)
    return np.asarray(means)


def calculate_cumulative_regret(
    actions: np.ndarray,
    true_mean_rewards: np.ndarray,
) -> np.ndarray:
    """Cumulative pseudo-regret against the best fixed arm."""
    best_mean = np.max(true_mean_rewards)
    instantaneous_regret = best_mean - true_mean_rewards[actions]
    return np.cumsum(instantaneous_regret)
