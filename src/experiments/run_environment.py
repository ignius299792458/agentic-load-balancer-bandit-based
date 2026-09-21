"""Run and compare bandit algorithms."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adaptive_routing.agents import EpsilonGreedyAgent, GreedyAgent, UCBAgent
from adaptive_routing.environment import default_environment
from adaptive_routing.metrics import (
    ExperimentResult,
    calculate_cumulative_regret,
    calculate_true_mean_rewards,
)


def run_agent(agent, steps: int = 10_000, seed: int = 42) -> ExperimentResult:
    """Run one agent against a fresh environment."""
    env = default_environment(seed=seed)

    rewards = np.zeros(steps)
    latencies = np.zeros(steps)
    failures = np.zeros(steps)
    actions = np.zeros(steps, dtype=int)

    for step in range(steps):
        action = agent.select_action(step)
        reward, info = env.step(action)
        agent.update(action, reward)

        actions[step] = action
        rewards[step] = reward
        latencies[step] = info["latency_ms"]
        failures[step] = info["failed"]

    return ExperimentResult(
        rewards=rewards,
        latencies_ms=latencies,
        failures=failures,
        actions=actions,
        q_values=agent.q_values.copy(),
    )


def main() -> None:
    """Compare the first three algorithms."""
    steps = 10

    agents = {
        "Greedy": GreedyAgent(5),
        "Epsilon-Greedy": EpsilonGreedyAgent(5, epsilon=0.1, seed=7),
        "UCB": UCBAgent(5, c=2.0),
    }

    results = {
        name: run_agent(agent, steps=steps, seed=42) for name, agent in agents.items()
    }

    true_means = calculate_true_mean_rewards(default_environment(seed=42))

    print("True mean reward by server:")
    for i, mean in enumerate(true_means):
        print(f"  server-{i + 1}: {mean:.4f}")

    print("\nResults:")
    for name, result in results.items():
        regret = calculate_cumulative_regret(result.actions, true_means)[-1]
        print(
            f"{name:16s} | "
            f"avg reward={result.average_reward:.4f} | "
            f"failure rate={result.failure_rate:.3%} | "
            f"avg latency={result.average_latency_ms:.2f} ms | "
            f"regret={regret:.2f}"
        )
        print(f"  learned Q={np.round(result.q_values, 4)}")

    Path("results").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    for name, result in results.items():
        plt.plot(result.cumulative_reward, label=name)
    plt.xlabel("Request")
    plt.ylabel("Cumulative reward")
    plt.title("Adaptive Server Request Routing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/cumulative_reward_{datetime.now()}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    for name, result in results.items():
        regret = calculate_cumulative_regret(result.actions, true_means)
        plt.plot(regret, label=name)
    plt.xlabel("Request")
    plt.ylabel("Cumulative pseudo-regret")
    plt.title("Bandit Regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/cumulative_regret_{datetime.now()}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
