"""Server-routing multi-armed bandit environment."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Server:
    """Hidden performance characteristics of one backend server."""

    name: str
    mean_latency_ms: float
    latency_std_ms: float
    failure_rate: float


class ServerRoutingEnvironment:
    """Stationary stochastic MAB environment.

    The agent only chooses a server and observes a reward.
    Server latency/failure characteristics remain hidden from the agent.
    """

    def __init__(self, servers: list[Server], seed: int | None = None) -> None:
        if not servers:
            raise ValueError("At least one server is required.")

        self.servers = servers
        self.rng = np.random.default_rng(seed)

    @property
    def n_actions(self) -> int:
        return len(self.servers)

    def reset(self, seed: int | None = None) -> None:
        """Reset the random generator."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def step(self, action: int) -> tuple[float, dict]:
        """Route one request to a server and return reward + diagnostics."""
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be in [0, {self.n_actions - 1}].")

        server = self.servers[action]
        failed = bool(self.rng.random() < server.failure_rate)

        if failed:
            latency_ms = float(
                self.rng.normal(server.mean_latency_ms, server.latency_std_ms)
            )
            latency_ms = max(1.0, latency_ms)
            reward = -1.0
        else:
            latency_ms = float(
                self.rng.normal(server.mean_latency_ms, server.latency_std_ms)
            )
            latency_ms = max(1.0, latency_ms)
            reward = 1.0 - latency_ms / 200.0

        info = {
            "server": server.name,
            "latency_ms": latency_ms,
            "failed": failed,
        }
        return float(reward), info


def five_environments_as_mab(seed: int | None = 42) -> ServerRoutingEnvironment:
    """Create the five-server as Multi_Arms_Bandit (mab) benchmark used by the experiments."""
    servers = [
        Server("server-1", 120, 15, 0.02),
        Server("server-2", 80, 12, 0.05),
        Server("server-3", 150, 20, 0.01),
        Server("server-4", 60, 10, 0.08),
        Server("server-5", 100, 14, 0.03),
    ]
    return ServerRoutingEnvironment(servers, seed=seed)
