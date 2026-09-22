# Agentic Load Balancer: Adaptive Server Request Routing Using Reinforcement Learning: A Multi-Armed Bandit Approach

> REPO: [agentic-load-balancer-bandit-based](https://github.com/ignius299792458/agentic-load-balancer-bandit-based)

## Introduction

Modern distributed applications often rely on multiple backend servers to process incoming requests. However, server performance is not necessarily identical. Different servers may have different latency, reliability, and response behavior.

Traditional routing strategies such as round-robin or fixed-server routing do not continuously learn from observed server performance.

This project investigates **adaptive server request routing using Reinforcement Learning (RL)** through a **Multi-Armed Bandit (MAB)** formulation.

For every incoming request, the agent selects one backend server and receives a reward based on the quality of that server's response. The agent does not know the true performance characteristics of the servers beforehand. It must estimate them through repeated interaction.

The core learning process is:

```text

Select Server (Arms or Action)
             │
             ▼
        Server processes
          request
(ENV: action-step and response)
             │
             ▼
       Observe reward
             │
             ▼
       Update Q(a)
             │
             └──────────────► Next request

```

The current project focuses on the **Multi-Armed Bandit stage of Reinforcement Learning**, where there is no action-dependent future state:

$$
A_t \rightarrow R_{t+1}
$$

This provides a controlled foundation for eventually extending the same problem toward **Contextual Bandits, Markov Decision Processes (MDPs), and Deep Reinforcement Learning**.

---

# Project Description

## Way To Run

### 1. Clone the repository

```bash
git clone https://github.com/ignius299792458/agentic-load-balancer-bandit-based
cd agentic-load-balancer-bandit-based
```

### 2. Install dependencies

The project uses **Python 3.12** and **Poetry**.

```bash
poetry install
```

<!-- ### Run the tests

```bash
poetry run pytest
``` -->

### 3. Run the benchmark

```bash
poetry run python -m adaptive_routing.experiments.run_experiment
# or
# use debugger if you running in vscode
```

The experiment generates:

```text
results/
├── cumulative_reward_{datetime.now()}.png
└── cumulative_regret_{datetime.now()}.png
```

---

## Problem Formulation

The environment contains five backend servers.

Each server has hidden performance characteristics:

| Server   | Mean Latency | Latency Std. | Failure Rate |
| -------- | -----------: | -----------: | -----------: |
| Server 1 |       120 ms |        15 ms |           2% |
| Server 2 |        80 ms |        12 ms |           5% |
| Server 3 |       150 ms |        20 ms |           1% |
| Server 4 |        60 ms |        10 ms |           8% |
| Server 5 |       100 ms |        14 ms |           3% |

The agent does **not** know these values.

Instead, every request becomes a bandit decision:

```text
Action 0 → Server 1
Action 1 → Server 2
Action 2 → Server 3
Action 3 → Server 4
Action 4 → Server 5
```

After the selected server processes the request, the environment returns a reward.

### Reward

For a successful request:

$$
R = 1-\frac{\text{latency}}{200}
$$

For a failed request:

$$
R=-1
$$

Therefore:

```text
Low latency + successful request → higher reward
High latency + successful request → lower reward
Failed request                  → negative reward
```

The agent learns an estimate:

$$
Q(a) \approx \mathbb{E}[R \mid A=a]
$$

where `Q(a)` represents the estimated expected reward for selecting server `a`.

---

# Algorithms

The project currently implements three Multi-Armed Bandit algorithms.

## 1. Greedy

The Greedy agent always selects the action with the highest estimated value:

$$
A_t=\arg\max_a Q_t(a)
$$

This is pure exploitation.

Its main weakness is that it does not deliberately explore uncertain actions.

---

## 2. Epsilon-Greedy

Epsilon-Greedy balances exploration and exploitation.

With probability \(\epsilon\), the agent selects a random server:

```text
ε probability
     ↓
Explore random server
```

Otherwise, it selects the server with the highest estimated value:

```text
1 - ε probability
     ↓
Exploit best-known server
```

The benchmark uses:

```text
epsilon = 0.1
```

---

## 3. Upper Confidence Bound (UCB)

UCB explicitly considers both estimated reward and uncertainty:

$$
A_t =
\arg\max_a
\left[
Q_t(a)
+
c\sqrt{\frac{\ln t}{N_t(a)}}
\right]
$$

where:

- \(Q_t(a)\) = estimated reward
- \(N_t(a)\) = number of times action `a` has been selected
- \(c\) = exploration parameter

An action that has been selected fewer times receives a larger exploration bonus.

---

# Learning Rule

The agents use incremental sample-average learning:

$$
Q_{n+1}(a)
=
Q_n(a)
+
\frac{1}{n}
\left[
R_n-Q_n(a)
\right]
$$

This updates the estimated value after every interaction without storing the complete reward history.

The learning cycle is:

```text
1. Select server
2. Send simulated request
3. Observe latency and failure
4. Calculate reward
5. Update Q-value
6. Repeat
```

---

# Benchmark Description

The benchmark evaluates the three algorithms in the same **stationary stochastic server-routing environment**.

## Configuration

```txt

- Number of servers(arms):      5
- Number of requests (steps):   10,000
- Epsilon:                      0.1
- UCB exploration c:            2.0
- Environment:                  stochastic
- Server characteristics:       hidden from the agent
```

The benchmark measures:

- cumulative reward
- cumulative pseudo-regret
- average latency
- request failure rate
- learned action values

---

## True Mean Rewards

The underlying expected rewards of the five servers are:

| Server   | True Mean Reward |
| -------- | ---------------: |
| Server 1 |           0.3720 |
| Server 2 |           0.5200 |
| Server 3 |           0.2375 |
| Server 4 |       **0.5640** |
| Server 5 |           0.4550 |

Therefore, **Server 4 has the highest true expected reward** in this benchmark.

The agents are not given these values. They must estimate them from experience.

---

# Benchmark Results

## Cumulative Reward

![Cumulative Reward](results/cumulative_reward_2026-09-22%2010:42:08.885969.png)

The cumulative reward increases as the agents continue interacting with the environment.

The final benchmark results were:

| Algorithm          | Average Reward | Failure Rate | Average Latency |
| ------------------ | -------------: | -----------: | --------------: |
| Greedy             |         0.5229 |       4.790% |        80.06 ms |
| **Epsilon-Greedy** |     **0.5355** |       6.390% |    **71.62 ms** |
| UCB                |         0.5307 |       5.980% |        73.89 ms |

In this particular run, **Epsilon-Greedy obtained the highest average reward and lowest average latency** among the three algorithms.

However, Epsilon-Greedy also experienced the highest failure rate. This is a consequence of exploration: it deliberately selects servers that may be less favorable, including servers with higher failure probability.

The results therefore demonstrate that evaluating a bandit algorithm using only one metric can be misleading. Reward, latency, reliability, and regret provide different perspectives on performance.

---

## Cumulative Regret

![Cumulative Regret](results/cumulative_regret_2026-09-22%2010:42:09.016063.png)

Bandit regret measures the performance lost by selecting an action whose expected reward is lower than that of the optimal fixed action.

The cumulative pseudo-regret is:

$$
\text{Regret}_T = \sum_{t=1}^{T} \left(q_*(a^*)-q_*(A_t)\right)
$$

where:

- \(a^\*\) = optimal server
- \(q\__(a^_)\) = true expected reward of the optimal server
- \(q\_\*(A_t)\) = true expected reward of the server selected at time \(t\)

Lower cumulative regret means that the algorithm spends less time selecting suboptimal servers.

---

# Observed Benchmark Results

```text
STEP = 10,000

Algorithm          Avg Reward    Failure Rate    Avg Latency    Regret
-----------------------------------------------------------------------
Greedy                0.5229         4.790%        80.06 ms     440.00
Epsilon-Greedy        0.5355         6.390%        71.62 ms     297.01
UCB                   0.5307         5.980%        73.89 ms     383.39
```

### Learned Q-values

```text
Greedy:
[0.0000, 0.5229, 0.0000, 0.0000, 0.0000]

Epsilon-Greedy:
[0.3506, 0.5222, 0.2324, 0.5666, 0.4308]

UCB:
[0.3787, 0.5249, 0.2398, 0.5697, 0.4562]
```

---

# Interpretation

The learned Q-values provide an important insight into the behavior of the algorithms.

![Learned Q-Values](/Learned%20Q-Values%20Sep%2022,%202026,%2002_20_36%20PM.png)

### Greedy

Greedy learned:

```text
Server 1 → 0.0000
Server 2 → 0.5229
Server 3 → 0.0000
Server 4 → 0.0000
Server 5 → 0.0000
```

The agent effectively committed to **Server 2** and did not sufficiently explore the remaining servers.

This is the classic exploration problem of a purely greedy bandit algorithm.

Even though Server 4 has the highest true expected reward (`0.5640`), Greedy failed to discover it in this run.

This explains its relatively high cumulative regret:

```text
Greedy regret = 440.00
```

---

### Epsilon-Greedy

Epsilon-Greedy explored all five servers:

```text
[0.3506, 0.5222, 0.2324, 0.5666, 0.4308]
```

Its estimate for Server 4:

```text
0.5666
```

is very close to the true value:

```text
0.5640
```

It therefore successfully discovered the optimal arm.

It achieved:

```text
Average reward = 0.5355
Regret         = 297.01
Average latency = 71.62 ms
```

The exploration behavior also caused it to encounter more failed requests:

```text
Failure rate = 6.390%
```

---

### UCB

UCB also explored all five servers and learned values close to the underlying true rewards:

```text
[0.3787, 0.5249, 0.2398, 0.5697, 0.4562]
```

These estimates are close to:

```text
[0.3720, 0.5200, 0.2375, 0.5640, 0.4550]
```

This indicates that UCB successfully estimated the reward distributions.

Its benchmark performance was:

```text
Average reward = 0.5307
Regret         = 383.39
Average latency = 73.89 ms
```

Its regret was higher than Epsilon-Greedy in this particular experiment because UCB spent more decisions exploring uncertain actions.

---

# Important Experimental Observation

This experiment demonstrates why **one benchmark run should not be interpreted as a universal ranking of algorithms**.

The observed result is:

```text
Epsilon-Greedy  → lowest regret
UCB             → intermediate regret
Greedy          → highest regret
```

But these results depend on:

- random seed
- reward stochasticity
- initial estimates
- exploration parameters
- number of requests
- server distributions

Therefore, a stronger experimental evaluation should run each algorithm across **many independent random seeds** and report mean and standard deviation of reward and regret.

That will be the next step toward making this project a more rigorous RL experiment.

---

# Conclusion

This project demonstrates how adaptive server request routing can be formulated as a **Reinforcement Learning problem using the Multi-Armed Bandit framework**.

The agent repeatedly:

```text
Observe experience
      ↓
Select server
      ↓
Receive reward
      ↓
Update Q-value
      ↓
Make next decision
```

The current implementation demonstrates fundamental RL concepts including:

- action selection
- reward feedback
- exploration vs. exploitation
- action-value estimation
- incremental learning
- stochastic environments
- cumulative reward
- regret

The benchmark also demonstrates a fundamental lesson in reinforcement learning:

> **Exploration is necessary because the agent does not initially know which action is optimal.**

Greedy became trapped after selecting Server 2, whereas Epsilon-Greedy and UCB explored the available servers and discovered that Server 4 had the highest true expected reward.

The current environment is intentionally a **Multi-Armed Bandit rather than a full MDP**. There is no explicit state and no action-dependent future state:

$$
A_t \rightarrow R_{t+1}
$$

The planned progression is:

```text
Multi-Armed Bandit
        ↓
Contextual Bandit
        ↓
MDP-based Server Routing
        ↓
Deep Reinforcement Learning
        ↓
Model-Based / Deep RL
```

A future MDP formulation can introduce server load, queue length, request characteristics, and changing server states. Routing decisions can then influence future system conditions:

$$
S_t \rightarrow A_t \rightarrow R_{t+1}, S_{t+1}
$$

This will transform the project from a simple bandit learning problem into a genuine **sequential decision-making RL problem**, while preserving the same real-world server-routing objective.
