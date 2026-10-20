# Deep Reinforcement Learning for Computation Offloading

This contribution adds a self-contained **Deep Reinforcement Learning** subsystem that
learns service-placement (offloading) decisions, using the weighted offloading cost
as the reward signal. It implements three agent families — **DQN** (with Double and
Dueling variants), **PPO**, and **A2C/A3C** — and ties the whole pipeline back to the
cost-model contribution.

PyTorch is the only extra dependency. To keep the core framework torch-free, the
subsystem is an **opt-in** package imported explicitly:

```python
from edge_sim_py.drl import OffloadingEnv, DQNAgent, PPOAgent, A2CAgent
```

## The decision problem

Offloading is framed as a short-horizon MDP, stepped by `OffloadingEnv`:

| Element | Definition |
| --- | --- |
| **State** | Service demand features + per-server features (free CPU/memory and the min-max-normalized latency/energy/monetary costs of hosting the service there). |
| **Action** | Index of the chosen edge server (with an action mask hiding servers without capacity). |
| **Reward** | Negative normalized weighted cost of the placement, in `[-1, 0]`, from `WeightedCostModel`; invalid actions are penalized. |
| **Transition** | Placing a service consumes capacity, changing the options for later services. |

## Components

| File | What it provides |
| --- | --- |
| `edge_sim_py/drl/offloading_env.py` | `OffloadingEnv` — the gym-like placement environment. |
| `edge_sim_py/drl/scenario.py` | `random_scenario` — randomized training infrastructures. |
| `edge_sim_py/drl/networks.py` | `QNetwork` (dueling) and `ActorCritic` torch modules. |
| `edge_sim_py/drl/replay_buffer.py` | `ReplayBuffer` for DQN. |
| `edge_sim_py/drl/agents/` | `DQNAgent`, `PPOAgent`, `A2CAgent`. |
| `edge_sim_py/drl/evaluation.py` | `evaluate_agent`, `evaluate_greedy`, `evaluate_random`. |
| `edge_sim_py/drl/policy.py` | `drl_offloading_policy` — runs a trained agent inside a live simulation. |
| `examples/drl_offloading_example.py` | Trains all three agents and compares them to baselines. |
| `tests/test_drl.py` | Unit tests (`pytest`). |

## Training and evaluating an agent

```python
from edge_sim_py.drl import OffloadingEnv, PPOAgent, random_scenario, evaluate_agent, evaluate_greedy

services, servers = random_scenario(num_services=8, num_servers=5, seed=42)
env = OffloadingEnv(services, servers, cost_parameters={"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}})

agent = PPOAgent(env.state_dim, env.num_actions, seed=0)
agent.train(env, num_episodes=150)

print(evaluate_agent(agent, env))    # {'mean_reward', 'mean_placement_rate', 'mean_total_cost'}
print(evaluate_greedy(env))          # min-cost baseline
```

All three agents share the constructor signature `(state_dim, action_dim, ...)`, a
masked `act(state, mask, greedy=False)` method, and a `train(env, num_episodes)` loop
returning the per-episode reward history.

| Agent | Family | Notes |
| --- | --- | --- |
| `DQNAgent` | Value-based | Replay buffer + target network; Double and Dueling on by default. |
| `PPOAgent` | On-policy actor-critic | Clipped surrogate objective + GAE. |
| `A2CAgent` | On-policy actor-critic | Synchronous A3C; GAE advantages, one step per rollout. |

## Running a trained agent in a live simulation

`drl_offloading_policy` adapts a trained agent into a standard EdgeSimPy
`resource_management_algorithm`, provisioning through the regular `Service.provision`
API so capacity reservation, layer pulling and migration behave normally:

```python
from edge_sim_py import Simulator
from edge_sim_py.drl import drl_offloading_policy

simulator = Simulator(
    resource_management_algorithm=drl_offloading_policy,
    resource_management_algorithm_parameters={"agent": trained_agent, "cost_model_parameters": cost_params},
    # ... other Simulator arguments ...
)
```

The agent's action-space size must equal `len(EdgeServer.all())` in the simulated
infrastructure.

## Running the tests and example

```bash
pip install torch          # the only extra dependency
pytest tests/test_drl.py
python examples/drl_offloading_example.py
```

## Extending

Add a new agent under `edge_sim_py/drl/agents/` exposing the shared
`act` / `train` interface and export it from the package `__init__`. To change the
learning signal, adjust the `cost_parameters` weights (or swap the reward computation in
`OffloadingEnv.step`). New state features go in `OffloadingEnv._encode` (update
`SERVICE_FEATURES` / `SERVER_FEATURES` accordingly).
