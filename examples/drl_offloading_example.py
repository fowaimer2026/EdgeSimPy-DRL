"""Worked example: DRL-based computation offloading in EdgeSimPy.

Trains the three DRL agents (DQN, PPO, A2C/A3C) on randomized service-placement
scenarios where the reward is the negative weighted offloading cost (latency + energy +
money), then compares their learned policies against the min-cost greedy and random
baselines.

Requires PyTorch. Run with:

    python examples/drl_offloading_example.py
"""

# Python libraries
import numpy as np

# EdgeSimPy DRL
from edge_sim_py.drl import (
    OffloadingEnv,
    DQNAgent,
    PPOAgent,
    A2CAgent,
    random_scenario,
    evaluate_agent,
    evaluate_greedy,
    evaluate_random,
)

COST_PARAMETERS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}
NUM_SERVICES = 8
NUM_SERVERS = 5
TRAIN_EPISODES = 150
EVAL_EPISODES = 50


def make_env(seed: int) -> OffloadingEnv:
    services, servers = random_scenario(num_services=NUM_SERVICES, num_servers=NUM_SERVERS, seed=seed)
    return OffloadingEnv(services, servers, cost_parameters=COST_PARAMETERS)


def main():
    print(f"Service-placement offloading: {NUM_SERVICES} services, {NUM_SERVERS} servers")
    print(f"Reward = -weighted_cost (latency 0.5 / energy 0.3 / money 0.2)\n")

    agents = {
        "DQN": DQNAgent,
        "PPO": PPOAgent,
        "A2C/A3C": A2CAgent,
    }

    results = {}
    for name, agent_class in agents.items():
        env = make_env(seed=42)
        agent = agent_class(env.state_dim, env.num_actions, seed=0)
        history = agent.train(env, num_episodes=TRAIN_EPISODES)
        results[name] = evaluate_agent(agent, env, num_episodes=EVAL_EPISODES)
        print(
            f"{name:<8} trained: reward {np.mean(history[:10]):+.3f} -> {np.mean(history[-10:]):+.3f}" f"  (eval cost {results[name]['mean_total_cost']:.3f})"
        )

    # Baselines on the same scenario distribution.
    env = make_env(seed=42)
    results["Greedy (min-cost)"] = evaluate_greedy(env, num_episodes=EVAL_EPISODES)
    results["Random"] = evaluate_random(env, num_episodes=EVAL_EPISODES)

    print("\n=== Evaluation (lower cost is better) ===")
    print(f"{'Policy':<20}{'Mean cost':>12}{'Placement':>12}")
    for name, metrics in results.items():
        print(f"{name:<20}{metrics['mean_total_cost']:>12.3f}{metrics['mean_placement_rate']:>12.2f}")


if __name__ == "__main__":
    main()
