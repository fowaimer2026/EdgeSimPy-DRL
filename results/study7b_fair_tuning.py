"""Study 7b - Fair hyperparameter exploration across DQN / PPO / A2C (Part 3, item 2).

Study 7 tuned only the on-policy agents. This study gives all three agents a **comparable
optimization budget**: the same number of training episodes, the same number of seeds, and an
equal-size hyperparameter grid (6 configurations each = learning rate {3} × one agent-appropriate
secondary parameter {2}). It reports the best configuration per agent and a budget-transparency
table (episodes, wall-clock) so the fairness of the comparison is explicit.

Secondary parameter per agent (kept to an equal-size grid):
    * DQN     -- epsilon_decay ∈ {0.99, 0.995}     (exploration schedule)
    * PPO/A2C -- entropy_coef  ∈ {0.01, 0.05}       (exploration bonus)

Requires PyTorch.
"""

# EdgeSimPy DRL
from edge_sim_py.drl import OffloadingEnv, DQNAgent, PPOAgent, A2CAgent, random_scenario, evaluate_agent, evaluate_greedy
from edge_sim_py.experiments import confidence_interval, results_to_csv

# Python libraries
import time

import numpy as np

from _common import output_path, grouped_bar_chart

NUM_SERVICES = 8
NUM_SERVERS = 5
EPISODES = 300  # identical budget for every agent
SEEDS = range(10)
COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}
LEARNING_RATES = [1e-4, 3e-4, 1e-3]

# Equal-size grid (2 secondary values) with an agent-appropriate secondary hyperparameter.
GRID = {
    "DQN": (DQNAgent, "epsilon_decay", [0.99, 0.995]),
    "PPO": (PPOAgent, "entropy_coef", [0.01, 0.05]),
    "A2C": (A2CAgent, "entropy_coef", [0.01, 0.05]),
}


def make_env(seed):
    services, servers = random_scenario(num_services=NUM_SERVICES, num_servers=NUM_SERVERS, seed=seed)
    return OffloadingEnv(services, servers, cost_parameters=COST_PARAMS)


def main():
    total_configs = sum(len(LEARNING_RATES) * len(vals) for _, (_, _, vals) in GRID.items())
    print(f"Study 7b - Fair tuning ({EPISODES} episodes/agent, {len(list(SEEDS))} seeds, " f"{total_configs // len(GRID)} configs/agent, equal budget)\n")

    rows, best_per_agent, budget = [], {}, {}
    for name, (agent_class, sec_name, sec_values) in GRID.items():
        best = None
        t0 = time.perf_counter()
        for lr in LEARNING_RATES:
            for sec in sec_values:
                finals = []
                for seed in SEEDS:
                    env = make_env(seed)
                    agent = agent_class(env.state_dim, env.num_actions, learning_rate=lr, seed=seed, **{sec_name: sec})
                    agent.train(env, num_episodes=EPISODES)
                    finals.append(evaluate_agent(agent, env, num_episodes=20)["mean_total_cost"])
                ci = confidence_interval(finals)
                # Use uniform column names so DQN (epsilon_decay) and PPO/A2C (entropy_coef) rows
                # share a schema for CSV export.
                rows.append(
                    {"agent": name, "lr": lr, "secondary_param": sec_name, "secondary_value": sec, "final_cost": ci["mean"], "cost_ci": ci["half_width"]}
                )
                if best is None or ci["mean"] < best["final_cost"]:
                    best = rows[-1]
        wall = time.perf_counter() - t0
        best_per_agent[name] = best
        budget[name] = {"episodes": EPISODES, "configs": len(LEARNING_RATES) * len(sec_values), "seeds": len(list(SEEDS)), "wall_s": wall}
        print(
            f"  {name}: best final_cost={best['final_cost']:.3f} (lr={best['lr']:g}, {best['secondary_param']}={best['secondary_value']:g})  "
            f"[{budget[name]['configs']} configs x {EPISODES} ep x {budget[name]['seeds']} seeds, {wall:.0f}s]"
        )

    greedy = np.mean([evaluate_greedy(make_env(seed), num_episodes=20)["mean_total_cost"] for seed in SEEDS])

    results_to_csv(rows, output_path("study7b_fair_tuning_grid.csv"))
    results_to_csv([{"agent": k, **v} for k, v in budget.items()], output_path("study7b_budget.csv"))

    print("\nBudget-transparency table:")
    print(f"  {'agent':<6}{'episodes':>10}{'configs':>9}{'seeds':>7}{'wall (s)':>10}{'best cost':>12}")
    for name in GRID:
        b = budget[name]
        print(f"  {name:<6}{b['episodes']:>10}{b['configs']:>9}{b['seeds']:>7}{b['wall_s']:>10.0f}{best_per_agent[name]['final_cost']:>12.3f}")
    print(f"  greedy baseline final cost: {greedy:.3f}")

    fig_path = grouped_bar_chart(
        categories=list(GRID.keys()) + ["greedy"],
        series={"best final cost": [best_per_agent[n]["final_cost"] for n in GRID] + [greedy]},
        errors={"best final cost": [best_per_agent[n]["cost_ci"] for n in GRID] + [0.0]},
        title=f"Fair tuning: best per agent at equal budget ({EPISODES} ep, 6 configs each)",
        ylabel="Mean weighted cost",
        path=output_path("study7b_fair_tuning.png"),
    )
    print(f"\nWrote grid + budget CSVs + {fig_path}")


if __name__ == "__main__":
    main()
