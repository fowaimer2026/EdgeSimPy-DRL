"""Study 7 - DRL hyperparameter tuning + learning curves (PPO / A2C).

Addresses the reviewer request for real hyperparameter tuning and learning curves for the
on-policy agents, rather than only the fixed-budget comparison of Study 2 (where PPO/A2C
lagged). A grid over learning rate x entropy coefficient is swept for PPO and A2C on a small
synthetic offloading scenario (fast, reproducible, no dataset needed); each configuration is
trained for a longer budget across seeds. We report the full grid (final cost with 95% CIs),
identify the best configuration per agent, and plot learning curves for the tuned agents next
to DQN and the greedy baseline.

Requires PyTorch.
"""

# EdgeSimPy DRL
from edge_sim_py.drl import OffloadingEnv, DQNAgent, PPOAgent, A2CAgent, random_scenario, evaluate_agent, evaluate_greedy
from edge_sim_py.experiments import confidence_interval, results_to_csv

# Python libraries
import numpy as np

from _common import output_path, line_chart, grouped_bar_chart

NUM_SERVICES = 8
NUM_SERVERS = 5
EPISODES = 400
SEEDS = range(4)
COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}

LEARNING_RATES = [1e-4, 3e-4, 1e-3]
ENTROPY_COEFS = [0.01, 0.05]


def make_env(seed):
    services, servers = random_scenario(num_services=NUM_SERVICES, num_servers=NUM_SERVERS, seed=seed)
    return OffloadingEnv(services, servers, cost_parameters=COST_PARAMS)


def _train_config(agent_class, lr, entropy, seed):
    """Trains one agent configuration on one seed; returns (reward_history, final_cost)."""
    env = make_env(seed)
    agent = agent_class(env.state_dim, env.num_actions, learning_rate=lr, entropy_coef=entropy, seed=seed)
    history = agent.train(env, num_episodes=EPISODES)
    final = evaluate_agent(agent, env, num_episodes=20)["mean_total_cost"]
    return history, final


def _sweep(agent_class, name):
    """Sweeps the lr x entropy grid; returns (grid_rows, best_config, best_histories)."""
    rows, best = [], None
    histories_by_config = {}
    for lr in LEARNING_RATES:
        for entropy in ENTROPY_COEFS:
            histories, finals = [], []
            for seed in SEEDS:
                hist, final = _train_config(agent_class, lr, entropy, seed)
                histories.append(hist)
                finals.append(final)
            ci = confidence_interval(finals)
            key = f"lr={lr:g},ent={entropy:g}"
            histories_by_config[key] = histories
            rows.append({"agent": name, "lr": lr, "entropy": entropy, "final_cost": ci["mean"], "cost_ci": ci["half_width"]})
            print(f"  {name:<4} {key:<18} final_cost={ci['mean']:.3f} +/-{ci['half_width']:.3f}")
            if best is None or ci["mean"] < best["final_cost"]:
                best = rows[-1]
    return rows, best, histories_by_config


def _smooth(curve, window=15):
    return np.convolve(curve, np.ones(window) / window, mode="valid")


def main():
    print(f"Study 7 - DRL hyperparameter tuning ({NUM_SERVICES} services/{NUM_SERVERS} servers, " f"{EPISODES} episodes, {len(list(SEEDS))} seeds, 95% CI)\n")

    all_rows = []
    curves = {}
    for agent_class, name in [(PPOAgent, "PPO"), (A2CAgent, "A2C")]:
        print(f"{name} grid:")
        rows, best, histories = _sweep(agent_class, name)
        all_rows.extend(rows)
        best_key = f"lr={best['lr']:g},ent={best['entropy']:g}"
        print(f"  -> best {name}: {best_key} (final_cost {best['final_cost']:.3f})\n")
        curves[f"{name} (tuned {best_key})"] = np.mean([_smooth(np.array(h)) for h in histories[best_key]], axis=0)

    # DQN reference (default hyperparameters) for the learning-curve comparison.
    dqn_hist = []
    for seed in SEEDS:
        env = make_env(seed)
        agent = DQNAgent(env.state_dim, env.num_actions, seed=seed)
        dqn_hist.append(agent.train(env, num_episodes=EPISODES))
    curves["DQN (default)"] = np.mean([_smooth(np.array(h)) for h in dqn_hist], axis=0)

    # Greedy baseline as a horizontal reference line (reward = -cost).
    greedy_costs = [evaluate_greedy(make_env(seed), num_episodes=20)["mean_total_cost"] for seed in SEEDS]
    greedy_reward = -float(np.mean(greedy_costs))

    csv_path = results_to_csv(all_rows, output_path("study7_drl_tuning_grid.csv"))

    x = list(range(min(len(c) for c in curves.values())))
    curves = {k: v[: len(x)] for k, v in curves.items()}
    curves[f"greedy baseline ({greedy_reward:.2f})"] = np.full(len(x), greedy_reward)
    curve_path = line_chart(
        x=x,
        series=curves,
        title="DRL learning curves after tuning (15-ep smoothed; higher = better)",
        xlabel="Training episode",
        ylabel="Episode reward (= -cost)",
        path=output_path("study7_drl_tuning_curves.png"),
    )

    # Best-config bar chart per agent.
    best_ppo = min((r for r in all_rows if r["agent"] == "PPO"), key=lambda r: r["final_cost"])
    best_a2c = min((r for r in all_rows if r["agent"] == "A2C"), key=lambda r: r["final_cost"])
    bar_path = grouped_bar_chart(
        categories=["PPO (tuned)", "A2C (tuned)", "greedy"],
        series={"final cost": [best_ppo["final_cost"], best_a2c["final_cost"], -greedy_reward]},
        errors={"final cost": [best_ppo["cost_ci"], best_a2c["cost_ci"], 0.0]},
        title="Tuned on-policy agents vs. greedy (lower is better)",
        ylabel="Mean weighted cost",
        path=output_path("study7_drl_tuning_best.png"),
    )
    print(f"\nBest PPO final_cost={best_ppo['final_cost']:.3f}, best A2C={best_a2c['final_cost']:.3f}, greedy={-greedy_reward:.3f}")
    print(f"Wrote: {csv_path}\n       {curve_path}\n       {bar_path}")


if __name__ == "__main__":
    main()
