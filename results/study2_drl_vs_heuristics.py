"""Study 2 - DRL agents vs. heuristic baselines.

Trains the three DRL agents (DQN, PPO, A2C) on the offloading environment and compares
their learned policies against the min-cost greedy and random baselines, across seeds with
95% confidence intervals. Emits learning curves and a final-cost comparison (CSV + figures).

Requires PyTorch. Runtime is a couple of minutes on CPU.
"""

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
from edge_sim_py.experiments import confidence_interval, results_to_csv, compare_all_pairs

# Python libraries
import numpy as np

from _common import output_path, grouped_bar_chart, line_chart

COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}
NUM_SERVICES = 12
NUM_SERVERS = 5
EPISODES = 300
SEEDS = range(16)
AGENTS = {"DQN": DQNAgent, "PPO": PPOAgent, "A2C": A2CAgent}


def make_env(seed):
    services, servers = random_scenario(num_services=NUM_SERVICES, num_servers=NUM_SERVERS, seed=seed)
    return OffloadingEnv(services, servers, cost_parameters=COST_PARAMS)


def main():
    print(f"Study 2 - DRL vs heuristics ({len(list(SEEDS))} seeds, {EPISODES} episodes each)\n")

    histories = {name: [] for name in AGENTS}  # per-agent list of per-seed reward curves
    final_cost = {name: [] for name in AGENTS}
    greedy_cost, random_cost = [], []

    for seed in SEEDS:
        for name, agent_class in AGENTS.items():
            env = make_env(seed)
            agent = agent_class(env.state_dim, env.num_actions, seed=seed)
            histories[name].append(agent.train(env, num_episodes=EPISODES))
            final_cost[name].append(evaluate_agent(agent, env, num_episodes=20)["mean_total_cost"])
        env = make_env(seed)
        greedy_cost.append(evaluate_greedy(env, num_episodes=20)["mean_total_cost"])
        random_cost.append(evaluate_random(env, num_episodes=20)["mean_total_cost"])

    # --- Learning curves (mean reward per episode across seeds, smoothed) ---
    def smooth(curve, window=10):
        return np.convolve(curve, np.ones(window) / window, mode="valid")

    curves = {name: np.mean([smooth(np.array(curve)) for curve in histories[name]], axis=0) for name in AGENTS}
    x = list(range(len(next(iter(curves.values())))))
    curve_path = line_chart(
        x=x,
        series=curves,
        title="DRL learning curves (mean episode reward, 10-ep smoothed)",
        xlabel="Training episode",
        ylabel="Episode reward (= -cost)",
        path=output_path("study2_learning_curves.png"),
    )

    # --- Final-cost comparison table + bar chart ---
    print(f"{'Policy':<10}{'Mean cost':>12}{'95% CI':>10}")
    rows, bar_values, bar_errors = [], [], []
    labels = list(AGENTS.keys()) + ["Greedy", "Random"]
    datasets = [final_cost[name] for name in AGENTS] + [greedy_cost, random_cost]
    for label, data in zip(labels, datasets):
        ci = confidence_interval(data)
        print(f"{label:<10}{ci['mean']:>12.4f}{ci['half_width']:>10.4f}")
        rows.append({"policy": label, "mean_cost": ci["mean"], "cost_ci": ci["half_width"]})
        bar_values.append(ci["mean"])
        bar_errors.append(ci["half_width"])

    csv_path = results_to_csv(rows, output_path("study2_drl_comparison.csv"))

    # --- Per-seed values + direct (paired) statistical comparison ---
    # Overlapping 95% CIs on the marginal means do not by themselves show that two policies are
    # equivalent (or different); the seeds are shared, so we compare the policies directly with
    # paired tests (Holm-corrected) and report the 95% CI of the paired mean difference.
    groups = dict(zip(labels, datasets))
    per_seed_rows = [{"seed": seed, **{label: float(values[i]) for label, values in groups.items()}} for i, seed in enumerate(SEEDS)]
    results_to_csv(per_seed_rows, output_path("study2_drl_per_seed.csv"))
    sig_rows = compare_all_pairs(groups, paired=True, correction="holm")
    for row in sig_rows:
        diff = np.array(groups[row["group_a"]]) - np.array(groups[row["group_b"]])
        dci = confidence_interval(list(diff))
        row["diff_ci95_low"] = dci["mean"] - dci["half_width"]
        row["diff_ci95_high"] = dci["mean"] + dci["half_width"]
    sig_path = results_to_csv(sig_rows, output_path("study2_drl_significance.csv"))
    print("\nPaired comparisons (Holm-corrected):")
    for r in sig_rows:
        print(f"  {r['group_a']:<7} vs {r['group_b']:<7} diff={r['mean_difference']:+.3f} "
              f"[{r['diff_ci95_low']:+.3f},{r['diff_ci95_high']:+.3f}] p_adj={r['p_adjusted']:.4f} d={r['cohens_d']:+.2f}")
    bar_path = grouped_bar_chart(
        categories=labels,
        series={"mean cost": bar_values},
        errors={"mean cost": bar_errors},
        title="DRL agents vs. heuristics (lower is better)",
        ylabel="Mean weighted cost",
        path=output_path("study2_drl_comparison.png"),
    )

    print(f"\nWrote: {csv_path}\n       {curve_path}\n       {bar_path}")


if __name__ == "__main__":
    main()
