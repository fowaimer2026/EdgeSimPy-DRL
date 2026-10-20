"""Statistical-significance report for the main comparisons (Part 3, item 3).

Demonstrates the significance module on two headline comparisons, collecting per-seed values
(paired across the shared seed set) and reporting, for every pair, the mean difference, the
Holm-corrected p-value and an effect size (Cohen's d / Cliff's delta):

    1. Offloading policies -- cost-driven vs. latency-greedy vs. random (paired t-tests).
    2. Routing policies    -- shortest-path vs. least-congested vs. latency-aware vs. ECMP
       (Friedman omnibus + pairwise Wilcoxon).

Synthetic, torch-free and fast, so it reproduces anywhere. Writes study_significance.csv.
"""

# EdgeSimPy
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.network_switch import NetworkSwitch
from edge_sim_py.components.network_link import NetworkLink
from edge_sim_py.components.routing import ShortestPathRouting, LeastCongestedRouting, LatencyAwareRouting, ECMPRouting
from edge_sim_py.components.routing.base_routing import link_utilization
from edge_sim_py.components.cost_models.latency_cost_model import LatencyCostModel
from edge_sim_py.drl.offloading_env import OffloadingEnv
from edge_sim_py.drl.scenario import random_scenario
from edge_sim_py.experiments import compare_all_pairs, friedman_test, results_to_csv

# Python libraries
import random

import numpy as np

from _common import output_path, evaluate_offloading_policy

SEEDS = range(25)
COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}


# ---------------------------------------------------------------- offloading policies


def _offloading_per_seed():
    """Returns {policy: [cost per seed]} for cost-driven / latency-greedy / random."""

    def latency_greedy(env):
        service = env.services[env.pointer]
        costs = [LatencyCostModel.get_cost(service, s, None, COST_PARAMS) for s in env.servers]
        masked = [c if v else np.inf for c, v in zip(costs, env.action_mask)]
        return int(np.argmin(masked))

    policies = {"cost_driven": lambda e: e.greedy_action(), "latency_greedy": latency_greedy, "random": lambda e: e.random_action()}
    out = {name: [] for name in policies}
    for seed in SEEDS:
        services, servers = random_scenario(num_services=12, num_servers=4, seed=seed)
        rng = random.Random(seed)
        for s in servers:
            s.cost_model_parameters = {"price_per_cpu_unit_time": rng.choice([0.02, 0.05, 0.2])}
        env = OffloadingEnv(services, servers, cost_parameters=COST_PARAMS)
        for name, fn in policies.items():
            out[name].append(evaluate_offloading_policy(env, fn)["total_cost"])
    return out


# ---------------------------------------------------------------- routing policies


def _routing_per_seed():
    """Returns {policy: [bottleneck utilization per seed]} on a randomized parallel-path topology."""
    policies = {
        "shortest_path": (ShortestPathRouting, None),
        "least_congested": (LeastCongestedRouting, None),
        "latency_aware": (LatencyAwareRouting, {"alpha": 0.5, "beta": 0.5}),
        "ecmp": (ECMPRouting, {"weight": None}),
    }
    out = {name: [] for name in policies}
    for seed in SEEDS:
        rng = random.Random(seed)
        for cls in (Topology, NetworkSwitch, NetworkLink):
            cls._instances = []
            cls._object_count = 0
        topology = Topology()
        src, dst = NetworkSwitch(), NetworkSwitch()
        topology.add_node(src)
        topology.add_node(dst)
        for _ in range(6):
            mid = NetworkSwitch()
            topology.add_node(mid)
            util = rng.uniform(0, 0.95)
            for a, b in ((src, mid), (mid, dst)):
                link = NetworkLink()
                link["topology"] = topology
                link["nodes"] = [a, b]
                link["delay"] = rng.randint(1, 10)
                link["bandwidth"] = 100
                link["bandwidth_demand"] = util * 100
                topology.add_edge(a, b, **link)
        for name, (policy, params) in policies.items():
            path = policy.get_path(topology, src, dst, parameters=params)
            out[name].append(max((link_utilization(topology[path[i]][path[i + 1]]) for i in range(len(path) - 1)), default=0.0))
    return out


def _print_pairs(title, rows):
    print(f"\n{title} (paired, Holm-corrected):")
    print(f"  {'comparison':<32}{'mean diff':>10}{'p_adj':>9}{'d':>8}  effect / significant")
    for r in rows:
        sig = "SIGNIFICANT" if r["significant"] else "n.s."
        print(
            f"  {r['group_a'] + ' vs ' + r['group_b']:<32}{r['mean_difference']:>10.3f}{r['p_adjusted']:>9.4f}" f"{r['cohens_d']:>8.2f}  {r['effect']} / {sig}"
        )


def main():
    print(f"Statistical significance report ({len(list(SEEDS))} seeds, paired tests, Holm correction)")

    csv_rows = []

    off = _offloading_per_seed()
    off_pairs = compare_all_pairs(off, paired=True, correction="holm")
    _print_pairs("Offloading policies (mean weighted cost)", off_pairs)
    for r in off_pairs:
        csv_rows.append(
            {
                "comparison_type": "offloading",
                **{k: r[k] for k in ("group_a", "group_b", "mean_difference", "p_value", "p_adjusted", "cohens_d", "cliffs_delta", "effect", "significant")},
            }
        )

    rout = _routing_per_seed()
    omnibus = friedman_test(rout)
    print(
        f"\nRouting policies (bottleneck utilization) -- Friedman omnibus: "
        f"chi2={omnibus['statistic']:.2f}, p={omnibus['p_value']:.4g}, Kendall's W={omnibus['kendalls_w']:.3f}"
    )
    rout_pairs = compare_all_pairs(rout, paired=True, correction="holm")
    _print_pairs("Routing policies (bottleneck utilization)", rout_pairs)
    for r in rout_pairs:
        csv_rows.append(
            {
                "comparison_type": "routing",
                **{k: r[k] for k in ("group_a", "group_b", "mean_difference", "p_value", "p_adjusted", "cohens_d", "cliffs_delta", "effect", "significant")},
            }
        )

    path = results_to_csv(csv_rows, output_path("study_significance.csv"))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
