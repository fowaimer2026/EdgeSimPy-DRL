""" Worked example: running the contributions on REAL datasets.

Loads real edge infrastructure (EUA, Melbourne) and a real task workload (Alibaba cluster
trace) and runs two of the contributed studies on them:

    1. Offloading placement (cost-driven vs. random) on the real EUA base-station layout.
    2. Task execution (queueing / response time / energy) of the real Alibaba workload on
       the EUA servers.

By default it uses the small format-accurate samples bundled with EdgeSimPy, so it runs
out of the box. Point it at the full downloaded files for real-scale results:

    python examples/real_dataset_example.py \
        --eua-servers path/to/eua_site.csv --eua-users path/to/eua_users.csv \
        --alibaba path/to/batch_task.csv --alibaba-limit 2000

Downloads:
    EUA      -> https://github.com/PhuLai/eua-dataset
    Alibaba  -> https://github.com/alibaba/clusterdata
    Shanghai -> http://sguangwang.com/TelecomDataset.html
"""

# Python libraries
import argparse

# EdgeSimPy
from edge_sim_py.datasets import load_eua, load_alibaba_tasks, sample_path
from edge_sim_py.drl.offloading_env import OffloadingEnv
from edge_sim_py.workload import ExecutionEngine, place_least_loaded, collect_workload_metrics
from edge_sim_py.energy import EnergyMeter

COST_PARAMS = {"weights": {"latency": 0.5, "energy": 0.3, "monetary": 0.2}}


def evaluate(env, action_fn):
    env.reset()
    total, done, info = 0.0, False, {"placed": 0}
    while not done:
        _, reward, done, info = env.step(action_fn(env))
        total += reward
    return -total, info["placed"] / len(env.services)


def main():
    parser = argparse.ArgumentParser(description="Run the contributions on real datasets.")
    parser.add_argument("--eua-servers", default=sample_path("eua_base_stations_sample.csv"))
    parser.add_argument("--eua-users", default=sample_path("eua_users_sample.csv"))
    parser.add_argument("--alibaba", default=sample_path("alibaba_batch_task_sample.csv"))
    parser.add_argument("--alibaba-limit", type=int, default=None)
    args = parser.parse_args()

    using_samples = args.eua_servers == sample_path("eua_base_stations_sample.csv")
    print(f"Data source: {'bundled samples' if using_samples else 'user-provided files'}\n")

    # --- 1. Offloading on real EUA infrastructure ---
    servers, services = load_eua(args.eua_servers, args.eua_users, seed=1)
    print(f"EUA infrastructure: {len(servers)} edge servers, {len(services)} services (users)")
    env = OffloadingEnv(services, servers, cost_parameters=COST_PARAMS)
    cost_driven, placed_cd = evaluate(env, lambda e: e.greedy_action())
    random_cost, placed_rnd = evaluate(env, lambda e: e.random_action())
    print(f"  offloading cost  -- cost-driven: {cost_driven:.3f} (placed {placed_cd:.0%})" f"   random: {random_cost:.3f} (placed {placed_rnd:.0%})")

    # --- 2. Real Alibaba workload executed on the EUA servers ---
    servers, _ = load_eua(args.eua_servers, args.eua_users, seed=1)  # fresh servers
    tasks = load_alibaba_tasks(args.alibaba, limit=args.alibaba_limit)
    print(f"\nAlibaba workload: {len(tasks)} tasks")
    place_least_loaded(tasks, servers)
    meter = EnergyMeter()
    engine = ExecutionEngine(servers, tasks, energy_meter=meter)
    engine.run()
    metrics = collect_workload_metrics(tasks, engine)
    print(
        f"  execution  -- makespan: {engine.current_step}  mean response: {metrics['mean_response_time']:.2f}"
        f"  completion: {metrics['completion_rate']:.0%}  deadline viol.: {metrics['deadline_violation_rate']:.2%}"
        f"  energy: {metrics['total_energy']:.0f}"
    )


if __name__ == "__main__":
    main()
