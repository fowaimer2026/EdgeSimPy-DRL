"""Runs every results study end-to-end, regenerating all CSV tables and figures.

Usage:  python results/generate_all.py

Notes:
  * Study 2 (DRL) requires PyTorch and takes a few minutes; the synthetic studies are
    torch-free and fast.
  * The real-data studies (1b/1c/2b/4b) run only if their datasets are present under
    dataset/ (EUA, Alibaba, Shanghai); otherwise they are skipped with a message.
  * Outputs are written to results/output/.
"""

import os

import study1_offloading_policies
import study3_routing_policies
import study4_execution_reliability
import study_significance

# Dataset files the real-data studies depend on.
EUA_SERVERS = "dataset/eua-dataset/edge-servers/site-optus-melbCBD.csv"
ALIBABA = "dataset/alibaba-clusterdata/batch_task.csv"
SHANGHAI = "dataset/Shanghai/data_6.16.15.xlsx"


def _run(module_name, label, required_files):
    """Imports and runs a study module if its required dataset files exist."""
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"Skipping {label}: missing dataset file(s) {missing}")
        return
    print("=" * 70)
    try:
        module = __import__(module_name)
        module.main()
    except ImportError as error:
        print(f"Skipping {label}: {error}")


def main():
    # --- Synthetic studies (always run) ---
    print("=" * 70)
    study1_offloading_policies.main()
    print("=" * 70)
    study3_routing_policies.main()
    print("=" * 70)
    study4_execution_reliability.main()
    print("=" * 70)
    study_significance.main()

    try:
        print("=" * 70)
        import study2_drl_vs_heuristics

        study2_drl_vs_heuristics.main()
    except ImportError:
        print("Skipping Study 2 (DRL): PyTorch not installed.")

    # --- Real-data studies (run if datasets are present) ---
    _run("study1_eua_real", "Study 1b (real EUA offloading)", [EUA_SERVERS])
    _run("study1_shanghai_real", "Study 1c (real Shanghai offloading)", [SHANGHAI])
    _run("study2_eua_real", "Study 2b (real EUA DRL)", [EUA_SERVERS])
    _run("study4_alibaba_real", "Study 4b (real Alibaba execution)", [EUA_SERVERS, ALIBABA])
    _run("study5_weight_sensitivity", "Study 5 (cost-model weight sensitivity)", [EUA_SERVERS])
    _run("study6_routing_topology", "Study 6 (routing on real-coordinate mesh)", [EUA_SERVERS])
    _run("study8_integrated", "Study 8 (integrated end-to-end)", [EUA_SERVERS, ALIBABA])
    _run("study9_ablation", "Study 9 (ablation)", [EUA_SERVERS, ALIBABA])
    _run("study11_integration_robustness", "Study 11 (integration robustness)", [EUA_SERVERS, ALIBABA])

    print("=" * 70)
    import study10_scalability

    study10_scalability.main()

    try:
        print("=" * 70)
        import study7_drl_tuning

        study7_drl_tuning.main()
        print("=" * 70)
        import study7b_fair_tuning

        study7b_fair_tuning.main()
    except ImportError:
        print("Skipping Studies 7/7b (DRL tuning): PyTorch not installed.")


if __name__ == "__main__":
    main()
