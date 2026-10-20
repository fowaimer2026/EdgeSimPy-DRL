# Results Suite

Reproducible experiments that evaluate the EdgeSimPy extensions and generate the tables and
figures in [`RESULTS.md`](RESULTS.md). Every study runs across multiple seeds and reports
means with **95% confidence intervals** via `edge_sim_py.experiments`.

## Studies

| Script | Study | Needs torch |
|---|---|---|
| [`study1_offloading_policies.py`](study1_offloading_policies.py) | Offloading policy comparison (cost-driven vs. latency-greedy vs. random) | no |
| [`study2_drl_vs_heuristics.py`](study2_drl_vs_heuristics.py) | DRL agents (DQN/PPO/A2C) vs. greedy/random | **yes** |
| [`study3_routing_policies.py`](study3_routing_policies.py) | Routing policies under congestion | no |
| [`study4_execution_reliability.py`](study4_execution_reliability.py) | Execution load sweep, failure sweep, model validation | no |
| [`study1_eua_real.py`](study1_eua_real.py) | Offloading policies on the **real EUA (Melbourne)** infrastructure | no (needs the EUA files) |
| [`study1_shanghai_real.py`](study1_shanghai_real.py) | Offloading policies on the **real Shanghai Telecom** infrastructure | no (needs the Shanghai .xlsx + openpyxl) |
| [`study2_eua_real.py`](study2_eua_real.py) | DRL agents (DQN/PPO/A2C) on the **real EUA (Melbourne)** infrastructure | **yes** (needs the EUA files) |
| [`study5_weight_sensitivity.py`](study5_weight_sensitivity.py) | Cost-model weight-sensitivity sweep on real EUA | no (needs EUA files) |
| [`study6_routing_topology.py`](study6_routing_topology.py) | Routing on a larger real-coordinate geographic mesh | no (needs EUA files) |
| [`study7_drl_tuning.py`](study7_drl_tuning.py) | DRL hyperparameter tuning + learning curves (PPO/A2C) | **yes** |
| [`study8_integrated.py`](study8_integrated.py) | **Integrated end-to-end**: placement+routing+execution+energy+failures | no (needs EUA + Alibaba files) |
| [`study4_alibaba_real.py`](study4_alibaba_real.py) | **Real Alibaba** workload on **real EUA** servers: capacity + failure sweeps | no (needs EUA + Alibaba files) |

## Running

```bash
# From the repository root. Windows uses ';' as the PYTHONPATH separator, Linux/macOS ':'.
PYTHONPATH=".;results" python results/study1_offloading_policies.py
PYTHONPATH=".;results" python results/study4_execution_reliability.py

# Or regenerate everything (Study 2 skipped automatically if torch is absent):
PYTHONPATH=".;results" python results/generate_all.py
```

Outputs (CSV tables + PNG figures) are written to [`output/`](output/). The committed
artifacts there were produced with the seeds fixed in each script, so results are
deterministic and reproducible.

## Part 3 (reviewer-response) studies

| Script | Study | Needs torch |
|---|---|---|
| [`study5_weight_sensitivity.py`](study5_weight_sensitivity.py) | Cost-model weight sensitivity | no |
| [`study6_routing_topology.py`](study6_routing_topology.py) | Larger real-coordinate routing mesh | no |
| [`study7_drl_tuning.py`](study7_drl_tuning.py) | DRL hyperparameter tuning + curves | yes |
| [`study8_integrated.py`](study8_integrated.py) | Integrated end-to-end (all layers) | no |
| [`study9_ablation.py`](study9_ablation.py) | Ablation (placement × routing grid) | yes |
| [`study10_scalability.py`](study10_scalability.py) | Scalability & runtime overhead | no |
| [`study11_integration_robustness.py`](study11_integration_robustness.py) | Integration robustness across scales × workloads | no (needs EUA + Alibaba) |
| [`study_significance.py`](study_significance.py) | Statistical significance (paired tests, effect sizes) | no |

See [`docs/PART3_EXPERIMENT_PLAN.md`](../docs/PART3_EXPERIMENT_PLAN.md) for the full plan.
