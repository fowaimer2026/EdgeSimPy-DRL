# DRL Hyperparameters, Implementation, and Training Budget

This page documents the exact configuration of the three deep-reinforcement-learning
offloading agents (DQN, PPO, A2C) shipped in `edge_sim_py.drl`, the training budget used
in the experiments reported in [`results/RESULTS.md`](../results/RESULTS.md), and the
rationale behind the number of seeds. It is intended as the single citable reference for
the reproducibility of the DRL studies.

## 1. Implementation

The agents are **implemented from scratch in PyTorch** — they are *not* wrappers around
Stable-Baselines3, RLlib, Tianshou, or any other RL library. The only third-party
dependency is PyTorch itself (tensors, autograd, `torch.optim.Adam`, and
`torch.distributions.Categorical`). This keeps the agents self-contained, auditable, and
free of a heavyweight RL framework as a hard dependency of EdgeSimPy.

Source files:

| Component | File |
|---|---|
| DQN (Double + Dueling) | `edge_sim_py/drl/agents/dqn_agent.py` |
| PPO (clipped objective + GAE) | `edge_sim_py/drl/agents/ppo_agent.py` |
| A2C (synchronous advantage AC + GAE) | `edge_sim_py/drl/agents/a2c_agent.py` |
| Shared networks (`QNetwork`, `ActorCritic`) | `edge_sim_py/drl/networks.py` |
| Replay buffer | `edge_sim_py/drl/replay_buffer.py` |
| Offloading environment (MDP) | `edge_sim_py/drl/environment.py` |

All three agents share the same interface — `act(state, mask, greedy=False)` and
`train(env, num_episodes)` — and all use **action masking**: servers without the capacity
to host the current service are masked out (`-inf` in the Q-values / policy logits) both
when acting and when computing bootstrap/GAE targets, so an agent never selects or is
updated toward an infeasible placement.

### Network architecture

Both network types use a fully-connected ReLU MLP with hidden layers `(128, 128)`.

- **`QNetwork`** (DQN): dueling by default — a shared feature trunk feeds a scalar
  state-value head and an action-advantage head, recombined as
  `Q = V + (A − mean(A))`.
- **`ActorCritic`** (PPO/A2C): a shared trunk feeds a policy-logits head and a scalar
  value head.

## 2. Hyperparameters

Defaults as defined in each agent's constructor. These are the values used in the studies
unless a study's own grid overrides them (Study 7b sweeps the learning rate and one
secondary exploration parameter per agent — see §4).

### Shared across all agents

| Parameter | Value |
|---|---|
| Hidden layers | `(128, 128)`, ReLU |
| Discount factor γ | `0.95` |
| Optimizer | Adam |
| Device | CPU |

### DQN (`DQNAgent`)

| Parameter | Value |
|---|---|
| Learning rate | `1e-3` |
| Replay buffer capacity | `50 000` |
| Batch size | `64` |
| Target-network update interval | `200` learn steps |
| ε (start → end) | `1.0 → 0.05` |
| ε decay (per episode) | `0.995` |
| Double DQN | enabled |
| Dueling network | enabled |
| Loss | Smooth L1 (Huber) |
| Gradient-norm clip | `10.0` |

### PPO (`PPOAgent`)

| Parameter | Value |
|---|---|
| Learning rate | `3e-4` |
| GAE λ | `0.95` |
| Clip ε | `0.2` |
| Value-loss coefficient | `0.5` |
| Entropy coefficient | `0.01` |
| Update epochs per rollout | `4` |
| Minibatch size | `64` |
| Rollout length | every `4` episodes |
| Value loss | MSE |
| Gradient-norm clip | `0.5` |
| Advantage normalization | per-rollout (zero mean, unit std) |

### A2C (`A2CAgent`)

| Parameter | Value |
|---|---|
| Learning rate | `7e-4` |
| GAE λ | `0.95` |
| Value-loss coefficient | `0.5` |
| Entropy coefficient | `0.01` |
| Update frequency | every `1` episode (one gradient step per rollout) |
| Value loss | MSE |
| Gradient-norm clip | `0.5` |
| Advantage normalization | per-rollout (zero mean, unit std) |

A2C is the synchronous form of A3C: the same advantage actor-critic update, collected from
sequential rollouts rather than asynchronous worker threads.

## 3. Seeds and determinism

Every agent seeds `torch`, Python `random`, and `numpy` from the single `seed` constructor
argument. Each experiment runs over a **range of seeds** and reports the mean with a 95%
confidence interval (Student-t), computed by
`edge_sim_py.experiments.confidence_interval`.

### On the number of seeds

The DRL studies were re-run with **more seeds than the original submission** specifically
to address the reviewer's concern about wide / overlapping confidence intervals. On-policy
agents (PPO, A2C) have higher run-to-run variance than value-based DQN, so more seeds are
needed to separate the policies with confidence. The seed counts used are:

| Study | Script | Seeds |
|---|---|---|
| 2 — DRL vs. heuristics (synthetic) | `results/study2_drl_vs_heuristics.py` | 16 |
| 2b — DRL on real EUA (Melbourne) | `results/study2_eua_real.py` | 12 |
| 7b — fair tuning (equal budget) | `results/study7b_fair_tuning.py` | 10 |
| 9 — placement × routing ablation | `results/study9_ablation.py` | 10 |

Increasing the seed count tightened the confidence intervals and, in one case, **changed a
claim**: on the synthetic benchmark (Study 2) DQN and the min-cost greedy baseline now have
overlapping CIs, so the earlier "DQN clearly beats greedy" statement was softened to
"statistically comparable." On the **real EUA** infrastructure (Study 2b) the DQN advantage
(~37% lower cost than greedy) held up and became tighter. We report both honestly; see
`results/RESULTS.md` for the full numbers.

## 4. Training budget (fairness and cost)

Study 7b (`results/study7b_fair_tuning.py`) gives all three agents a **comparable
optimization budget** — the same number of training episodes (300), the same number of
seeds (10), and an equal-size hyperparameter grid (6 configs each: 3 learning rates × 2
agent-appropriate secondary values). The secondary parameter is `epsilon_decay ∈ {0.99,
0.995}` for DQN and `entropy_coef ∈ {0.01, 0.05}` for PPO/A2C. The study writes a
budget-transparency table (`results/study7b_budget.csv`) alongside the results so the
fairness of the comparison is explicit.

### Measured wall-clock training cost (Study 7b, full 6-config × 10-seed grid, CPU)

| Agent | Best final cost | Wall-clock (full grid) |
|---|---|---|
| DQN | ~0.006 | ~1375 s |
| PPO | ~0.664 | ~492 s |
| A2C | ~1.167 | ~307 s |
| greedy baseline | ~0.008 | negligible |

DQN reaches the lowest cost but is the most expensive to train (≈2.8× PPO, ≈4.5× A2C) —
its per-step replay-buffer updates cost more than the on-policy agents' periodic rollout
updates. All runs are single-threaded on CPU; no GPU is required. These numbers are
machine-dependent and are reported as orders of magnitude, not benchmarks.

## 5. Reproducing the studies

```bash
# From the repository root, with PyTorch installed.
python results/study2_drl_vs_heuristics.py     # synthetic, 16 seeds
python results/study2_eua_real.py              # real EUA Melbourne, 12 seeds
python results/study7b_fair_tuning.py          # equal-budget tuning + cost table, 10 seeds
python results/study9_ablation.py              # placement x routing ablation, 10 seeds
```

Each script writes its CSV(s) and figure(s) into `results/` and prints the mean ± 95% CI
table to stdout. Results are deterministic given the seed range.
