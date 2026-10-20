# Part 3 — Practical Work Plan (Experiments, Code, Statistics)

This plan lists the coding / simulation / statistical tasks needed to fully answer the
review. Each task specifies **required changes**, **inputs/datasets**, **metrics**, and
**expected outputs**. Status tags: ✅ done, 🟡 partially done, ⬜ new.

**Existing building blocks** (already in the repo, reused by every task below):
`edge_sim_py.experiments` (multi-seed runner, CIs, sweeps, CSV/plots) · the offloading
`OffloadingEnv` + cost models · `components.routing` policies · `workload` execution engine +
`energy` + `reliability` · dataset loaders (`datasets.load_eua/load_alibaba_tasks/load_shanghai`)
· the results suite (`results/study1..study8`).

---

## 1. Ablation Study ⬜ (extends Study 8 🟡)

**Goal.** Isolate each module's contribution by toggling it on/off, so the paper can state
"module X contributes Δ of metric Y."

**Required changes.**
- New script `results/study9_ablation.py` generalizing `study8_integrated.py` from 2 stacks to
  a full **placement × routing** grid. Two axes:
  - *placement* ∈ {`random`, `cost-greedy`, `DRL`}
  - *routing* ∈ {`shortest-path`, `latency-aware`}
- Add a **DRL-placement adapter for the task stream**: train a `DQNAgent` on the scenario's
  server set (action space = #servers), then use its greedy actions to place tasks in the
  integrated pipeline (reuse `edge_sim_py.drl.policy` pattern). This is the only genuinely new
  code; the cost/random placements already exist in `study8`.
- Configuration → module mapping (covers the 7 requested configs):

  | Config | Placement | Routing | Modules enabled |
  |---|---|---|---|
  | Baseline | random | shortest-path | none |
  | Cost only | cost-greedy | shortest-path | cost model |
  | Routing only | random | latency-aware | routing |
  | DRL only | DRL | shortest-path | DRL (cost as reward) |
  | Cost + routing | cost-greedy | latency-aware | cost + routing |
  | DRL + routing / **Full** | DRL | latency-aware | cost + routing + DRL |

**Inputs/datasets.** Real EUA base stations on a geographic mesh (`site-optus-melbCBD.csv`) +
real **Alibaba** task stream (`batch_task.csv`, windowed). Also run the torch-free cells
(random/cost × SP/LA) on synthetic scenarios for a fast reproducible variant.

**Metrics.** End-to-end **median** response time (robust), network transit delay, **deadline-
violation rate**, **total energy**, completion rate; per-module Δ vs. the baseline and marginal
Δ of adding each module last.

**Expected outputs.** `results/output/study9_ablation.csv` (one row per config, mean ± 95% CI
per metric); a grouped bar chart per metric; an **ablation contribution table** decomposing each
metric into per-module contributions; write-up section "Study 9 — Ablation" in `RESULTS.md`.

---

## 2. Fair Hyperparameter Exploration 🟡 (extends Study 7)

**Goal.** Give DQN, PPO, A2C **comparable optimization budgets** so the algorithm comparison is
fair, and document the protocol.

**Required changes.**
- Extend `results/study7_drl_tuning.py` into `study7b_fair_tuning.py` that sweeps **all three**
  agents (not just PPO/A2C) over an identical grid and an **equal environment-step budget**:
  - Same `episodes × steps_per_episode` for every agent (compute the per-agent step count and
    hold it fixed; on-policy and off-policy then see the same amount of data).
  - Search: learning-rate ∈ {1e-4, 3e-4, 1e-3}; DQN also {`target_update_interval`, `epsilon_decay`},
    PPO/A2C {`entropy_coef`, `gae_lambda`}. Keep the grid size equal per agent (e.g. 6 configs each).
  - ≥ 5 seeds per config; report best config + full grid.

**Inputs/datasets.** Synthetic offloading scenario (fast, reproducible) for the grid; confirm the
winner on real-EUA (`study2_eua_real`).

**Metrics.** Final mean weighted cost (95% CI); episodes-to-threshold (sample efficiency);
wall-clock training time per agent (budget transparency).

**Expected outputs.** `study7b_fair_tuning_grid.csv`; per-agent learning curves at matched budget;
a **budget-transparency table** (steps, wall-clock, #configs per agent).

**If full retuning is too expensive — document instead** (recommended minimum):
- Report the exact budget used per algorithm (episodes, steps, #configs, seeds, wall-clock).
- State known limitations (e.g. "DQN used default target-network/ε schedules; a wider DQN search
  was not run due to compute").
- **Fairness impact + recommended paper language:**
  > "All agents were trained for an equal environment-step budget of N steps and an equal-size
  > hyperparameter grid (K configurations each, S seeds). We tuned learning rate for all agents
  > and the entropy coefficient for the policy-gradient agents; DQN's target-update and
  > exploration schedules were left at defaults. Results should therefore be read as *'best of an
  > equal, modest budget'* rather than fully optimized performance for any single algorithm."

---

## 3. Statistical Analysis & Significance ⬜ (code helper implemented)

**Goal.** Replace "eyeballing CIs" with hypothesis tests, effect sizes and multiple-comparison
control.

**Required changes.**
- New module `edge_sim_py/experiments/significance.py` (implemented alongside this plan) with:
  - **Paired tests** across shared seeds (the correct model, since all configs use the same seed
    set): paired *t*-test and Wilcoxon signed-rank (non-parametric fallback).
  - **Two-independent-group** tests where needed: Welch's *t*-test, Mann–Whitney U.
  - **Omnibus** for >2 groups (routing policies; DRL algorithms): one-way / repeated-measures
    ANOVA, and Kruskal–Wallis / Friedman (non-parametric).
  - **Effect sizes:** Cohen's *d* (paired and pooled), Cliff's delta (non-parametric), η² for ANOVA.
  - **Multiple-comparison correction:** Holm–Bonferroni (default) and Benjamini–Hochberg (FDR).
- SciPy is used when available (already a dependency for CIs) with pure-NumPy fallbacks.

**Test-selection guide (which test for which comparison):**

| Comparison | Structure | Primary test | Effect size |
|---|---|---|---|
| Full stack vs. baseline | paired (same seeds) | paired *t* / Wilcoxon | Cohen's *d* / Cliff's δ |
| 4 routing policies | paired, >2 groups | repeated-measures ANOVA / Friedman | η² |
| 3 DRL algorithms | paired, >2 groups | RM-ANOVA / Friedman + post-hoc | η² + pairwise *d* |
| EUA vs. Shanghai (cross-dataset) | independent | Welch *t* / Mann–Whitney | Cohen's *d* / Cliff's δ |

**Inputs/datasets.** The per-seed metric arrays already produced by `run_experiment(...)["raw"]`
for every study.

**Metrics/outputs.** For each headline comparison, report **mean difference + 95% CI, p-value
(corrected), and effect size** in one table. `results/study_significance.py` (implemented) does
this over 25 paired seeds and writes `study_significance.csv`. It already surfaces an honest
result the reviewer will value: cost-driven vs. latency-greedy is **not** significant on the
synthetic scenario (p≈0.57, d=0.12), while both beat random with large effects (d≈−2.4); routing
policies differ by a Friedman omnibus (χ²=50.9, p=5×10⁻¹¹) with corrected pairwise effects.

**Guardrails against overclaiming (paper language).**
- Only claim significance after multiple-comparison correction; report corrected p-values.
- Pair effect size with every p-value ("statistically significant *and* d = 0.9, a large effect").
- For overlapping-CI metrics (e.g. Study 8 median response) state explicitly "not statistically
  significant at α = 0.05".

---

## 4. Scalability & Generalizability ⬜

**Goal.** Quantify the runtime/resource overhead of the five added layers and bound the claims.

**Required changes.**
- New script `results/study10_scalability.py` measuring:
  - **Per-layer runtime overhead**: wall-clock of a fixed workload with each layer toggled on
    (cost model, routing, execution, energy meter, failure model), reported as µs/decision and
    % over the bare simulator.
  - **Scaling curves**: wall-clock and peak memory vs. #servers ∈ {25, 50, 100, 200} and
    #tasks ∈ {500, 1k, 2k, 4k} (use `tracemalloc` / `time.perf_counter`).

**Inputs/datasets.** Synthetic scenarios (to scale freely) + one real-EUA point for calibration.

**Metrics.** Wall-clock (s), throughput (decisions/s), peak memory (MB), overhead % per layer.

**Expected outputs.** `study10_scalability.csv`; runtime-vs-size and memory-vs-size plots; a
"Runtime overhead" paragraph.

**Generalizability phrasing (Results/Discussion):**
> "Topology sizes here (≤ 200 nodes, ≤ 4k tasks) are modest relative to production edge
> deployments. The results establish relative ordering of policies and the direction/magnitude of
> per-module effects under controlled conditions; absolute latencies/energy and the persistence of
> effect sizes at production scale would require further validation on larger infrastructures and
> live traces. The added layers impose < X% runtime overhead, so scaling is bounded by the base
> simulator rather than by the contributed modules."

---

## 5. Reproducibility & Extensibility ⬜ (mostly structural)

**Goal.** A clean public artifact and a documented path to add new DRL algorithms without touching
the simulator core.

**Required changes.**
- **Code organization** (already largely in place): core simulator (`edge_sim_py/components`,
  `simulator.py`) vs. opt-in modules (`cost_models`, `routing`, `workload`, `energy`, `reliability`,
  `network`, `drl`, `datasets`, `experiments`). Document this split in a top-level `ARTIFACT.md`.
- **Configuration files per experiment**: add `results/configs/*.yaml` (or `.json`) capturing each
  study's parameters (seeds, sizes, weight vectors, dataset paths), and make the study scripts read
  a `--config` file. Enables one-command reproduction and parameter transparency.
- **Runner scripts**: `results/generate_all.py` (exists) + a `run_ablation.sh` / `run_drl.sh`
  wrapper and a pinned `requirements.txt` / `poetry.lock` for the exact environment.

**Adding new DRL agents (SAC / TD3 / Rainbow) without modifying the core.**
- **Interface** — a new agent only needs the existing duck-typed contract already used by all three
  agents and the trainer/evaluator:
  ```python
  class MyAgent:
      def __init__(self, state_dim, action_dim, **hyperparams): ...
      def act(self, state, mask, greedy=False) -> int: ...     # respect the action mask
      def train(self, env, num_episodes) -> list[float]: ...    # returns per-episode reward
  ```
- **API hooks** — register in `edge_sim_py/drl/agents/__init__.py`; the environment
  (`OffloadingEnv`), evaluation (`evaluate_agent`) and the live-sim adapter (`drl_offloading_policy`)
  are agent-agnostic and need no change. Discrete actions (Rainbow DQN) drop in directly; continuous
  agents (SAC/TD3) either use a discretized action head or a `argmax` over a scored action set.
- **Example template** — ship `edge_sim_py/drl/agents/agent_template.py` (a documented skeleton
  implementing the three methods) so contributors copy-edit rather than reverse-engineer.
- **Expected outputs.** `ARTIFACT.md`, `results/configs/`, `agent_template.py`; a "how to add an
  agent" doc section; unchanged 74-test suite still green.

---

## Priority & effort

| Task | Reviewer weight | Effort | Depends on |
|---|---|---|---|
| 1 Ablation | high | medium (DRL-in-stream adapter) | Study 8 |
| 3 Significance module | high | **low (done here)** | experiment harness |
| 2 Fair tuning | medium | medium (compute) | Study 7 |
| 4 Scalability | medium | low | — |
| 5 Artifact/extensibility | medium | low (mostly docs + template) | — |

Recommended order: **3 (done) → 1 → 4 → 5 → 2**, so the statistical machinery is in place before
the ablation/tuning numbers are reported.
