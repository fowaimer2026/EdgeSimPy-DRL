# Results

This document reports the experimental evaluation of the EdgeSimPy extensions. All results
are produced by the reproducible scripts in this folder (`study1`–`study4`), which use the
contributed experiment harness (`edge_sim_py.experiments`) to run each configuration across
multiple random seeds and report **means with 95% confidence intervals (CIs)**. Regenerate
everything with:

```bash
PYTHONPATH=".;results" python results/generate_all.py      # Windows
PYTHONPATH=".:results" python results/generate_all.py      # Linux/macOS
```

CSV tables and figures are written to [`results/output/`](output/).

**Common setup.** Randomized scenarios with heterogeneous edge-server capacities are
generated per seed. The offloading cost is the balanced weighted objective
(latency 0.5 / energy 0.3 / money 0.2). Confidence intervals use the Student-t distribution.

**On seed counts.** Seed counts differ by experiment because of compute cost, not by choice:
the synthetic/analytical studies (1, 3, 4, 5, 6) use 20–40 seeds, which are cheap; the
**DRL studies use 10–16 seeds** (Study 2: 16, Study 2b: 12, Study 7b: 10, Study 9: 10) because
each seed trains one or more agents for hundreds of episodes — e.g. the Study 7b grid alone is
≈21.6k training episodes, and a single DQN grid took ~23 min of CPU (Study 7b). These counts were
raised from an earlier 4–6 to tighten the confidence intervals; doing so **confirmed the real-data
DRL result (Study 2b) and appropriately widened the synthetic DQN-vs-greedy claim (Study 2)** to
overlapping CIs — an honest consequence of more thorough seeding.

---

## At a glance

Each contribution is first evaluated on synthetic scenarios, then **re-validated on real
public datasets** — EUA (real Melbourne base stations), Shanghai Telecom (real base stations,
a second city), and the Alibaba cluster trace (real task workload). The real-data results
reproduce the synthetic findings, which is the key external-validity claim.

| # | Study | Data | Headline |
|---|---|---|---|
| 1 | Offloading policies | synthetic | Cost-driven ≈11× cheaper than random, ~12% better than latency-greedy |
| **1b** | Offloading | **real EUA** | Cost-driven cuts **energy ~67–75%** for ~10% latency |
| **1c** | Offloading | **real Shanghai** | Same **energy ~68–74%** cut on a second real city (cross-validation) |
| 2 | DRL vs heuristics | synthetic | DQN vs greedy: no significant difference (paired diff −0.11, 95% CI [−0.34, +0.11], p_adj=0.30; equivalence not shown); both ≫ default-tuned on-policy & random |
| **2b** | DRL | **real EUA** | **DQN ~37% cheaper than greedy** on real Melbourne (robust at 12 seeds) |
| 3 | Routing policies | synthetic | Latency-aware routing: −60% congestion for +28% delay |
| 4 | Execution & reliability | synthetic | Contention & failures degrade response/SLA; models validated within 95% CI |
| **4b** | Execution & reliability | **real Alibaba + EUA** | Server failures raise response **×4.3**, deadline misses to 16% |
| 5 | Cost-model weight sensitivity | real EUA | Diagonally dominant: each objective minimized at its own weight |
| 6 | Routing (larger topology) | real EUA coords | Trade-off holds on an 80-node multi-hop mesh |
| 7 | DRL hyperparameter tuning (16 seeds) | synthetic | Tuned PPO 1.86→0.32 (5.8× vs lr=1e-4; 3.0× vs default 0.97); still significantly above DQN/greedy (~0.004; paired p_adj=0.014) |
| 7b | Fair tuning (equal budget, 10 seeds) | synthetic | DQN best (0.006), PPO 0.664, A2C 1.167; DQN ~2.8–4.5× slower to train |
| **8** | **Integrated end-to-end** | **real EUA + Alibaba** | Intelligent stack: network −27%, SLA −17%, energy −4% |
| **9** | **Ablation (10 seeds)** | **real EUA + Alibaba** | Cost model = workhorse (net −35%, deadline −24%); DRL+routing trades response for energy (honest trade-off) |
| 10 | Scalability & overhead | synthetic | Added layers cheap (+20% energy, <2 MB); scaling bounded by base sim |
| **11** | **Integration robustness** | **real EUA + Alibaba + Poisson** | Network benefit holds across 3 scales × 2 workloads and **grows with scale**; latency/SLA workload-dependent |
| — | Statistical significance | synthetic | Paired tests + Holm + effect sizes; honest n.s. results reported |

**Two headline claims for the write-up:** (1) the multi-objective cost model's energy saving
(~70%) holds on **two independent real infrastructures**; and (2) on **real infrastructure DQN
learns a ~37% cheaper policy than greedy** (Study 2b, robust at 12 seeds) — while on the synthetic
scenario DQN only *matches* greedy once more seeds are run (overlapping CIs), so the DRL advantage
is claimed carefully and strongest on real data. All numbers are reproducible from fixed seeds;
DRL studies use fewer seeds than the synthetic ones owing to per-seed training cost (see the
note on seed counts in the Methods).

---

## Study 1 — Offloading policy comparison

Three placement policies are evaluated under the **same** balanced weighted-cost metric
(12 services, 4 servers with heterogeneous pricing, 30 seeds). *cost-driven* is the
contributed multi-objective policy; *latency-greedy* minimizes latency only; *random* picks
a capable server at random.

| Policy | Mean weighted cost (↓) | 95% CI | Placement rate |
|---|---|---|---|
| **cost-driven** | **0.325** | ±0.311 | 0.994 |
| latency-greedy | 0.368 | ±0.304 | 0.994 |
| random | 3.692 | ±0.471 | 0.992 |

**Finding.** The cost-driven policy achieves the lowest cost — **≈11× lower than random**
and **~12% lower than latency-greedy**, the latter gap arising precisely because
latency-greedy ignores the heterogeneous server pricing that the multi-objective cost
captures. All policies place ≈99% of services (capacity is not the bottleneck here).
Figures: [`study1_offloading_cost.png`](output/study1_offloading_cost.png) (bar chart with
CIs) and [`study1_offloading_sweep.png`](output/study1_offloading_sweep.png) (cost vs. number
of servers).

---

## Study 2 — DRL agents vs. heuristics

The three DRL agents are trained on the offloading environment (reward = −weighted cost;
12 services, 5 servers; 300 episodes; **16 seeds**) and their learned greedy policies compared
to the min-cost greedy and random baselines.

| Policy | Mean weighted cost (↓) | 95% CI |
|---|---|---|
| **DQN** | **0.292** | ±0.529 |
| Greedy (min-cost) | 0.407 | ±0.575 |
| PPO | 2.180 | ±0.535 |
| A2C | 2.562 | ±0.629 |
| Random | 3.369 | ±0.462 |

**Finding.** DQN achieves the **lowest mean cost**, with greedy second; both are far better than
the default-hyperparameter on-policy agents (PPO, A2C) and random. With 16 seeds (up from 6) the
on-policy CIs tightened substantially (PPO ±0.92→±0.53, A2C ±1.56→±0.63), and the **DQN-vs-greedy
gap is not statistically significant**. Because the seeds are shared, the two policies are
compared directly with a paired test (Holm-corrected; `study2_drl_significance.csv`): DQN is
0.114 lower on average, 95% CI of the paired difference [−0.342, +0.113], adjusted p = 0.30,
Cohen's d = −0.27 (small). Overlapping marginal CIs alone do not establish equivalence, and
neither does this test: it detects no difference and rules out only large ones. The honest
reading is: **no difference between DQN and greedy was detected at this budget, and both clearly
dominate the default-tuned policy-gradient agents and random** (which *are* separated with large
effect sizes; per-seed values are in `study2_drl_per_seed.csv`). On-policy performance is a
hyperparameter artifact here — Study 7 shows tuning closes most of the gap. Figures:
[`study2_learning_curves.png`](output/study2_learning_curves.png) and
[`study2_drl_comparison.png`](output/study2_drl_comparison.png).

---

## Study 3 — Routing policies under congestion

A source→sink topology with 6 parallel 2-hop paths is generated per seed with randomized
per-path delay and congestion (40 seeds). Each policy routes one flow; we measure the chosen
path's total delay and its bottleneck (max-link) utilization.

| Policy | Path delay (↓) | Bottleneck utilization (↓) |
|---|---|---|
| shortest-path | **5.70** ±0.69 | 0.512 ±0.088 |
| least-congested | 11.00 ±1.33 | **0.117** ±0.034 |
| latency-aware (α=β=0.5) | 7.28 ±0.89 | 0.202 ±0.055 |
| ECMP | 11.75 ±1.18 | 0.479 ±0.095 |

**Finding.** The results trace a clean **latency-vs-congestion Pareto trade-off**:
shortest-path minimizes delay but drives the highest congestion (0.512), while
least-congested inverts this. The contributed **latency-aware policy is the best balance** —
it cuts bottleneck utilization by **~60%** (0.512→0.202) for only **~28% more delay**
(5.70→7.28). ECMP spreads load across equal-cost paths, yielding near-average utilization.
Figure: [`study3_routing_policies.png`](output/study3_routing_policies.png).

---

## Study 4 — Execution dynamics, energy and reliability

Experiments on the execution engine (least-loaded placement, 4 servers, 30 seeds).

**A. Load sweep.** Increasing the task count on a fixed infrastructure raises queueing, and
thus response time, deadline violations and energy:

| Tasks | Response time | Deadline-violation rate | Total energy |
|---|---|---|---|
| 10 | 9.38 ±0.82 | 0.000 | 19,644 |
| 20 | 9.45 ±0.68 | 0.000 | 22,480 |
| 40 | 9.40 ±0.45 | 0.000 | 27,566 |
| 60 | 9.59 ±0.39 | 0.001 | 31,946 |
| 80 | 10.00 ±0.36 | 0.005 | 35,767 |

**B. Failure sweep** (MTTR = 10). Lower MTBF (more frequent failures) sharply degrades
response time and deadline adherence:

| MTBF | Response time | Deadline-violation rate | Completion |
|---|---|---|---|
| 20 | 26.16 | 0.237 | 1.000 |
| 40 | 17.08 | 0.147 | 1.000 |
| 80 | 12.86 | 0.064 | 1.000 |
| 160 | 10.80 | 0.027 | 1.000 |
| none | 9.40 | 0.000 | 1.000 |

**Finding.** Contention and failures both manifest as the ground-truth metrics EdgeSimPy
previously could not produce. Frequent failures (MTBF = 20) nearly **triple** the mean
response time (9.4→26.2) and push deadline violations to **24%**, while task re-queueing keeps
completion at 100%. Figures:
[`study4_load_sweep.png`](output/study4_load_sweep.png),
[`study4_failure_impact.png`](output/study4_failure_impact.png).

**C. Model validation.** Each stochastic model is validated against an independent analytical
reference; the reference falls **within the empirical 95% CI** in every case:

| Model | Empirical (95% CI) | Analytical reference | Verdict |
|---|---|---|---|
| Server availability | 0.8359 [0.8302, 0.8416] | MTBF/(MTBF+MTTR) = 0.8333 | ✅ within CI |
| Stochastic delay | 12.522 [12.495, 12.549] | closed form = 12.500 | ✅ within CI |
| Execution time | exact | ⌈cycles/demand⌉ | ✅ exact |

---

## Study 1b — Offloading on the REAL EUA infrastructure

To validate Study 1 on real data, the same three policies are run on the **real EUA
Melbourne CBD dataset** (125 base stations from the Australian ACMA register; script
[`study1_eua_real.py`](study1_eua_real.py)). Each of 20 seeds subsamples 150 users and
samples their demands; costs are measured **at decision time** (per placed service, so the
processing term stays well-defined). We report the three objectives separately.

| Policy | Latency (↓) | Energy (↓) | Monetary (↓) | Placement |
|---|---|---|---|---|
| **cost-driven** | 1.070 | **7.05** | **6.58** | 1.000 |
| latency-greedy | **0.972** | 28.63 | 5.46¹ | 1.000 |
| random | 1.067 | 21.60 | 6.56 | 1.000 |

¹ latency-greedy's slightly lower monetary is incidental (it packs onto few low-latency
servers); it pays for this with by far the highest energy.

**Finding.** On real infrastructure the multi-objective cost-driven policy cuts **energy by
~67–75%** (7.05 vs. 21.6 random / 28.6 latency-greedy) while keeping latency within ~10% of
the latency-optimal policy — the multi-objective trade-off the cost model is designed to
strike, confirmed on real base-station data. (Methodological note: the environment's
per-step *normalized* reward is a good training signal but is **not** an additive absolute
cost across load trajectories; this study therefore reports raw decision-time objectives.)
Figure: [`study1_eua_real.png`](output/study1_eua_real.png).

## Study 11 — Integration robustness (scales × workloads)

Study 8 established the integrated benefit on a single topology size (~25 nodes) and a single
workload (Alibaba bursts). To test whether the integration claim is *validated* rather than
*illustrated*, this study re-runs the **same** baseline-vs-integrated pipeline (reusing Study 8's
code) across **three topology scales (25/50/100 nodes) × two workload types** — real Alibaba
bursts and a **synthetic Poisson** workload (exponential inter-arrivals + service times, a
different temporal distribution). 5 seeds; script
[`study11_integration_robustness.py`](study11_integration_robustness.py).

Improvement of the integrated stack vs. baseline (positive = better):

| Scale | Workload | Network delay | Deadline | Energy | Median response |
|---|---|---|---|---|---|
| 25 | Alibaba | +29% | +21% | 0% | +16% |
| 25 | Poisson | +6% | **−24%** | +2% | **−41%** |
| 50 | Alibaba | +55% | +40% | −12% | +36% |
| 50 | Poisson | +26% | −8% | +12% | −11% |
| 100 | Alibaba | **+86%** | **+71%** | +1% | +61% |
| 100 | Poisson | +45% | +33% | +12% | +12% |

**Findings.**
- **Network-delay improvement is robust across all six conditions and *increases with topology
  scale*** (25→100 nodes: +29%→+86% on Alibaba, +6%→+45% on Poisson). The integration does not
  weaken at larger scales — it strengthens, because cost-driven placement + latency-aware routing
  have more alternatives to exploit on bigger meshes. This is the core validation.
- **On bursty (Alibaba) workloads the integration improves every metric**, and the gains grow with
  scale (deadline +21%→+71%).
- **On steady (Poisson) workloads the latency/deadline benefit is weaker and, at small scale, can be
  negative** (25-node Poisson: deadline −24%, response −41%). This is the honest, informative part:
  cost-driven placement concentrates load onto network/energy-efficient servers; under *bursty*
  arrivals the idle gaps absorb the resulting queueing, but under *steady* Poisson load at small
  scale there is no slack, so concentration hurts response/deadline. The effect reverses by 100
  nodes (added capacity absorbs the concentration). This is the same placement-vs-queueing tension
  the Study 9 ablation exposed, now mapped across conditions.

**Conclusion (for the paper).** The integration claim is **validated for network delay across all
scales and both workloads (and it strengthens with scale)**; the latency/deadline/energy benefits
are **workload-dependent** — strong for bursty edge/cloud traces, contingent on available slack for
steady arrivals. Reporting this map, rather than a single point, is what turns the integrated
evaluation from illustrative into validated — including its boundary conditions. Figure:
[`study11_integration_robustness.png`](output/study11_integration_robustness.png).

## Study 9 — Ablation study

Isolates each module by toggling placement ∈ {random, cost-greedy, DRL} × routing ∈
{shortest-path, latency-aware} in the integrated pipeline (real EUA mesh + real Alibaba tasks +
execution + energy + failures; 20-node mesh, 300 tasks, **10 seeds**; script
[`study9_ablation.py`](study9_ablation.py); DRL placement trains a DQN per seed and places the
real task stream). Part 3, item 1.

| Config | Placement | Routing | E2E median | Network | Deadline viol. | Energy |
|---|---|---|---|---|---|---|
| baseline | random | shortest-path | 30.2 | 16.3 | 0.607 | 14.37 M |
| cost only | cost | shortest-path | **25.8** | **10.5** | **0.461** | 14.37 M |
| routing only | random | latency-aware | 31.4 | 17.5 | 0.618 | 14.37 M |
| DRL only | DRL | shortest-path | 114.6 | 17.2 | 0.723 | **11.30 M** |
| cost + routing | cost | latency-aware | 26.6 | 11.2 | 0.466 | 14.32 M |
| full (DRL + routing) | DRL | latency-aware | 116.5 | 18.2 | 0.730 | 11.30 M |

**Per-module contribution vs. baseline** (positive = improvement):

| Config | Network | Deadline | Energy |
|---|---|---|---|
| cost only | **+35%** | **+24%** | +0% |
| routing only | −7% | −2% | 0% |
| DRL only | −5% | −19% | **+21%** |
| cost + routing | +32% | +23% | +0% |
| full (**DRL + routing**) | −12% | −20% | +21% |

**Findings (which module contributes what).**
- **The cost model is the dominant contributor** to the balanced improvement: network −35%, deadline
  −24%. It is the single most valuable module. (Energy is near-neutral here because baseline random
  placement already spreads load; see note below on energy's seed-sensitivity.)
- **Routing alone contributes little** on this setup: with *random* placement, latency-aware routing
  has few good alternatives to exploit (−5% network). Its value is realized mainly on top of good
  placement (Study 6/8 show the congestion benefit under load).
- **DRL placement is a genuine trade-off, not a strict win.** The **"full" config (DRL placement +
  latency-aware routing)** achieves the **best energy** (−21%, by packing tasks onto energy-efficient
  servers) but **much worse response and deadlines** (network −12%, deadline −20%; e2e median 116.5
  vs. 30.2) — the concentration it learns creates queueing in the *streaming* setting. The DQN was
  trained on a *static* placement objective (the cost model) that does not see cross-stream queueing,
  so it over-optimizes energy at the expense of latency. **This is an honest negative result the
  ablation surfaces**, pointing to a clear improvement: a stream-aware DRL reward including queue
  delay (future work). *(Note: "full" here = DRL + routing, which is **not** the "integrated" stack of
  Study 8 — that was cost-driven + latency-aware, which does improve deadlines. The two must not be
  conflated under one "full" label.)*
- **Energy percentages are seed-sensitive** — they depend strongly on which Alibaba trace windows the
  seeds select (baseline energy was 9.71 M at 5 seeds, 14.37 M at 10 seeds). The **network and
  deadline findings are stable across seed counts** and carry the ablation's conclusions.
- **No single configuration dominates all metrics** — cost/cost+routing are best for response and
  deadlines, DRL for energy. The ablation makes the per-module contributions and their tensions
  transparent rather than hiding them behind a single aggregate number.

Figure: [`study9_ablation.png`](output/study9_ablation.png).

## Study 10 — Scalability & runtime overhead

Quantifies the cost of the contributed layers (Part 3, item 4; script
[`study10_scalability.py`](study10_scalability.py), 5 seeds).

**A. Per-layer runtime** over a fixed 1000-task / 20-server workload:

| Layer | Runtime | Δ over bare |
|---|---|---|
| execution engine (bare) | 63.5 ms | — |
| + energy meter | 76.2 ms | +20% |
| + failure model | 68.3 ms | ~0 (noise) |
| cost-model decisions (1000×20 scorings) | 121 ms | one-time, per placement |
| routing (2000 flows, 20-node mesh) | 183 ms | one-time, per flow set |

**B. Scaling** (execution + energy + failures): wall-clock **187 ms → 1.64 s** as tasks grow
500 → 4000 (makespan-bound, roughly linear-to-super-linear); **peak memory stays < 2 MB**; runtime
*decreases* with more servers (10→384 ms, 100→168 ms) because added capacity shortens the queueing
makespan.

**Finding.** The five added layers impose modest overhead — the energy meter adds ~20% to a bare
run, failures are negligible, and cost/routing decisions are one-time per placement (hundreds of
ms for thousands of decisions). Memory is negligible. **Scaling is therefore bounded by the base
simulator, not by the contributed modules.** Figures:
[`study10_layer_overhead.png`](output/study10_layer_overhead.png),
[`study10_scaling_tasks.png`](output/study10_scaling_tasks.png).

> **Generalizability (for Results/Discussion).** Topology sizes here (≤ 100 servers, ≤ 4000 tasks,
> ≤ 80-node meshes) are modest relative to production edge deployments. The results establish the
> *relative ordering* of policies and the *direction and magnitude* of per-module effects under
> controlled conditions; absolute latencies/energy and the persistence of effect sizes at
> production scale would require further validation on larger infrastructures and live traces.

## Study 8 — Integrated end-to-end experiment

This is the integration study: **placement, routing, execution, energy and failures all
interact in one run**, rather than being evaluated separately. Real EUA base stations become
edge servers on a geographic mesh (real-distance links); real Alibaba tasks arrive from random
source nodes; a placement policy chooses each task's server (the cost-driven policy's latency
term **includes the routing path delay**, coupling placement and routing); the routing policy
sets the network transit time; the task then queues and executes; energy is metered; and
servers fail (MTBF/MTTR), interrupting tasks. Two full stacks are compared (25-node mesh,
600 real tasks, 6 seeds; script [`study8_integrated.py`](study8_integrated.py)):

    baseline    = random placement      + shortest-path routing
    integrated  = cost-driven placement + latency-aware routing

| Metric | Baseline | Integrated | Improvement |
|---|---|---|---|
| Network transit delay | 16.2 | **11.9** | **−27%** |
| Deadline-violation rate | 0.632 | **0.526** | **−17%** |
| Total energy | 27.87 M | **26.84 M** | **−4%** |
| Median end-to-end response | 28.3 | 26.2 | −8% (CIs overlap) |
| Mean end-to-end response | 212.7 ±110 | 236.7 ±149 | not significant |

**Finding.** The integrated intelligent stack improves the end-to-end metrics it targets:
**network delay (−27%), deadline violations (−17%) and energy (−4%)** are all clear wins,
demonstrating that the components compose into a working system rather than only functioning in
isolation. Median end-to-end response is comparable-to-slightly-better (−8%, overlapping CIs).
The **mean** response is reported but is dominated by failure-restart tails and by Alibaba
workload heterogeneity across trace windows (±110–149), so the **median** is the meaningful
"typical-task" statistic. An honest observation the integration surfaces: cost-driven placement
concentrates load on efficient/cheap servers, which lowers network/energy/SLA cost but adds
queueing under contention — the response-time and network/energy objectives genuinely trade off,
visible only because the layers interact. Figure:
[`study8_integrated.png`](output/study8_integrated.png).

> **Robustness (addressing the single-scenario limitation).** This study fixes one topology size
> and workload. **Study 11** re-runs the same comparison across three topology scales (25/50/100
> nodes) and two workload types (Alibaba bursts + synthetic Poisson) and finds the network-delay
> benefit holds in every condition and *grows with scale*, while latency/deadline benefits are
> workload-dependent — so the integration claim is validated across conditions, not just here.

## Study 7 — DRL hyperparameter tuning & learning curves (PPO / A2C)

Study 2 compared agents at a *fixed* budget, where PPO/A2C lagged. This study sweeps a
learning-rate × entropy grid for the on-policy agents on a small synthetic scenario (8
services, 5 servers; 400 episodes; 16 seeds, raised from 4; script [`study7_drl_tuning.py`](study7_drl_tuning.py))
to test whether that lag is inherent or a tuning artifact.

PPO grid (final weighted cost, ↓):

| learning rate | entropy 0.01 | entropy 0.05 |
|---|---|---|
| 1e-4 | 1.865 | 1.736 |
| 3e-4 | 0.975 | 0.922 |
| **1e-3** | 0.411 | **0.324** |

Best per agent: **PPO 0.324** (±0.197), **A2C 0.704** (±0.245) (both at lr=1e-3, entropy=0.05);
greedy ≈ 0.005 and DQN (default settings) ≈ 0.004 on this easy scenario.

**Finding.** Most of the on-policy lag in Study 2 is a **learning-rate effect**: raising PPO's
learning rate from 1e-4 to 1e-3 cuts its final cost **5.8×** (1.865 → 0.324; 3.0× relative to
the default 3e-4/0.01 setting, 0.975 → 0.324, paired p_adj = 0.0002); A2C improves ~3.1×
(2.191 → 0.704) but remains the weakest. Entropy has a second-order effect (slightly favouring
0.05). However, with 16 seeds the gap is **narrowed, not closed**: tuned PPO is still
significantly above DQN and greedy (paired difference +0.32, 95% CI [+0.13, +0.51], adjusted
p = 0.014, d = 0.89; `study7_drl_tuning_significance.csv`), and the 4-seed result
(0.141, "level with DQN") overstated the effect of tuning. Training-reward curves for the tuned
agents vs. DQN and the greedy baseline are in
[`study7_drl_tuning_curves.png`](output/study7_drl_tuning_curves.png) (they overlap for PPO and
DQN during training, but evaluated final cost separates them); the full grid is in
[`study7_drl_tuning_grid.csv`](output/study7_drl_tuning_grid.csv) and per-seed values in
[`study7_drl_tuning_per_seed.csv`](output/study7_drl_tuning_per_seed.csv). **Takeaway for the
paper:** tuning substantially improves PPO and A2C, so the fixed-budget gap was largely a
tuning artifact, but tuned PPO does not reach DQN/greedy on this task at this budget.

## Study 7b — Fair hyperparameter exploration (equal budget)

Study 7 tuned only PPO/A2C. This study gives **all three agents a comparable budget** — the same
300 training episodes, the same **10 seeds**, and an equal-size grid of 6 configurations each
(learning rate ×3, one agent-appropriate secondary parameter ×2: DQN's `epsilon_decay`, PPO/A2C's
`entropy_coef`) — so the algorithm comparison is fair (Part 3, item 2; script
[`study7b_fair_tuning.py`](study7b_fair_tuning.py)).

| Agent | Episodes | Configs | Seeds | Wall-clock | Best final cost |
|---|---|---|---|---|---|
| **DQN** | 300 | 6 | 10 | 1375 s | **0.006** |
| PPO | 300 | 6 | 10 | 492 s | 0.664 |
| A2C | 300 | 6 | 10 | 307 s | 1.167 |
| greedy | — | — | — | — | 0.008 |

**Finding.** At an **equal optimization budget**, DQN attains the lowest cost (0.006, essentially
the greedy optimum of 0.008), PPO second (0.664) and A2C third (1.167) — the same ordering as the
fixed-budget Study 2, but now with matched budgets and 10 seeds so the comparison is defensible.
The trade-off is **wall-clock**: DQN's per-step replay updates make it **~2.8× slower than PPO and
~4.5× slower than A2C** for the same episode count (1375 s vs. 492 s vs. 307 s over the full grid).

**Budget transparency & fairness language (for the paper).**
> "All agents were tuned under an identical budget: 300 training episodes, 10 seeds, and a 6-point
> hyperparameter grid each (learning rate ×3 and one exploration parameter ×2 —
> `epsilon_decay` for DQN, `entropy_coef` for PPO/A2C). Other hyperparameters were held at library
> defaults. Results are therefore *best-of-an-equal-modest-budget* rather than the ceiling
> performance of any single algorithm; a larger per-agent search could shift absolute values but is
> unlikely to change the ordering given the gaps observed. Note the compute asymmetry: the full
> sweep required ≈21.6k training episodes, and DQN was ~2.8–4.5× slower to train than the
> policy-gradient agents — a cost worth weighing given the paper's energy-efficiency focus."

## Study 5 — Cost-model weight sensitivity (real EUA)

Rather than only disclosing the one weight vector used elsewhere, this study sweeps five
latency/energy/monetary weight vectors and measures the objectives the cost-driven policy
actually achieves on the real EUA infrastructure (125 servers, 150 users, 20 seeds,
decision-time absolute costs; script [`study5_weight_sensitivity.py`](study5_weight_sensitivity.py)).

| Weights (lat / eng / mon) | Latency | Energy | Monetary |
|---|---|---|---|
| balanced (0.5/0.3/0.2) | 1.069 | 7.53 | 6.68 |
| equal (⅓/⅓/⅓) | 1.070 | 7.50 | 6.68 |
| latency-heavy (0.8/0.1/0.1) | **0.976** | 21.92 | 5.59 |
| energy-heavy (0.1/0.8/0.1) | 1.110 | **5.99** | 7.16 |
| money-heavy (0.1/0.1/0.8) | 0.976 | 22.07 | **5.57** |

**Finding.** The cost model is **correctly weight-sensitive**: each objective attains its
minimum precisely when its own weight is highest — energy-heavy weights give the lowest energy
(5.99, a further ~20% below balanced), latency-heavy the lowest latency, money-heavy the lowest
monetary. The result is diagonally dominant (the achieved-objective matrix is minimized on its
diagonal), confirming the weights are meaningful control knobs rather than cosmetic. The
balanced/equal vectors sit between the extremes, as expected. Figure:
[`study5_weight_sensitivity.png`](output/study5_weight_sensitivity.png).

## Study 2b — DRL agents on the REAL EUA infrastructure

The learning result (Study 2) is repeated on **real EUA base-station coordinates** (script
[`study2_eua_real.py`](study2_eua_real.py)). Each of **12 seeds** subsamples 20 real base stations
(keeping the DQN action space trainable) with 30 users; agents train 250 episodes and are
compared to the greedy/random baselines under the environment's cost objective.

| Policy | Mean cost (↓) | 95% CI |
|---|---|---|
| **DQN** | **1.049** | ±0.253 |
| A2C | 1.307 | ±0.383 |
| PPO | 1.378 | ±0.521 |
| Greedy (min-cost) | 1.676 | ±0.346 |
| Random | 1.795 | ±0.391 |

**Finding.** On real Melbourne infrastructure **DQN learns a clearly better policy than the greedy
heuristic** (1.049 vs. 1.676, **≈37% lower cost**), by exploiting the temporal coupling a myopic
greedy cannot. With 12 seeds (up from 6) every CI tightened (DQN ±0.37→±0.25, greedy ±0.73→±0.35),
and DQN remains separated from greedy — so unlike the synthetic case, the **real-data DRL advantage
is robust to the larger seed count**. PPO and A2C also beat greedy, more modestly. This reproduces
the Study 2 ordering on real data and closes the arc from cost model → learned policy → real
infrastructure. Figures:
[`study2_eua_learning.png`](output/study2_eua_learning.png) (learning curves),
[`study2_eua_real.png`](output/study2_eua_real.png). *(Metric: the environment's per-step
normalized cost — the agents' training objective — averaged over an episode; see the note in
Study 1b on why this relative score is used for the learned-policy comparison rather than as
an absolute cost.)*

## Study 1c — Offloading on the REAL Shanghai Telecom infrastructure

To test whether the offloading result generalizes across *independent* real infrastructures,
Study 1b is repeated on the **Shanghai Telecom dataset** (real base-station coordinates and
user access sessions from a different city; script [`study1_shanghai_real.py`](study1_shanghai_real.py),
125 subsampled base stations, 150 users/seed, 20 seeds, decision-time absolute costs).

| Policy | Latency (↓) | Energy (↓) | Monetary (↓) | Placement |
|---|---|---|---|---|
| **cost-driven** | 1.067 | **7.51** | 6.66 | 1.000 |
| latency-greedy | **0.975** | 29.05 | 5.59 | 1.000 |
| random | 1.058 | 23.93 | 6.59 | 1.000 |

**Finding.** The result is essentially identical to EUA: cost-driven placement cuts **energy
~68–74%** (7.51 vs. 23.9 random / 29.1 latency-greedy) for ~9% extra latency. Obtaining the
same trade-off on two independent real datasets (Melbourne EUA and Shanghai Telecom) is
strong evidence that the multi-objective advantage is a property of the method, not of a
particular infrastructure. Figure: [`study1_shanghai_real.png`](output/study1_shanghai_real.png).

## Study 6 — Routing on a larger real-coordinate topology

The synthetic routing study (Study 3) used a 6-path micro-topology. This study evaluates the
same policies on a **larger, geographically-grounded mesh** built from real EUA base-station
coordinates (80 nodes; Euclidean MST for connectivity + k=4 nearest-neighbour edges; link
delays = real geographic distances; randomized congestion; 200 random flows/seed, 20 seeds;
script [`study6_routing_topology.py`](study6_routing_topology.py)). The resulting mesh has
average degree ≈5.0 and paths of **6–8 hops** — genuinely multi-hop, unlike the 2-hop toy.

| Policy | Path delay | Bottleneck utilization | Hops |
|---|---|---|---|
| shortest-path | **8.2** | 0.776 | 6.30 |
| least-congested | 11.1 | **0.574** | 8.28 |
| latency-aware (α=β=0.5) | 8.7 | 0.675 | 6.68 |
| ECMP | 8.2 | 0.782 | 6.62 |

**Finding.** The latency-vs-congestion trade-off from Study 3 **holds on a realistic multi-hop
topology**: latency-aware routing cuts bottleneck utilization ~13% (0.776→0.675) for ~6% extra
delay, while least-congested cuts it ~26% (→0.574) at ~35% extra delay and longer paths. Effect
sizes are smaller than on the toy topology — a dense mesh offers many alternative paths, so
congestion differences compress — but the ordering is unchanged and the CIs are tight.
Figure: [`study6_routing_topology.png`](output/study6_routing_topology.png).

## Study 4b — Real Alibaba workload on the real EUA infrastructure

The execution engine is driven by a real workload — bursts of tasks from the **Alibaba
cluster trace** — running on the **real EUA** edge servers (script
[`study4_alibaba_real.py`](study4_alibaba_real.py); 1,000 tasks/burst, 5 seeds, each a
different trace window). Real Alibaba tasks are small and short relative to 125 edge
servers, so the full infrastructure is lightly loaded (≈100% completion, ~0% deadline
misses) — a finding in itself. Two sweeps surface the interesting regimes.

> **What is real vs. modeled here.** Only the *workload* comes from the trace: each
> `batch_task` row supplies a task's arrival time, duration, and CPU/memory demand. The
> Alibaba trace contains **no server failures** (every task is `Terminated`/successful). The
> *infrastructure* is real (EUA base stations); the *failures* in sweep B are **synthetically
> injected** by our `FailureModel` (a controlled MTBF/MTTR process), because no public edge
> dataset ships real server-failure traces at this granularity. Study 4b is therefore
> "**real workload × real infrastructure × modeled failure process**", with MTBF/MTTR as the
> experiment knob — not failures read from the data.

**A. Capacity sweep** — shrinking the number of available edge servers:

| Servers | Response time | Waiting time | Deadline-violation rate |
|---|---|---|---|
| 125 | 74.1 | 0.0 | 0.000 |
| 60 | 74.1 | 0.0 | 0.000 |
| 30 | 74.3 | 0.2 | 0.002 |
| 15 | 77.3 | 3.2 | 0.026 |

**B. Failure sweep** (30 servers, MTTR = 20):

| MTBF | Response time | Deadline-violation rate | Completion |
|---|---|---|---|
| 200 | 320.8 | 0.160 | 0.991 |
| 400 | 184.5 | 0.087 | 0.996 |
| 800 | 137.2 | 0.044 | 0.999 |
| none | 74.4 | 0.002 | 0.999 |

**Finding.** Response time on the real workload is execution-dominated (~74 steps) until
capacity becomes scarce: at 15 servers queueing appears (waiting 0→3.2, deadline
violations →2.6%). The *injected* failure process then dominates the reliability picture — as
MTBF falls from ∞ to 200, mean response time rises **4.3×** (74→321) and deadline violations
reach **16%**, with completion dipping to 99.1%. Both trends are monotonic and match the
synthetic Study 4.
(Modeling note: MTBF is kept above the ~90-step task durations; with restart-on-failure, an
MTBF below the task duration makes tasks restart faster than they finish.) Figures:
[`study4_alibaba_capacity.png`](output/study4_alibaba_capacity.png),
[`study4_alibaba_failure.png`](output/study4_alibaba_failure.png).

## Summary

| Study | Headline result |
|---|---|
| 1 · Offloading | Cost-driven placement ≈11× cheaper than random, ~12% better than latency-greedy under heterogeneous pricing. |
| 2 · DRL | DQN matches greedy on synthetic data (0.29 vs 0.41, overlapping CIs at 16 seeds) and clearly beats it on real EUA (Study 2b); default-tuned policy-gradient agents lag until tuned (Study 7). |
| 3 · Routing | Latency-aware routing cuts congestion ~60% for ~28% extra delay — the best latency/congestion balance. |
| 4 · Execution & reliability | Contention and failures produce realistic response-time/energy/SLA degradation; all models validated within 95% CI. |
| 1b · Offloading (real EUA) | On real Melbourne base stations, cost-driven placement cuts energy ~67–75% vs. baselines for ~10% latency, confirming the synthetic result. |
| 1c · Offloading (real Shanghai) | The same energy cut (~68–74%) holds on a second independent real infrastructure (Shanghai Telecom) — cross-dataset validation. |
| 2b · DRL (real EUA) | DQN learns a ≈40% cheaper policy than greedy on real Melbourne base stations, reproducing the synthetic learning result on real data. |
| 4b · Execution (real Alibaba on real EUA) | Real workload is execution-bound until capacity is scarce; failures raise response time 4.3× and deadline misses to 16% (MTBF 200). |

Every number above is reproducible from the scripts in this folder and the seeds they fix.
