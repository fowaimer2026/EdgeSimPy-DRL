# Methodology: Stochastic Networking, Experiments & Validation

Beyond features, rigorous simulation research needs **non-deterministic models**,
**statistically grounded experiments**, and **validation**. This contribution adds all
three as opt-in packages.

## Stochastic network delay (`edge_sim_py.network`)

EdgeSimPy's link/wireless delays are fixed scalars, so latency is deterministic.
`StochasticDelayModel` perturbs any base delay with **jitter** (Gaussian or exponential)
and **packet loss** (retransmissions inflate effective delay), and exposes both a sampler
and the closed-form expectation:

```python
from edge_sim_py.network import StochasticDelayModel

model = StochasticDelayModel(jitter_std=2.0, packet_loss=0.2, seed=1)
model.sample(10.0)          # one stochastic realization
model.expected_delay(10.0)  # analytical expectation (for validation)
```

Effective-delay model: a transmission is retried until success (each attempt fails with
probability `packet_loss`, capped at `max_retransmissions`); every attempt costs one
`base_delay`, plus zero-mean jitter.

## Experiment + statistics harness (`edge_sim_py.experiments`)

A *trial function* maps a seed to a metrics dict; the harness runs it across many seeds and
aggregates with confidence intervals — answering "over how many runs, with what variance?"

```python
from edge_sim_py.experiments import run_experiment, parameter_sweep, results_to_csv, plot_sweep

def trial(seed, num_servers=3):
    servers, tasks = random_workload(num_tasks=50, num_servers=num_servers, seed=seed)
    place_least_loaded(tasks, servers)
    engine = ExecutionEngine(servers, tasks); engine.run()
    return collect_workload_metrics(tasks, engine)

exp = run_experiment(trial, seeds=range(30))            # {'raw': [...], 'summary': {metric: CI}}
rows = parameter_sweep(trial, "num_servers", [2, 3, 4, 6], seeds=range(30))
results_to_csv(rows, "sweep.csv")
plot_sweep(rows, x="num_servers", y="mean_response_time", path="sweep.png")
```

`confidence_interval` uses a Student-t critical value (SciPy when available, normal
approximation otherwise). CSV export uses the standard library; pandas and matplotlib are
imported lazily, so the core harness has no hard dependency on them.

## Validation study (`examples/validation_study.py`)

The methodology pieces validate the contributed models against independent references, each
reported with a 95% confidence interval (and locked in by `tests/test_validation.py`):

| Model | Empirical measurement | Analytical reference | Result |
| --- | --- | --- | --- |
| Reliability | Long-run server availability | `MTBF / (MTBF + MTTR)` | within CI |
| Network | Mean sampled delay | `expected_delay()` closed form | within CI |
| Execution | Measured execution time | `ceil(cpu_cycles / (cpu_demand · rate))` | exact |

A model is considered validated when its analytical reference falls within the empirical
confidence interval. The execution check also documents an important property: the engine
uses **integer-step, space-sharing** semantics, so the correct reference is the *ceiling* of
the real-valued compute time.

## Running

```bash
pytest tests/test_network.py tests/test_experiments.py tests/test_validation.py
python examples/validation_study.py
```
