# Real Datasets

The contributed studies run on synthetic scenarios by default. This module adds
**loaders/adapters** that convert well-known public edge-computing datasets into EdgeSimPy
objects, so the same studies can be run on **real data** for external validity.

Because these datasets are hosted externally (some behind registration), **you download the
raw files** and point the loaders at them. Small **format-accurate samples** ship under
[`edge_sim_py/datasets/samples/`](../edge_sim_py/datasets/samples/), so every loader — and
the example and tests — runs out of the box before you download anything.

## Available loaders

| Loader | Dataset | Download | Produces |
|---|---|---|---|
| `load_eua` | EUA (Melbourne base stations + users) | [github.com/PhuLai/eua-dataset](https://github.com/PhuLai/eua-dataset) | `(servers, services)` |
| `load_shanghai` | Shanghai Telecom (base stations + access) | [sguangwang.com/TelecomDataset.html](http://sguangwang.com/TelecomDataset.html) | `(servers, services)` |
| `load_alibaba_tasks` | Alibaba cluster trace (task workload) | [github.com/alibaba/clusterdata](https://github.com/alibaba/clusterdata) | `[Task]` |
| `load_edgesimpy_dataset` | Native EdgeSimPy JSON | [edgesimpy-tutorials](https://github.com/EdgeSimPy/edgesimpy-tutorials) | initialized `Simulator` |

`EUA`/`Shanghai` give real **infrastructure** (edge servers at real base-station
coordinates, assigned realistic hardware profiles from `datasets/profiles.py`) for the
offloading, routing and DRL studies. `Alibaba` gives a real **workload** (task arrivals, CPU
demand, durations, deadlines) for the execution engine.

## Usage

```python
from edge_sim_py.datasets import load_eua, load_alibaba_tasks

# Real EUA infrastructure -> EdgeServer + Service objects
servers, services = load_eua("eua_site.csv", "eua_users.csv", seed=1)

# Real Alibaba workload -> Task objects, run on those servers
from edge_sim_py.workload import ExecutionEngine, place_least_loaded, collect_workload_metrics
tasks = load_alibaba_tasks("batch_task.csv", limit=2000)
place_least_loaded(tasks, servers)
engine = ExecutionEngine(servers, tasks); engine.run()
print(collect_workload_metrics(tasks, engine))
```

The loaders are **column-configurable** (`lat_column`, `lon_column`, Alibaba column indices)
so they adapt to dataset variants. `load_shanghai` reads both `.csv` and the real Shanghai
Telecom `.xlsx` files (columns `month, date, start time, end time, latitude, longitude,
user id`); Excel support needs `pandas` + `openpyxl` (`pip install openpyxl`), imported
lazily. The Alibaba trace is 14M+ unsorted rows, so for a dense workload use
`sort_by_arrival=True` with `limit`/`window_offset` (and `scan_limit` to bound the read). The Alibaba loader defaults to the headerless
`batch_task.csv` schema of cluster-trace-v2018 (`plan_cpu` in hundredths of a core) and
normalizes arrivals so the earliest task starts at step 0; each task's `cpu_cycles` is set so
that, run in isolation, its execution time reproduces the traced duration.

## Example & tests

```bash
# Runs on the bundled samples by default; pass real files to scale up.
python examples/real_dataset_example.py
python examples/real_dataset_example.py \
    --eua-servers eua_site.csv --eua-users eua_users.csv --alibaba batch_task.csv --alibaba-limit 2000

pytest tests/test_datasets.py
```

To reproduce the [results studies](../results/RESULTS.md) on real data, swap the
`random_scenario` / `random_workload` calls in the study scripts for the loaders above —
the rest of each pipeline is unchanged.
