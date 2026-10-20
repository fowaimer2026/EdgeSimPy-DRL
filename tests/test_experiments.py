"""Unit tests for the experiment + statistics harness."""

# Python libraries
import os

import pytest

# EdgeSimPy
from edge_sim_py.experiments import (
    confidence_interval,
    aggregate,
    run_trials,
    run_experiment,
    parameter_sweep,
    results_to_csv,
)


def test_confidence_interval_constant_sample():
    summary = confidence_interval([5.0, 5.0, 5.0, 5.0])
    assert summary["mean"] == pytest.approx(5.0)
    assert summary["std"] == pytest.approx(0.0)
    assert summary["half_width"] == pytest.approx(0.0)


def test_confidence_interval_known_values():
    summary = confidence_interval([2.0, 4.0, 6.0, 8.0])
    assert summary["mean"] == pytest.approx(5.0)
    # Sample std of [2,4,6,8] is sqrt(20/3) ~ 2.582.
    assert summary["std"] == pytest.approx(2.5820, rel=1e-3)
    assert summary["ci_low"] < 5.0 < summary["ci_high"]


def test_confidence_interval_single_value():
    summary = confidence_interval([3.0])
    assert summary["mean"] == 3.0
    assert summary["half_width"] == 0.0
    assert summary["n"] == 1


def test_aggregate_over_results():
    results = [{"a": 1.0, "b": 10.0}, {"a": 3.0, "b": 20.0}]
    summary = aggregate(results)
    assert summary["a"]["mean"] == pytest.approx(2.0)
    assert summary["b"]["mean"] == pytest.approx(15.0)


def test_run_trials_runs_once_per_seed():
    calls = []

    def trial(seed):
        calls.append(seed)
        return {"value": seed * 2}

    results = run_trials(trial, seeds=[1, 2, 3])
    assert calls == [1, 2, 3]
    assert [r["value"] for r in results] == [2, 4, 6]


def test_run_experiment_returns_raw_and_summary():
    def trial(seed, offset=0):
        return {"score": seed + offset}

    experiment = run_experiment(trial, seeds=[0, 1, 2, 3], offset=10)
    assert len(experiment["raw"]) == 4
    assert experiment["summary"]["score"]["mean"] == pytest.approx(11.5)


def test_parameter_sweep_rows():
    def trial(seed, k=1):
        return {"y": seed * k}

    rows = parameter_sweep(trial, "k", [1, 2, 3], seeds=[0, 1, 2, 3])
    assert [row["k"] for row in rows] == [1, 2, 3]
    assert all("y_mean" in row and "y_ci" in row for row in rows)
    # Mean of seeds 0..3 is 1.5, scaled by k.
    assert rows[1]["y_mean"] == pytest.approx(3.0)


def test_results_to_csv(tmp_path):
    rows = [{"k": 1, "y_mean": 2.0}, {"k": 2, "y_mean": 4.0}]
    path = results_to_csv(rows, os.path.join(tmp_path, "out.csv"))
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as handle:
        contents = handle.read()
    assert "k,y_mean" in contents and "2,4.0" in contents
