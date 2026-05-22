"""Regression tests for combine_lerobot's task-dedup contract.

Locks the invariant that repeated ``--task "<count>:<text>"`` specs with
identical text collapse to a single canonical ``task_index`` in the
output — never produce N distinct task_index rows under one prompt
string. This is the bug that left ``real_synth_dagger1`` with
``total_tasks: 6`` despite only two unique prompts.
"""

from __future__ import annotations

import math

from falsify.cli.combine_lerobot import (
    _canonical_task_map,
    _parse_tasks,
    _renumber_stats,
    _scalar_stats_arange,
    _scalar_stats_constant,
    _task_index_for,
)


def test_repeated_specs_collapse_to_one_index():
    specs = _parse_tasks([
        "50:left",
        "50:right",
        "50:left",
        "rest:right",
    ])
    canon = _canonical_task_map(specs)
    assert canon == {"left": 0, "right": 1}


def test_per_episode_lookup_uses_canonical_index():
    specs = _parse_tasks([
        "50:left",
        "50:right",
        "50:left",
        "50:right",
        "50:left",
        "rest:right",
    ])
    canon = _canonical_task_map(specs)
    by_index = {0: 0, 1: 0}  # canonical task_index -> count
    for ep in range(300):
        by_index[_task_index_for(ep, specs, canon, 300)] += 1
    assert by_index == {0: 150, 1: 150}


def test_first_seen_wins_canonical_index():
    specs = _parse_tasks([
        "10:right",   # right is seen first → gets index 0
        "10:left",
        "rest:right",
    ])
    canon = _canonical_task_map(specs)
    assert canon == {"right": 0, "left": 1}


def test_distinct_texts_get_distinct_indices():
    specs = _parse_tasks([
        "10:a",
        "10:b",
        "rest:c",
    ])
    canon = _canonical_task_map(specs)
    assert canon == {"a": 0, "b": 1, "c": 2}


# -- precomputed-stats reuse ------------------------------------------------


def test_scalar_stats_constant():
    s = _scalar_stats_constant(7, n=100)
    assert s == {"min": [7.0], "max": [7.0], "mean": [7.0], "std": [0.0], "count": [100]}


def test_scalar_stats_arange_matches_numeric_truth():
    # Closed-form must equal a numerical computation over the same range.
    start, n = 1000, 241
    s = _scalar_stats_arange(start, n)
    assert s["min"] == [1000.0]
    assert s["max"] == [float(start + n - 1)]
    # Mean of {start..start+n-1} = start + (n-1)/2
    assert s["mean"] == [start + (n - 1) / 2.0]
    # std of {0..n-1} is sqrt((n^2 - 1)/12); shift doesn't affect std.
    expected_std = math.sqrt((n * n - 1) / 12.0)
    assert math.isclose(s["std"][0], expected_std, rel_tol=1e-12)
    assert s["count"] == [n]


def test_renumber_stats_reuses_immutable_fields():
    prior = {
        "image":         {"min": [[[0.1]], [[0.2]], [[0.3]]], "max": [], "mean": [], "std": [], "count": [100]},
        "state":         {"min": [-1.0, -2.0], "max": [1.0, 2.0], "mean": [0.0, 0.0], "std": [0.5, 1.0], "count": [241]},
        "timestamp":     {"min": [0.0], "max": [24.0], "mean": [12.0], "std": [6.928], "count": [241]},
        "frame_index":   {"min": [0.0], "max": [240.0], "mean": [120.0], "std": [69.4], "count": [241]},
        # the three we expect to overwrite:
        "episode_index": {"min": [9.0], "max": [9.0], "mean": [9.0], "std": [0.0], "count": [241]},
        "index":         {"min": [42.0], "max": [282.0], "mean": [162.0], "std": [69.4], "count": [241]},
        "task_index":    {"min": [5.0], "max": [5.0], "mean": [5.0], "std": [0.0], "count": [241]},
    }
    out = _renumber_stats(
        prior, n_rows=241, global_episode_index=0, global_index_start=0, task_index=1,
    )
    # Immutable fields unchanged.
    assert out["image"] is prior["image"]
    assert out["state"] is prior["state"]
    assert out["timestamp"] is prior["timestamp"]
    assert out["frame_index"] is prior["frame_index"]
    # Renumbered fields overwritten.
    assert out["episode_index"] == {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [241]}
    assert out["task_index"]    == {"min": [1.0], "max": [1.0], "mean": [1.0], "std": [0.0], "count": [241]}
    assert out["index"]["min"] == [0.0]
    assert out["index"]["max"] == [240.0]
    assert out["index"]["mean"] == [120.0]
    assert math.isclose(out["index"]["std"][0], math.sqrt((241 * 241 - 1) / 12.0), rel_tol=1e-12)
