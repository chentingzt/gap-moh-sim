# -*- coding: utf-8 -*-
"""`metrics_v2.concurrency` must count OCCUPANCY, not arrivals.

WHY THIS TEST EXISTS. The metric this replaces -- `metrics.capacity_violations`
-- never decrements its per-satellite counter, so a satellite serving one
terminal after another reads as a long queue and trips the cap without ever
being concurrently overloaded. Swapping arrivals for occupancy is the whole
point of the metric, and the difference only shows up when terminals are
SERIALISED on one satellite: exactly the case a naive test would not construct.
`test_serialised_use_of_one_satellite_is_not_overload` is that case.

Synthetic traces are used so the test costs nothing and runs without geometry.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gapmoh.metrics_v2 import concurrency                            # noqa: E402
from gapmoh.sim import TerminalTrace                                 # noqa: E402


def _trace(sat_ids, horizon_s=100.0):
    tr = TerminalTrace(horizon_s=horizon_s)
    tr.serving_sat_trace = np.asarray(sat_ids, dtype=np.int64)
    return tr


def test_serialised_use_of_one_satellite_is_not_overload():
    """One terminal at a time on sat 0, ten in a row, cap 2.

    `capacity_violations` reports a violation here; occupancy does not, and
    occupancy is the correct answer for a concurrency cap.
    """
    traces = []
    for i in range(10):
        # terminal i is on sat 0 only during its own slice of the horizon
        s = np.full(100, -1, dtype=np.int64)
        s[i * 10:(i + 1) * 10] = 0
        traces.append(_trace(s))
    c = concurrency(traces, capacity_per_sat=2)
    assert c["n_traces_used"] == 10
    assert c["peak_concurrent"] == 1, (
        f"serialised single occupancy read as {c['peak_concurrent']} concurrent")
    assert c["peak_over_capacity"] == 0
    assert c["overflow_terminal_seconds"] == 0.0


def test_simultaneous_terminals_are_counted_and_overloaded():
    """Five terminals on sat 0 at once, cap 2 -> peak 5, 3 over."""
    traces = [_trace(np.full(50, 0, dtype=np.int64)) for _ in range(5)]
    c = concurrency(traces, capacity_per_sat=2)
    assert c["peak_concurrent"] == 5
    assert c["peak_over_capacity"] == 3
    # 3 terminals over the cap, for the whole 50-step horizon.
    dt = 100.0 / 49.0
    assert abs(c["overflow_terminal_seconds"] - 3 * 50 * dt) < 1e-9
    assert c["top1_share"] == 1.0
    assert c["n_active_sat"] == 1


def test_unserved_steps_are_not_attributed_to_a_satellite():
    """-1 is 'no serving satellite', not 'a satellite indexed -1'."""
    traces = [_trace(np.array([-1, -1, 7, 7], dtype=np.int64))]
    c = concurrency(traces, capacity_per_sat=50)
    assert c["n_active_sat"] == 1
    assert c["peak_concurrent"] == 1
    assert c["top1_share"] == 1.0


def test_load_spread_lowers_top1_share():
    """Splitting terminals across satellites must show up as a lower share."""
    together = [_trace(np.full(40, 0, dtype=np.int64)) for _ in range(4)]
    split = [_trace(np.full(40, i, dtype=np.int64)) for i in range(4)]
    c_tog = concurrency(together, capacity_per_sat=50)
    c_spl = concurrency(split, capacity_per_sat=50)
    assert c_tog["top1_share"] == 1.0
    assert abs(c_spl["top1_share"] - 0.25) < 1e-12
    assert c_spl["n_active_sat"] == 4
    assert c_spl["peak_concurrent"] == 1


def test_traces_without_the_satellite_trace_are_skipped_not_zeroed():
    """A trace built without `trace_sat=True` must not read as zero load.

    Reading it as zero would make a misconfigured run look perfectly balanced,
    which is the failure mode this whole metric exists to avoid.
    """
    good = _trace(np.full(30, 3, dtype=np.int64))
    bad = TerminalTrace(horizon_s=100.0)          # serving_sat_trace is None
    assert bad.serving_sat_trace is None

    c = concurrency([good, bad], capacity_per_sat=50)
    assert c["n_traces_used"] == 1, "a trace without the satellite trace was used"
    assert c["peak_concurrent"] == 1

    none = concurrency([bad, bad], capacity_per_sat=50)
    assert none["n_traces_used"] == 0
    assert none["peak_concurrent"] is None
    assert none["overflow_terminal_seconds"] is None
