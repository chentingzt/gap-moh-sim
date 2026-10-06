# -*- coding: utf-8 -*-
"""Pass-table and FSM sanity tests.

These are deliberately small and fast (one terminal, a few minutes of
simulated time). They lock in the properties that, when broken, produce
plausible-looking but meaningless results -- which is exactly how the first
three drafts of this simulator failed:

  * passes are PAIRED windows, not open edge events (mean duration ~ minutes)
  * the active index is built on the DECISION grid, not the sampling grid
  * a pass open at the horizon end carries its TRUE exit, not the horizon
  * the GBPT trigger fires even though its window is 20x narrower than one
    decision step

Nothing here asserts a handover SUCCESS RATE -- that is an output, and
asserting it would be the circularity the integrity lint exists to prevent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gapmoh.constants import Scenario
from gapmoh.orbit import build_constellation
from gapmoh.passes import build_pass_table
from gapmoh.policies import SCHEMES
from gapmoh.sim import simulate_terminal

LAT, LON, SPD = np.deg2rad(25.0), np.deg2rad(130.0), 15.0
HORIZON = 600.0


@pytest.fixture(scope="module")
def env():
    cfg = Scenario()
    con = build_constellation(cfg.n_planes, cfg.sats_per_plane,
                              cfg.altitude_km, np.deg2rad(cfg.inclination_deg),
                              cfg.phase_factor)
    table = build_pass_table(LAT, LON, SPD, con, cfg, HORIZON, dt_samp=2.0)
    return cfg, con, table


def test_passes_are_paired_windows(env):
    cfg, con, table = env
    dur = table.t_exit - table.t_enter
    assert table.n_pass > 20, "too few passes for a 10-minute episode"
    assert np.all(dur > 0), "some pass has a non-positive duration"
    # a 550 km / 15 deg mask pass runs for minutes, never seconds
    assert dur.mean() > 60.0, f"mean pass duration only {dur.mean():.1f} s"
    assert np.median(dur) > 30.0


def test_visible_count_is_plausible(env):
    cfg, con, table = env
    table.build_active_index(int(HORIZON / cfg.dt_dec_s) + 1, cfg.dt_dec_s)
    n = table.active_at(3000).size
    assert 20 <= n <= 120, f"{n} satellites visible; expected a few dozen"


def test_active_index_uses_decision_grid(env):
    """A 0.1 s index over a 600 s episode must cover ~6000 steps."""
    cfg, con, table = env
    n_step = int(HORIZON / cfg.dt_dec_s) + 1
    flat, off = table.build_active_index(n_step, cfg.dt_dec_s)
    assert off.size == n_step + 1
    assert off[-1] == flat.size > 0
    # if the index had been built on the 2 s sampling grid it would collapse
    assert off[-1] > 10 * n_step


def test_pass_open_at_horizon_has_true_exit(env):
    """No pass may report an exit exactly at the horizon as its real end."""
    cfg, con, table = env
    late = table.t_enter > HORIZON - 400.0
    if late.any():
        assert table.t_exit[late].max() > HORIZON, (
            "a pass opening late in the episode reports an exit at the "
            "horizon -- the scan tail is missing")


@pytest.mark.parametrize("key", ["gapmoh", "grid_only", "rss", "dijkstra",
                                 "cnnlstm", "madrl_std"])
def test_every_scheme_produces_handovers(env, key):
    """No scheme may silently produce an empty timeline."""
    cfg, con, table = env
    scheme = SCHEMES[key]
    dt_exec = 5e-3
    tr = simulate_terminal(table, scheme, cfg, con, LAT, LON, SPD,
                           dt_exec_s=dt_exec, dt_ho_s=dt_exec + 2e-3,
                           horizon_s=HORIZON)
    assert tr.n_handover > 0, f"{scheme.label} produced no handovers"
    assert tr.n_forced_fail <= 2, f"{scheme.label} had {tr.n_forced_fail} losses"


def test_gbpt_budget_equals_margin(env):
    """GBPT's budget is dt_ho by construction -- that is Definition 3."""
    cfg, con, table = env
    scheme = SCHEMES["gapmoh"]
    dt_exec, dt_ho = 3.2e-3, 5.2e-3
    tr = simulate_terminal(table, scheme, cfg, con, LAT, LON, SPD,
                           dt_exec_s=dt_exec, dt_ho_s=dt_ho, horizon_s=HORIZON)
    b = tr.budgets()
    assert b.size > 0
    assert np.allclose(b, dt_ho, atol=1e-6), \
        f"GBPT budgets {b * 1e3} ms, expected dt_ho = {dt_ho * 1e3} ms"
    # and therefore every GBPT handover succeeds, since dt_exec < dt_ho
    assert bool(tr.successes().all())


def test_no_ping_pong(env):
    """No scheme may alternate between two satellites at the step rate."""
    cfg, con, table = env
    for key in ("dijkstra", "cnnlstm", "madrl_std"):
        scheme = SCHEMES[key]
        dt_exec = 5e-3
        tr = simulate_terminal(table, scheme, cfg, con, LAT, LON, SPD,
                               dt_exec_s=dt_exec, dt_ho_s=dt_exec + 2e-3,
                               horizon_s=HORIZON)
        max_plausible = HORIZON / 20.0    # one handover per 20 s is the floor
        assert tr.n_handover < max_plausible, (
            f"{scheme.label} made {tr.n_handover} handovers in {HORIZON:.0f} s "
            "-- alternating targets, i.e. a trigger/admissibility artefact")
