#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry regression: does the ported orbit code still reproduce the figures
that the already-released geometry simulation publishes?

The released code (`handover_frequency_sim/`) states, and its committed
`results.json` confirms, for 5040 satellites / 550 km / 53 deg / 15 deg mask /
20 N / 120 E / 15 kn / 12 h:

    Reactive (max-elev)  : 18.9167 events/hour  (window mean 3.1673 min, n=227)
    Necessary (GBPT)     :  9.1667 events/hour  (window mean 6.5127 min, n=110)

Those two numbers are the anchor for everything downstream: if the physics in
`gapmoh/orbit.py` ever drifts from the released model, this script fails.

This script deliberately re-implements the released `simulate` loop rather than
importing it, so that a change to either side is caught.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gapmoh.orbit import (build_constellation, elevation, sat_ecef,
                          terminal_ecef, exit_times_for)

# --- configuration of the released run ------------------------------------
PLANES, SATS_PER_PLANE = 72, 70
ALTITUDE_KM, INCLINATION_DEG = 550.0, 53.0
DT, DT_LOOK, LOOK_CAP = 2.0, 10.0, 600.0
DURATION_H = 12.0
LAT_DEG, LON_DEG, SPEED_KN = 20.0, 120.0, 15.0
MIN_ELEV_DEG = 15.0

# --- published / committed reference values -------------------------------
REF_REACTIVE_RATE = 18.9167
REF_NECESSARY_RATE = 9.1667
REF_REACTIVE_WINDOW_MIN = 3.1673
REF_NECESSARY_WINDOW_MIN = 6.5127
TOL_RATE = 0.02          # events/hour
TOL_WINDOW = 0.02        # minutes


def _select_max_elev(t, term, con, elev, min_elev):
    vis = np.where(elev > min_elev)[0]
    return int(vis[np.argmax(elev[vis])]) if vis.size else -1


def _select_max_remaining(t, term, cone, elev, min_elev):
    vis = np.where(elev > min_elev)[0]
    if not vis.size:
        return -1
    rem = exit_times_for(t, term, cone, vis, min_elev, DT_LOOK, LOOK_CAP)
    return int(vis[np.argmax(rem)])


def simulate(select_fn):
    """Replicates the released loop. Returns (n_handovers, windows_in_minutes)."""
    lat = np.deg2rad(LAT_DEG)
    lon0 = np.deg2rad(LON_DEG)
    min_elev = np.deg2rad(MIN_ELEV_DEG)
    con = build_constellation(PLANES, SATS_PER_PLANE, ALTITUDE_KM,
                              np.deg2rad(INCLINATION_DEG), 1)

    nstep = int(DURATION_H * 3600.0 / DT)
    serving = -1
    n_ho = 0
    windows = []
    last_switch = 0.0
    for k in range(nstep):
        t = k * DT
        term = terminal_ecef(t, lat, lon0, SPEED_KN)
        elev = elevation(sat_ecef(t, con), term)
        if serving < 0:
            serving = select_fn(t, term, con, elev, min_elev)
            last_switch = t
            continue
        if elev[serving] < min_elev:
            windows.append(t - last_switch)
            serving = select_fn(t, term, con, elev, min_elev)
            last_switch = t
            n_ho += 1
    return n_ho, np.array(windows) / 60.0


def main():
    print("Geometry regression against the released handover_frequency_sim")
    print("-" * 66)

    n_react, win_react = simulate(_select_max_elev)
    n_nec, win_nec = simulate(_select_max_remaining)

    react_rate = n_react / DURATION_H
    nec_rate = n_nec / DURATION_H

    rows = [
        ("Reactive (max-elev)", react_rate, REF_REACTIVE_RATE,
         float(win_react.mean()), REF_REACTIVE_WINDOW_MIN, n_react, 227),
        ("Necessary (GBPT)", nec_rate, REF_NECESSARY_RATE,
         float(win_nec.mean()), REF_NECESSARY_WINDOW_MIN, n_nec, 110),
    ]

    ok = True
    print(f"{'policy':<20} {'rate/h':>9} {'ref':>9} {'window':>8} {'ref':>8} "
          f"{'n':>5} {'ref':>5}")
    for name, rate, ref_rate, win, ref_win, n, ref_n in rows:
        d_rate = abs(rate - ref_rate)
        d_win = abs(win - ref_win)
        good = d_rate <= TOL_RATE and d_win <= TOL_WINDOW
        ok &= good
        print(f"{name:<20} {rate:>9.4f} {ref_rate:>9.4f} {win:>8.4f} "
              f"{ref_win:>8.4f} {n:>5d} {ref_n:>5d}   "
              f"{'OK' if good else 'MISMATCH'}")

    print("-" * 66)
    if ok:
        print("PASS: ported orbit code reproduces the released geometry model.")
    else:
        print("FAIL: physics has drifted from the released model. "
              "Do not proceed until this is green.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
