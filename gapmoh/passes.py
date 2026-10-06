# -*- coding: utf-8 -*-
"""Per-terminal pass table: the grid's enter/exit-time pre-index.

The manuscript's grid pre-index records, for each fine cell and each candidate
satellite, the times at which the satellite enters and leaves coverage
(CN:104, EN:196-268). That table is what makes the GBPT trigger O(1) at
runtime: `t_exit` is a lookup, not a propagation.

This module builds the same quantity directly per terminal, by sampling the
whole constellation over the horizon and recording contiguous above-mask runs.
A cell-based build would be the deployment form; per-terminal is the research
form and is what the episodes actually need, since each episode is one voyage.

Memory note: one terminal at a time. The expanded active-pass index is
(n_steps, n_active_max) int32, a few MB at a 20-minute voyage, so terminals are
streamed rather than held together.

[DEVIATION] The manuscript describes a global three-tier geohash database
(EN:196-268); this builds the equivalent only over the scenario theatre and
only for the terminals actually simulated.
"""

from dataclasses import dataclass

import numpy as np

from .orbit import elevation, sat_ecef, terminal_ecef

__all__ = ["PassTable", "build_pass_table"]


@dataclass
class PassTable:
    """Contiguous above-mask windows for one terminal."""
    sat: np.ndarray        # (P,) satellite index
    t_enter: np.ndarray    # (P,) seconds
    t_exit: np.ndarray     # (P,) seconds
    elev_peak: np.ndarray  # (P,) radians, elevation at the sampled peak
    horizon_s: float
    dt_s: float
    _flat: np.ndarray = None
    _offsets: np.ndarray = None

    @property
    def n_pass(self) -> int:
        return int(self.sat.size)

    def build_active_index(self, n_steps, dt_step):
        """Ragged active-pass index over the decision grid.

        Returns `(flat, offsets)`:
            active passes at step k  ==  flat[offsets[k]:offsets[k + 1]]

        `dt_step` is the DECISION interval (0.1 s), which is finer than the
        pass-table sampling interval `dt_s` (2 s); the index must be built on
        the decision grid, not the sampling one.

        A ragged layout rather than a padded (n_steps, n_active) matrix: the
        number of live passes is ~53 and n_steps is ~12,000, so the padded form
        is mostly sentinel padding, and the ragged form is also what the
        stepping loop wants (a slice, no filtering).

        Built with an incremental sweep: passes are opened at their first step
        and closed after their last, so the cost is O(total active slots).
        """
        if self.n_pass == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(n_steps + 1, np.int64)

        start = np.clip(np.ceil(self.t_enter / dt_step).astype(np.int64),
                        0, n_steps)
        stop = np.clip(np.floor(self.t_exit / dt_step).astype(np.int64) + 1,
                       0, n_steps)

        opening, closing = {}, {}
        for p in range(self.n_pass):
            a, b = int(start[p]), int(stop[p])
            if b <= a:
                continue
            opening.setdefault(a, []).append(p)
            closing.setdefault(b, []).append(p)

        flat = []
        offsets = np.zeros(n_steps + 1, dtype=np.int64)
        live = []
        for k in range(n_steps):
            # a pass covering [a, b) is NOT live at step b, so close first
            for p in closing.get(k, ()):
                try:
                    live.remove(p)
                except ValueError:
                    pass
            live.extend(opening.get(k, ()))
            offsets[k + 1] = offsets[k] + len(live)
            flat.extend(live)

        flat = np.asarray(flat, dtype=np.int32)
        self._flat, self._offsets = flat, offsets
        return flat, offsets

    def active_at(self, k):
        """Pass indices live at grid step k (requires `build_active_index`)."""
        return self._flat[self._offsets[k]:self._offsets[k + 1]]

    def exit_time_of(self, pass_idx, t):
        return self.t_exit[pass_idx]


def build_pass_table(lat, lon0, speed_kn, con, cfg, horizon_s, dt_samp=2.0,
                     refine_iters=20, tail_s=900.0):
    """Sample a terminal's visibility to the whole constellation.

    `lat`, `lon0` are radians; `speed_kn` knots. Sweeps the horizon, pairs each
    rising edge with its falling edge, and bisects both to ~dt_samp/2^iters.
    A satellite already up at t=0 is treated as rising at t=0.

    `tail_s` extends the scan past the horizon so that a pass still open at
    the end gets its TRUE exit time instead of being truncated there. Without
    the tail every pass overlapping the horizon end reports `t_exit = horizon`,
    which fabricates a shrinking remaining-time ramp for the last few hundred
    seconds and latches any "remaining < threshold" trigger into a 10 Hz
    alternation. 900 s comfortably exceeds the longest excursion above a 15
    deg mask at 550 km.

    Sample spacing is a trade-off: the bisection recovers edge times to
    sub-millisecond precision, but a pass whose whole excursion above the mask
    is shorter than `dt_samp` can be missed entirely. At 2 s and a 15 deg mask
    that is far below the shortest geometric pass.
    """
    min_elev = np.deg2rad(cfg.min_elev_deg)
    scan_s = horizon_s + tail_s
    n_samp = int(scan_s / dt_samp) + 1
    ts = np.arange(n_samp) * dt_samp

    open_since = {}      # sat idx -> (t_below, t_above) bracket of its rise
    peak = {}            # sat idx -> best elevation seen so far this pass
    records = []         # (sat, t_enter, t_exit, peak_elev)
    prev = None

    for k in range(n_samp):
        t = float(ts[k])
        e = elevation(sat_ecef(t, con), terminal_ecef(t, lat, lon0, speed_kn))
        vis = e > min_elev

        if prev is None:
            prev = vis
            for idx in np.flatnonzero(vis):
                open_since[int(idx)] = (0.0, 0.0)   # already up at t = 0
                peak[int(idx)] = float(e[idx])
            continue

        for idx in np.flatnonzero(vis):
            i = int(idx)
            if i in open_since:
                if e[idx] > peak[i]:
                    peak[i] = float(e[idx])
            else:
                open_since[i] = (t - dt_samp, t)    # rise bracket
                peak[i] = float(e[idx])

        for idx in np.flatnonzero(~vis & prev):
            i = int(idx)
            if i not in open_since:
                continue
            lo, hi = open_since.pop(i)
            t_enter = _bisect_crossing(i, lo, hi, lat, lon0, speed_kn, con,
                                       min_elev, refine_iters)
            t_exit = _bisect_crossing(i, t - dt_samp, t, lat, lon0, speed_kn,
                                      con, min_elev, refine_iters)
            records.append((i, t_enter, t_exit, peak.pop(i)))
        prev = vis

    # anything still up when the scan window ends
    for i, (lo, hi) in open_since.items():
        t_enter = _bisect_crossing(i, lo, hi, lat, lon0, speed_kn, con,
                                   min_elev, refine_iters)
        records.append((i, t_enter, scan_s, peak.get(i, min_elev)))

    if not records:
        return PassTable(np.zeros(0, dtype=np.int64), np.zeros(0),
                         np.zeros(0), np.zeros(0), horizon_s, dt_samp)

    sat_arr = np.array([r[0] for r in records], dtype=np.int64)
    te_arr = np.array([r[1] for r in records], dtype=float)
    tx_arr = np.array([r[2] for r in records], dtype=float)
    ep_arr = np.array([r[3] for r in records], dtype=float)

    # keep only passes that overlap the episode proper
    keep = (tx_arr > 0.0) & (te_arr < horizon_s) & (tx_arr > te_arr)
    order = np.argsort(te_arr[keep], kind="stable")
    return PassTable(sat_arr[keep][order], te_arr[keep][order],
                     tx_arr[keep][order], ep_arr[keep][order],
                     horizon_s, dt_samp)


def _elev_of(idx, t, lat, lon0, speed_kn, con):
    return float(elevation(sat_ecef(t, con)[idx:idx + 1],
                           terminal_ecef(t, lat, lon0, speed_kn))[0])


def _bisect_crossing(idx, t_a, t_b, lat, lon0, speed_kn, con, min_elev,
                     iters):
    """Time in [t_a, t_b] at which elevation crosses `min_elev`.

    The caller guarantees the two ends straddle the mask. If they do not (the
    rise bracket of a pass already up at t=0 is degenerate) the midpoint is
    returned unchanged.
    """
    fa = _elev_of(idx, t_a, lat, lon0, speed_kn, con) - min_elev
    fb = _elev_of(idx, t_b, lat, lon0, speed_kn, con) - min_elev
    if (fa > 0) == (fb > 0):
        return 0.5 * (t_a + t_b)
    for _ in range(iters):
        m = 0.5 * (t_a + t_b)
        fm = _elev_of(idx, m, lat, lon0, speed_kn, con) - min_elev
        if (fm > 0) == (fa > 0):
            t_a = m
            fa = fm
        else:
            t_b = m
    return 0.5 * (t_a + t_b)
