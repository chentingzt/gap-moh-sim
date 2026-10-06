#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which definition of "handover success" can actually separate the schemes?

WHY. The manuscript's Table 7 row 1 reports a spread (79.6 -> 96.8) under

    success <=> dt_exec < t_exit - t_init.

That criterion CANNOT produce a spread for the two GBPT schemes, and the reason
is algebraic, not empirical: GBPT fires at t_init = t_exit - dt_ho with
dt_ho = dt_exec + 2 ms (CN:140), so the budget is dt_exec + 2 ms by
construction and `dt_exec < budget` holds identically. GBPT's success is
manufactured by its own trigger definition, at a 2 ms margin, for any data.

So the row has to be re-defined, and this script finds out WHICH re-definition
is worth adopting by measuring candidate criteria on the same traces. A
candidate is only usable if it (a) is unambiguous, (b) is not self-fulfilling,
and (c) actually separates the schemes.

CANDIDATES
    C0  timing only                        (the current row; expected degenerate)
    C1  timing AND target above the mask at completion
    C2  timing AND target still above the mask T_hold after completion
    C3  timing AND target rain-feasible at completion (ITU-R P.618-13)
    C4  C2 AND no re-handover within T_hold  (punishes handing to a dying link)

C0-C4 all use the same traces, so the comparison isolates the DEFINITION.

The `--require-rising` switch matters and is reported both ways, because the
manuscript's target rules as literally written do not exclude a satellite that
is already setting (`first_above_gate` picks by list order; `longest_remaining`
picks by remaining time, which is finite for a setting satellite too). The
rising filter is this package's [CHOICE], not the manuscript's rule, so the
result must not depend on it.

Nothing here is tuned to Table 7; the published numbers appear only in the
final comparison block as a report.

Usage:
    python scripts/07_probe_success.py --terminals 20 --horizon 1800
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gapmoh.channel import rain_loss_db, snr_from_elev_db, rain_rate_for_loss_db
from gapmoh.constants import Scenario                             # noqa: E402
from gapmoh.orbit import build_constellation, sat_elev_series     # noqa: E402
from gapmoh.passes import build_pass_table                        # noqa: E402
from gapmoh.policies import REPORT_ORDER, SCHEMES                 # noqa: E402
from gapmoh.sim import simulate_terminal                          # noqa: E402

DT_EXEC_MS = {"rss": 187.3, "dijkstra": 98.5, "cnnlstm": 45.2,
              "madrl_std": 24.7, "grid_only": 12.5, "gapmoh": 3.2}
DT_HO_EXTRA_MS = 2.0

T_HOLD_S = 30.0          # the dwell a delivered link must hold, for C2/C4


def make_terminals(n, cfg, seed):
    """Radians -- see the note in `tests/test_theatre_geography.py`."""
    rng = np.random.default_rng(seed)
    lat = np.deg2rad(rng.uniform(cfg.lat_center_deg - cfg.lat_spread_deg,
                                 cfg.lat_center_deg + cfg.lat_spread_deg, n))
    lon = np.deg2rad(rng.uniform(cfg.lon_min_deg, cfg.lon_max_deg, n))
    spd = np.clip(rng.normal(cfg.speed_mean_kn, cfg.speed_sd_kn, n), 2.0, 40.0)
    return lat, lon, spd


def _elev_of_sat(con, sat_id, t, lat, lon0, spd):
    from gapmoh.orbit import terminal_series
    term = terminal_series(np.array([t]), lat, lon0, spd)[0]
    return float(sat_elev_series(np.array([t]), np.array([term]), con,
                                 np.array([sat_id]))[0, 0])


def evaluate(traces, con, lats, lons, spds, cfg, rain_rate, t_hold=T_HOLD_S):
    """C0-C4 for one scheme's traces. Returns dict name -> percentage."""
    mask = np.deg2rad(cfg.min_elev_deg)
    n_ev = 0
    hits = {k: 0 for k in ("c0", "c1", "c2", "c3", "c4")}

    for tr, lat, lon0, spd in zip(traces, lats, lons, spds):
        ev = tr.events
        for i, e in enumerate(ev):
            n_ev += 1
            t_done = e.t_init + e.dt_exec_s

            # C0 -- the manuscript's criterion, exactly as stated.
            c0 = bool(e.dt_exec_s < e.budget_s)
            if not c0:
                continue

            # The target must still exist as a pass worth serving.
            if e.target_sat < 0:
                continue
            e_done = _elev_of_sat(con, e.target_sat, t_done, lat, lon0, spd)
            e_hold = _elev_of_sat(con, e.target_sat, t_done + t_hold,
                                  lat, lon0, spd)

            c1 = e_done >= mask                      # usable target at completion
            c2 = c1 and e_hold >= mask               # still usable T_hold later
            net = float(snr_from_elev_db(np.array([e_done]), cfg)[0]) \
                - float(rain_loss_db(np.array([e_done]), cfg, rain_rate)[0])
            c3 = c1 and net >= cfg.snr_min_db        # closes under rain
            # C4 -- no immediate re-handover: the delivered link must be the
            # one the terminal actually keeps for T_hold.
            nxt = ev[i + 1] if i + 1 < len(ev) else None
            c4 = c2 and (nxt is None or (nxt.t_init - t_done) >= t_hold)

            hits["c0"] += 1
            hits["c1"] += int(c1)
            hits["c2"] += int(c2)
            hits["c3"] += int(c3)
            hits["c4"] += int(c4)

    if n_ev == 0:
        return {k: float("nan") for k in hits}, 0
    return {k: 100.0 * v / n_ev for k, v in hits.items()}, n_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminals", type=int, default=20)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out_dir or os.path.join(root, "results", "success_probe")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "probe_log.txt"), "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfg = Scenario()
    rain_rate, _ = rain_rate_for_loss_db(np.deg2rad(cfg.min_elev_deg), cfg,
                                         cfg.rain_margin_db)

    log("=" * 96)
    log("SUCCESS-CRITERION PROBE")
    log(f"  M={args.terminals}  horizon={args.horizon:.0f} s  seed={args.seed}"
        f"  T_hold={T_HOLD_S:.0f} s  rain={rain_rate:.2f} mm/h  "
        f"SNR_min={cfg.snr_min_db:.0f} dB  mask={cfg.min_elev_deg:.0f} deg")
    log("=" * 96)

    con = build_constellation(cfg.n_planes, cfg.sats_per_plane, cfg.altitude_km,
                              np.deg2rad(cfg.inclination_deg), cfg.phase_factor)
    lats, lons, spds = make_terminals(args.terminals, cfg, args.seed)
    t0 = time.time()
    tables = [build_pass_table(float(lats[i]), float(lons[i]), float(spds[i]),
                               con, cfg, args.horizon, dt_samp=2.0)
              for i in range(args.terminals)]
    log(f"pass tables ready in {time.time() - t0:.1f} s "
        f"({np.mean([t.n_pass for t in tables]):.0f} passes/terminal)")

    payload = {"spec": {"M": args.terminals, "horizon_s": args.horizon,
                        "seed": args.seed, "t_hold_s": T_HOLD_S,
                        "rain_rate_mm_h": float(rain_rate)},
               "order": REPORT_ORDER, "runs": {}}

    for rising in (True, False):
        tag = "require_rising=True" if rising else "require_rising=False (rules as written)"
        log("")
        log("=" * 96)
        log(f"  {tag}")
        log("=" * 96)
        res = {}
        for key in REPORT_ORDER:
            scheme = SCHEMES[key]
            dt_exec = DT_EXEC_MS[key] * 1e-3
            traces = [simulate_terminal(
                tables[i], scheme, cfg, con, float(lats[i]), float(lons[i]),
                float(spds[i]), dt_exec_s=dt_exec,
                dt_ho_s=dt_exec + DT_HO_EXTRA_MS * 1e-3,
                horizon_s=args.horizon, trace_serving=True,
                require_rising=rising) for i in range(args.terminals)]
            res[key], n_ev = evaluate(traces, con, lats, lons, spds, cfg,
                                      rain_rate)
            res[key]["n_events"] = n_ev
        payload["runs"][tag] = res

        log("")
        log(f"{'scheme':<18}{'n':>7}" + "".join(
            f"{c:>9}" for c in ("C0", "C1", "C2", "C3", "C4")))
        for key in REPORT_ORDER:
            r = res[key]
            log(f"{SCHEMES[key].label:<18}{r['n_events']:>7}" + "".join(
                f"{r[c]:>9.1f}" for c in ("c0", "c1", "c2", "c3", "c4")))
        log("")
        log(f"{'spread (pp)':<18}{'':>7}" + "".join(
            f"{max(res[k][c] for k in REPORT_ORDER) - min(res[k][c] for k in REPORT_ORDER):>9.1f}"
            for c in ("c0", "c1", "c2", "c3", "c4")))
        log("")
        log("  C0 timing only | C1 +target above mask at completion "
            "| C2 +still above mask T_hold later")
        log("  C3 +closes under rain | C4 C2 +no re-handover within T_hold")

    with open(os.path.join(out, "success_probe.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log("")
    log(f"wrote {os.path.join(out, 'success_probe.json')}")
    log(f"wall clock {time.time() - t0:.1f} s")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
