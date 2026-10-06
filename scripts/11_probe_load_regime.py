#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""At what terminal count does the per-satellite concurrency cap C=50 bind?

WHY THIS SCRIPT EXISTS. The authors' provenance record §6 established, and the
`10_probe_headroom.py` run confirms to machine precision, that no metric this
package computes gives the DRL refinement layer room -- and that the reason is
structural, not statistical. `select_target`'s `highest_elev` branch returns the
ARGMAX of the candidate set's elevation, and a clear-sky channel makes SNR
monotone in elevation, so `best_quality` and `highest_elev` select the *same
satellite every step* (verified bit-identical). Every metric that is monotone in
serving elevation -- M1 target elevation and SNR, M2 serving elevation and SNR --
therefore has its optimum sitting ON the framework layer. No policy can beat it
there, and no parameter change to the orbital or channel model can alter a
property that follows from monotonicity.

The one quantity the framework rule does NOT optimize is LOAD. Each terminal
names its target independently, with no knowledge of the other terminals'
choices. So load is the only candidate left for carrying the framework-vs-
refinement decomposition -- which is also what the manuscript's own contribution
(2) says the DRL layer does ("optimize target selection AND load balancing",
CN:31).

But the load metric is degenerate today, and the degeneracy is a PARAMETER
problem rather than a physics problem. The manuscript runs M=100, which its own
text labels a low-load scenario (CN:90, CN:240: "M=100 ... 低负载"). Measured at
M=40: peak concurrent 3 against a cap of 50, busiest satellite holding 2.3% of
terminal-time across 4,994 active satellites. A cap that never binds is not a
metric, and admission rejection -- the failure mode CN:90 explicitly declines to
model -- never occurs at that load.

So this is a calibration question: at what M does C=50 bind? This script answers
it rather than assuming it.

METHOD. Terminal sets are NESTED: one draw of `m_max` positions, prefixes taken
for each sweep point, so load grows monotonically with M by construction instead
of jittering with the RNG. Pass tables are built ONCE at `m_max` and sliced, so
the sweep costs one build, not one per point.

Two deployments are compared because they cost the manuscript very different
amounts:

    uniform   the current 20 +/- 15 N, 110-150 E box, terminals spread evenly
    lane      the same box, terminals concentrated along maritime corridors

Marine traffic concentrates in lanes, straits and port approaches; a uniform
spread over a 30 x 40 degree box is the low-density limit of the theatre, not a
description of it. If the lane deployment reaches the cap at a much smaller M,
the cheaper scenario change is also the more faithful one.

READ THE HORIZON. Peak concurrent load is a MAXIMUM over the horizon, so a short
horizon UNDERESTIMATES it. The probe runs short for cost and states the horizon
in every output line. The number that goes into the manuscript must be
re-measured at the manuscript's own horizon, on the final M and deployment.

Usage:
    python scripts/11_probe_load_regime.py
    python scripts/11_probe_load_regime.py --horizon 3600 --uniform-max 700
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gapmoh.constants import Scenario                            # noqa: E402
from gapmoh.metrics_v2 import concurrency                        # noqa: E402
from gapmoh.orbit import build_constellation                     # noqa: E402
from gapmoh.passes import build_pass_table                       # noqa: E402
from gapmoh.policies import SCHEMES                              # noqa: E402
from gapmoh.sim import simulate_terminal                         # noqa: E402

DT_EXEC_MS = {"rss": 187.3, "dijkstra": 98.5, "cnnlstm": 45.2,
              "madrl_std": 24.7, "grid_only": 12.5, "gapmoh": 3.2}
DT_HO_EXTRA_MS = 2.0

# CN:90 / Table 5: 50 concurrent users per satellite.
CAPACITY_PER_SAT = 50

# The framework rule's own arm -- this measures the load the deterministic layer
# produces, which is the baseline the refinement layer would have to improve on.
ARM_KEY = "gapmoh"


def draw_terminals(m_max, cfg, seed, mode, n_lane, sigma_deg):
    """One draw of `m_max` positions; the caller takes prefixes.

    Radians throughout -- see `tests/test_theatre_geography.py`.
    """
    rng = np.random.default_rng(seed)
    lo_lat = cfg.lat_center_deg - cfg.lat_spread_deg
    hi_lat = cfg.lat_center_deg + cfg.lat_spread_deg

    if mode == "uniform":
        lat_deg = rng.uniform(lo_lat, hi_lat, m_max)
        lon_deg = rng.uniform(cfg.lon_min_deg, cfg.lon_max_deg, m_max)
    elif mode == "lane":
        # Corridor centres spread across the box, terminals clustered on them.
        c_lat = rng.uniform(lo_lat, hi_lat, n_lane)
        c_lon = rng.uniform(cfg.lon_min_deg, cfg.lon_max_deg, n_lane)
        pick = rng.integers(0, n_lane, m_max)
        lat_deg = c_lat[pick] + rng.normal(0.0, sigma_deg, m_max)
        lon_deg = c_lon[pick] + rng.normal(0.0, sigma_deg, m_max)
    else:
        raise ValueError(f"unknown deployment {mode!r}")

    spd = np.clip(rng.normal(cfg.speed_mean_kn, cfg.speed_sd_kn, m_max),
                  2.0, 40.0)
    return np.deg2rad(lat_deg), np.deg2rad(lon_deg), spd


def build_tables(lats, lons, spds, con, cfg, horizon, log):
    t0 = time.time()
    tables = [build_pass_table(float(lats[i]), float(lons[i]), float(spds[i]),
                               con, cfg, horizon, dt_samp=2.0)
              for i in range(len(lats))]
    log(f"    built {len(tables)} pass tables in {time.time() - t0:.0f} s")
    return tables


def sweep(mode, m_list, lats, lons, spds, con, cfg, horizon, base, dt, log):
    """Slice the pre-built tables; the M-sweep is nested by construction."""
    rows = []
    for m in m_list:
        t0 = time.time()
        traces = [simulate_terminal(
            tables_cache[mode][i], base, cfg, con, float(lats[i]),
            float(lons[i]), float(spds[i]), dt_exec_s=dt,
            dt_ho_s=dt + DT_HO_EXTRA_MS * 1e-3, horizon_s=horizon,
            trace_sat=True) for i in range(m)]
        c = concurrency(traces, CAPACITY_PER_SAT)
        row = {
            "deployment": mode,
            "terminals": m,
            "peak_concurrent": c["peak_concurrent"],
            "peak_over_capacity": c["peak_over_capacity"],
            "overflow_terminal_seconds": c["overflow_terminal_seconds"],
            "top1_share": c["top1_share"],
            "n_active_sat": c["n_active_sat"],
            "mean_load_active_sat": c["mean_load_active_sat"],
            "cap_binds": bool((c["peak_over_capacity"] or 0) > 0),
        }
        rows.append(row)
        log(f"    M={m:<5} peak={row['peak_concurrent']:<4} "
            f"over_cap={row['peak_over_capacity']:<4} "
            f"top1={100.0 * (row['top1_share'] or 0):6.2f}%  "
            f"active_sat={row['n_active_sat']:<5} "
            f"cap_binds={'YES' if row['cap_binds'] else 'no':<4} "
            f"({time.time() - t0:.0f} s)")
    return rows


tables_cache = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=900.0,
                    help="short by default; peak load is a max, so this "
                         "UNDERESTIMATES the manuscript-horizon value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uniform-max", type=int, default=700)
    ap.add_argument("--uniform-m", default="100,200,400,700")
    ap.add_argument("--lane-max", type=int, default=300)
    ap.add_argument("--lane-m", default="50,100,200,300")
    ap.add_argument("--n-lane", type=int, default=3)
    ap.add_argument("--sigma-deg", type=float, default=1.0,
                    help="lane corridor half-width in degrees (~111 km/deg)")
    ap.add_argument("--skip-lane", action="store_true")
    ap.add_argument("--skip-uniform", action="store_true",
                    help="skip the uniform deployment (its 700 pass tables "
                         "cost ~83 min and are irrelevant to a sigma sweep)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results",
                                                      "load_regime"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "load_regime_log.txt"), "w",
                encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfg = Scenario()
    con = build_constellation(cfg.n_planes, cfg.sats_per_plane, cfg.altitude_km,
                              np.deg2rad(cfg.inclination_deg), cfg.phase_factor)
    base = SCHEMES[ARM_KEY]
    dt = DT_EXEC_MS[ARM_KEY] * 1e-3

    log("=" * 96)
    log("LOAD-REGIME PROBE -- at what M does the per-satellite cap C bind?")
    log(f"  horizon={args.horizon:.0f} s  seed={args.seed}  cap C={CAPACITY_PER_SAT}")
    log(f"  box: {cfg.lat_center_deg:.0f}+/-{cfg.lat_spread_deg:.0f} N, "
        f"{cfg.lon_min_deg:.0f}-{cfg.lon_max_deg:.0f} E   arm={ARM_KEY} "
        f"(target={base.target})")
    log(f"  NOTE: peak concurrent is a MAXIMUM, so h={args.horizon:.0f} s "
        f"underestimates the h=3600 s value.")
    log("=" * 96)

    plans = []
    if not args.skip_uniform:
        plans.append(("uniform", args.uniform_max,
                      [int(s) for s in args.uniform_m.split(",")
                       if s.strip()]))
    if not args.skip_lane:
        plans.append(("lane", args.lane_max,
                      [int(s) for s in args.lane_m.split(",") if s.strip()]))

    all_rows = []
    for mode, m_max, m_list in plans:
        log("")
        log(f"--- deployment: {mode} ---")
        lats, lons, spds = draw_terminals(m_max, cfg, args.seed, mode,
                                          args.n_lane, args.sigma_deg)
        # Geography guard: a deployment that throws terminals outside the
        # theatre, or sees no satellite, is an error, not a result.
        if not (np.all(np.isfinite(lats)) and np.all(np.isfinite(lons))):
            raise SystemExit(f"ABORT: non-finite terminal position in {mode}")
        tables_cache[mode] = build_tables(lats, lons, spds, con, cfg,
                                          args.horizon, log)
        n_pass = np.array([t.n_pass for t in tables_cache[mode]])
        log(f"    passes/terminal min {n_pass.min()} mean {n_pass.mean():.0f} "
            f"max {n_pass.max()}")
        if (n_pass == 0).any():
            raise SystemExit(
                f"ABORT: {int((n_pass == 0).sum())}/{m_max} terminals in "
                f"{mode} see no satellite above the mask -- geography error")
        all_rows += sweep(mode, m_list, lats, lons, spds, con, cfg,
                          args.horizon, base, dt, log)

    # ---- verdict ----------------------------------------------------------
    log("")
    log("=" * 96)
    log("VERDICT")
    log("=" * 96)
    for mode in {r["deployment"] for r in all_rows}:
        rows = [r for r in all_rows if r["deployment"] == mode]
        first = next((r for r in rows if r["cap_binds"]), None)
        peak = [r["peak_concurrent"] for r in rows]
        log(f"  {mode:<8} peak grows {min(peak)} -> {max(peak)} over "
            f"M {min(r['terminals'] for r in rows)} -> "
            f"{max(r['terminals'] for r in rows)}")
        if first is None:
            # Extrapolate on the last two points rather than guessing.
            a, b = rows[-2], rows[-1]
            dm = b["terminals"] - a["terminals"]
            dp = b["peak_concurrent"] - a["peak_concurrent"]
            if dp > 0:
                need = a["terminals"] + dm * (CAPACITY_PER_SAT
                                              - a["peak_concurrent"]) / dp
                log(f"           cap does NOT bind within the sweep; linear "
                    f"extrapolation on the last segment puts it at "
                    f"M ~ {need:.0f}")
            else:
                log("           cap does not bind and the last segment is flat; "
                    "no M in this regime will bind it")
        else:
            log(f"           cap binds from M={first['terminals']} "
                f"(peak {first['peak_concurrent']} > C={CAPACITY_PER_SAT}), "
                f"overflow {first['overflow_terminal_seconds']:.1f} "
                "terminal-seconds")
    log("")
    log("  A scenario is usable for the framework-vs-refinement decomposition")
    log("  only where the cap BINDS: below that point every arm reports zero")
    log("  overflow and the metric cannot separate them.")

    with open(os.path.join(args.out_dir, "load_regime.json"), "w",
              encoding="utf-8") as f:
        json.dump({"spec": {"horizon_s": args.horizon, "seed": args.seed,
                            "capacity_per_sat": CAPACITY_PER_SAT,
                            "n_lane": args.n_lane,
                            "sigma_deg": args.sigma_deg,
                            "arm": ARM_KEY},
                   "rows": all_rows}, f, indent=2, ensure_ascii=False,
                  default=float)
    log("")
    log(f"wrote {os.path.join(args.out_dir, 'load_regime.json')}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
