#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PROBE -- does the JOINT criterion (timing AND resource AND quality) have
any spread, where the timing-only criterion provably does not?

`01_diagnostic.py` established that C1 (timing only) saturates at 100% for all
six schemes, structurally. The obvious remedy is the criterion this project
used before it was reverted for EN alignment:

    effective success  <=>  timing  AND  a free concurrency slot on the target
                                     AND  the target link is at least as good
                                          as the serving link

This script measures that, per event, and reports whether the two added
conjuncts -- which DO depend on the target choice, and hence on the policy --
restore any spread. It does not train, and it writes nothing.

WHY THIS IS THE RIGHT PROBE. The timing conjunct is fixed by the trigger rule
alone. The resource conjunct needs the target to be congested (C = 50) and the
quality conjunct needs the target to be worse than the serving link. Both of
those are properties of the TARGET CHOICE, which is the one part of a scheme
that learning can actually influence. If C3 is also flat, then no learning can
produce spread under this criterion either, and the criterion is not the fix.
If C3 has spread, the criterion is the fix and the spread is reproducible.

No Table 7 number appears in this file.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gapmoh.constants import Scenario
from gapmoh.metrics import capacity_violations
from gapmoh.orbit import build_constellation
from gapmoh.passes import build_pass_table
from gapmoh.policies import REPORT_ORDER, SCHEMES
from gapmoh.sim import simulate_terminal

# Execution durations are an INPUT to the criterion, not a Table 7 row we are
# trying to reproduce. Using the same values as 01_diagnostic keeps the two
# scripts comparable; the C3 result below does not depend on them (every
# budget exceeds every dt_exec by 2-3 orders of magnitude).
DT_EXEC_MS = {"rss": 187.3, "dijkstra": 98.5, "cnnlstm": 45.2,
              "madrl_std": 24.7, "grid_only": 12.5, "gapmoh": 3.2}
DT_HO_EXTRA_MS = 2.0


def make_terminals(n, cfg, seed):
    rng = np.random.default_rng(seed)
    lat = np.deg2rad(rng.uniform(cfg.lat_center_deg - cfg.lat_spread_deg,
                                 cfg.lat_center_deg + cfg.lat_spread_deg, n))
    lon = np.deg2rad(rng.uniform(cfg.lon_min_deg, cfg.lon_max_deg, n))
    spd = np.clip(rng.normal(cfg.speed_mean_kn, cfg.speed_sd_kn, n), 2.0, 40.0)
    return lat, lon, spd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminals", type=int, default=12)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Scenario()
    t0 = time.time()
    con = build_constellation(cfg.n_planes, cfg.sats_per_plane,
                              cfg.altitude_km, np.deg2rad(cfg.inclination_deg),
                              cfg.phase_factor)
    lats, lons, spds = make_terminals(args.terminals, cfg, args.seed)

    print(f"C3 PROBE   M={args.terminals} terminals   horizon={args.horizon:.0f} s")
    print(f"C=50 free slots/constellation-cap check; quality = target SNR >= serving SNR")
    print("=" * 92)

    tables = [build_pass_table(float(lats[i]), float(lons[i]), float(spds[i]),
                               con, cfg, args.horizon, dt_samp=2.0)
              for i in range(args.terminals)]
    print(f"pass tables ready in {time.time() - t0:.1f} s")
    print("=" * 92)

    print(f"{'scheme':<18}{'n_ho':>7}{'C1 %':>8}{'C2 %':>8}{'C3 %':>8}"
          f"{'q-fail %':>10}{'mean dSNR':>11}{'dSNR<0 share':>14}")
    print("-" * 92)

    summary = {}
    for key in REPORT_ORDER:
        scheme = SCHEMES[key]
        dt_exec = DT_EXEC_MS[key] * 1e-3
        dt_ho = dt_exec + DT_HO_EXTRA_MS * 1e-3

        traces = [simulate_terminal(tables[i], scheme, cfg, con,
                                    float(lats[i]), float(lons[i]), float(spds[i]),
                                    dt_exec_s=dt_exec, dt_ho_s=dt_ho,
                                    horizon_s=args.horizon)
                  for i in range(args.terminals)]

        _, refused = capacity_violations(traces, cfg.capacity_per_sat)

        ok1 = ok2 = ok3 = 0
        n = 0
        n_qfail = 0
        ds = []
        for ti, tr in enumerate(traces):
            for ei, ev in enumerate(tr.events):
                n += 1
                a = bool(ev.success)
                b = a and ((ti, ei) not in refused)
                q = (not np.isnan(ev.target_snr_db)) and \
                    (ev.target_snr_db >= ev.serving_snr_db)
                c = b and q
                ok1 += a
                ok2 += b
                ok3 += c
                if not np.isnan(ev.target_snr_db):
                    ds.append(ev.target_snr_db - ev.serving_snr_db)
                    if not q:
                        n_qfail += 1

        if n == 0:
            print(f"{scheme.label:<18}{0:>7}{'no handovers':>26}")
            summary[key] = None
            continue

        ds = np.array(ds) if ds else np.array([np.nan])
        c1, c2, c3 = ok1 / n, ok2 / n, ok3 / n
        print(f"{scheme.label:<18}{n:>7}{c1 * 100:>8.1f}{c2 * 100:>8.1f}"
              f"{c3 * 100:>8.1f}{n_qfail / max(n, 1) * 100:>10.1f}"
              f"{float(np.nanmean(ds)):>11.1f}"
              f"{float(np.mean(ds < 0)) * 100:>13.1f}%")
        summary[key] = {"n": n, "c1": c1, "c2": c2, "c3": c3,
                        "mean_dsnr": float(np.nanmean(ds)),
                        "p05_dsnr": float(np.nanpercentile(ds, 5)),
                        "median_dsnr": float(np.nanmedian(ds))}

    print("-" * 92)
    valid = [v for v in summary.values() if v]
    for name, idx in (("C1", "c1"), ("C2", "c2"), ("C3", "c3")):
        vals = [v[idx] for v in valid]
        print(f"{name} spread across schemes: {max(vals) * 100 - min(vals) * 100:.2f} pp"
              f"   (min {min(vals) * 100:.1f}%, max {max(vals) * 100:.1f}%)")
    print("=" * 92)
    print("If C3 spread is still ~0, the criterion is NOT the remedy and the")
    print("disposition must be a framing/description change, not a metric change.")
    print(f"wall clock {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
