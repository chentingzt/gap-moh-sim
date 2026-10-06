#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STAGE 0 GATE -- does the paper's own success criterion have any spread?

The manuscript's criterion (CN:90, CN:240) is purely temporal:

    success  <=>  dt_exec < t_exit - t_init

and it explicitly excludes sync failure, authentication failure, target-link
rain fade and admission rejection. The budget `t_exit - t_init` is therefore a
pure geometric quantity fixed by the SCHEME'S TRIGGER RULE alone. Training
cannot move it. If every scheme's budget distribution sits far above its
dt_exec, all six schemes succeed ~100% and the criterion cannot produce the
spread Table 7 reports -- no amount of MADRL will change that.

This script measures the budget distribution per scheme and reports:

  (a) C1/C2/C3 success rates under the paper's stated execution durations
  (b) the budget distribution per scheme
  (c) the dt_exec each scheme would NEED to hit its published success rate
  (d) the decision rule

Run this before any training. If C1 and C2 are flat, stop and report.

This file is the ONLY place under gapmoh_sim/ allowed to contain Table 7
numbers (row 2: execution durations). They are an INPUT to the criterion --
the row the criterion is supposed to predict (row 1) is never read here.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gapmoh.constants import Scenario
from gapmoh.metrics import aggregate, capacity_violations, score_trace
from gapmoh.orbit import build_constellation
from gapmoh.passes import build_pass_table
from gapmoh.policies import REPORT_ORDER, SCHEMES
from gapmoh.sim import simulate_terminal

# ---------------------------------------------------------------------------
# INPUTS taken from the manuscript (allowed here, forbidden under gapmoh/).
# Row 2 of Table 7, execution-phase duration in ms. These are an input to the
# criterion, exactly as CN:199 says ("各对比方案按表 7 各自执行时长加 2 ms").
# ---------------------------------------------------------------------------
TABLE7_DT_EXEC_MS = {
    "rss": 187.3,
    "dijkstra": 98.5,
    "cnnlstm": 45.2,
    "madrl_std": 24.7,
    "grid_only": 12.5,
    "gapmoh": 3.2,
}
# Row 1, published success rate in percent -- READ ONLY, for comparison, never
# fed back into the simulation.
TABLE7_SUCCESS_PCT = {
    "rss": 79.6, "dijkstra": 87.0, "cnnlstm": 90.1,
    "madrl_std": 92.4, "grid_only": 94.1, "gapmoh": 96.8,
}
# Row 3, unnecessary-handover share, for comparison only.
TABLE7_UNNECESSARY_PCT = {
    "rss": 23.4, "dijkstra": 15.1, "cnnlstm": 11.6,
    "madrl_std": 8.3, "grid_only": 6.7, "gapmoh": 4.1,
}

DT_HO_EXTRA_MS = 2.0   # CN:199 -- dt_ho = dt_exec + 2 ms


def make_terminals(n, cfg, seed=0):
    rng = np.random.default_rng(seed)
    lat = np.deg2rad(rng.uniform(cfg.lat_center_deg - cfg.lat_spread_deg,
                                 cfg.lat_center_deg + cfg.lat_spread_deg, n))
    lon = np.deg2rad(rng.uniform(cfg.lon_min_deg, cfg.lon_max_deg, n))
    spd = np.clip(rng.normal(cfg.speed_mean_kn, cfg.speed_sd_kn, n), 2.0, 40.0)
    return lat, lon, spd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminals", type=int, default=8)
    ap.add_argument("--horizon", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = Scenario()
    t0 = time.time()

    con = build_constellation(cfg.n_planes, cfg.sats_per_plane,
                              cfg.altitude_km, np.deg2rad(cfg.inclination_deg),
                              cfg.phase_factor)
    lats, lons, spds = make_terminals(args.terminals, cfg, args.seed)
    print(f"STAGE 0 GATE   M={args.terminals} terminals   horizon={args.horizon:.0f} s")
    print(f"constellation {cfg.total_sats} sats, mask {cfg.min_elev_deg} deg, "
          f"dt_dec {cfg.dt_dec_s * 1e3:.0f} ms")
    print("=" * 78)

    tables, build_s = [], []
    for i in range(args.terminals):
        tb = time.time()
        tables.append(build_pass_table(float(lats[i]), float(lons[i]),
                                       float(spds[i]), con, cfg, args.horizon,
                                       dt_samp=2.0))
        build_s.append(time.time() - tb)
    print(f"pass tables built in {sum(build_s):.1f} s "
          f"({np.mean([t.n_pass for t in tables]):.0f} passes/terminal avg)")
    print("=" * 78)

    report = {"config": vars(args), "schemes": {}}
    rows = []
    for key in REPORT_ORDER:
        scheme = SCHEMES[key]
        dt_exec = TABLE7_DT_EXEC_MS[key] * 1e-3
        dt_ho = dt_exec + DT_HO_EXTRA_MS * 1e-3

        tb = time.time()
        traces = []
        for i in range(args.terminals):
            traces.append(simulate_terminal(
                tables[i], scheme, cfg, con, float(lats[i]), float(lons[i]),
                float(spds[i]), dt_exec_s=dt_exec, dt_ho_s=dt_ho,
                horizon_s=args.horizon))
        scores = [score_trace(t, cfg) for t in traces]
        agg = aggregate(scores)
        elapsed = time.time() - tb

        # C2 -- the one cross-terminal coupling (C = 50 free slots on target)
        n_refused, refused = capacity_violations(traces, cfg.capacity_per_sat)
        if agg:
            total_ho = max(agg["handovers"], 1)
            c2 = agg["c1"] * (1.0 - n_refused / total_ho)
        else:
            c2 = float("nan")

        all_budgets = np.concatenate([t.budgets() for t in traces]) \
            if any(t.n_handover for t in traces) else np.array([0.0])
        all_snr = np.array([e.serving_snr_db for t in traces
                            for e in t.events], dtype=float)

        report["schemes"][key] = {
            "label": scheme.label,
            "dt_exec_ms": dt_exec * 1e3,
            "dt_ho_ms": dt_ho * 1e3,
            "agg": agg,
            "c2_adjusted": c2,
            "refused_for_capacity": n_refused,
            "budget_ms": {
                "mean": float(np.mean(all_budgets) * 1e3),
                "p01": float(np.percentile(all_budgets, 1) * 1e3),
                "p05": float(np.percentile(all_budgets, 5) * 1e3),
                "min": float(np.min(all_budgets) * 1e3),
                "max": float(np.max(all_budgets) * 1e3),
            },
            "serving_snr": ({} if all_snr.size == 0 else {
                "n": int(all_snr.size),
                "min": float(all_snr.min()),
                "p05": float(np.percentile(all_snr, 5)),
                "mean": float(all_snr.mean()),
                "max": float(all_snr.max()),
            }),
            "seconds": elapsed,
        }
        rows.append((key, scheme, agg, c2, all_budgets))

    # ---------------- table (a): success under the paper's own dt_exec -------
    print("(a) success under the manuscript's stated execution durations")
    print(f"{'scheme':<18}{'dt_exec':>9}{'C1 %':>9}{'C2 %':>9}"
          f"{'unnec %':>9}{'refused':>9}{'paper C1':>10}")
    for key, scheme, agg, c2, _ in rows:
        if not agg:
            print(f"{scheme.label:<18}{'-':>9}{'no handovers':>18}")
            continue
        print(f"{scheme.label:<18}{TABLE7_DT_EXEC_MS[key]:>8.1f}m"
              f"{agg['c1'] * 100:>9.1f}{c2 * 100:>9.1f}"
              f"{agg['unnecessary'] * 100:>9.1f}"
              f"{agg['forced_fail']:>9d}"
              f"{TABLE7_SUCCESS_PCT[key]:>9.1f}%")
    print()

    # ---------------- table (b): the budget distribution --------------------
    print("(b) budget = t_exit - t_init, the geometric quantity the criterion")
    print("    actually compares against dt_exec  (ms)")
    print(f"{'scheme':<18}{'mean':>10}{'p01':>10}{'p05':>10}"
          f"{'min':>10}{'max':>10}")
    for key, scheme, agg, c2, b in rows:
        print(f"{scheme.label:<18}{np.mean(b) * 1e3:>10.1f}"
              f"{np.percentile(b, 1) * 1e3:>10.1f}"
              f"{np.percentile(b, 5) * 1e3:>10.1f}"
              f"{np.min(b) * 1e3:>10.1f}{np.max(b) * 1e3:>10.1f}")
    print()

    # ---------------- table (c): what dt_exec WOULD be needed ---------------
    print("(c) dt_exec each scheme would need to reproduce its published rate")
    print(f"{'scheme':<18}{'needed dt_exec (ms)':>22}{'paper dt_exec (ms)':>22}")
    for key, scheme, agg, c2, b in rows:
        need = np.percentile(b, 100.0 - TABLE7_SUCCESS_PCT[key]) * 1e3
        print(f"{scheme.label:<18}{need:>22.3f}{TABLE7_DT_EXEC_MS[key]:>22.1f}")
    print()

    # ---------------- table (d): the "unnecessary" metric's threshold ------
    thr = cfg.snr_min_db + cfg.unnecessary_margin_db
    print(f"(d) serving-link SNR at t_init vs the {thr:.0f} dB 'unnecessary'")
    print("    threshold (SNR_min 10 dB + 3 dB). CN:240 classifies a handover")
    print("    as unnecessary when this SNR is ABOVE the threshold.")
    print(f"{'scheme':<18}{'min':>9}{'p05':>9}{'mean':>9}{'max':>9}"
          f"{'unnec %':>10}{'paper':>9}")
    for key, scheme, agg, c2, b in rows:
        v = report["schemes"][key].get("serving_snr")
        if not v or v["n"] == 0:
            print(f"{scheme.label:<18}{'no handovers':>36}")
            continue
        print(f"{scheme.label:<18}{v['min']:>9.1f}{v['p05']:>9.1f}"
              f"{v['mean']:>9.1f}{v['max']:>9.1f}"
              f"{(agg['unnecessary'] * 100) if agg else float('nan'):>10.1f}"
              f"{TABLE7_UNNECESSARY_PCT[key]:>9.1f}")
    print()

    # ---------------- decision rule ----------------------------------------
    c1s = [r[2]["c1"] for r in rows if r[2]]
    c2s = [r[3] for r in rows if r[2] and np.isfinite(r[3])]
    spread1 = (max(c1s) - min(c1s)) * 100 if c1s else float("nan")
    spread2 = (max(c2s) - min(c2s)) * 100 if c2s else float("nan")

    print("=" * 78)
    print(f"C1 spread across the six schemes : {spread1:.2f} pp")
    print(f"C2 spread across the six schemes : {spread2:.2f} pp")
    if not np.isfinite(spread1):
        verdict = "NO DATA -> check the schemes actually produced handovers"
    elif spread1 < 1.0:
        verdict = "FLAT -> STOP AND REPORT (risk R1)"
    else:
        verdict = "SPREAD EXISTS -> may proceed to training"
    print(f"VERDICT: {verdict}")
    print("=" * 78)
    print(f"total wall clock: {time.time() - t0:.1f} s")

    report["summary"] = {"c1_spread_pp": spread1, "c2_spread_pp": spread2,
                         "verdict": verdict}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str),
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
