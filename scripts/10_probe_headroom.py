#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does ANY metric this package computes give the DRL layer room to improve?

WHY. The manuscript's contribution (3) decomposes GAP-MOH into a deterministic
framework layer (grid + GBPT) and a DRL refinement layer on top, and reports
+14.5 pp / +2.7 pp. Every number in that decomposition sits on Table 7 row 1,
whose criterion is identically true (see the authors' provenance record §1). To redo
the decomposition on a metric that does not degenerate, the metric must satisfy
one precondition:

    the framework layer's own rule must NOT already be at that metric's optimum

Otherwise no policy -- trained or not -- can beat it, and an ablation reporting
zero is telling you about the metric, not about the layer. This script tests
that precondition for every metric the package can compute, by evaluating each
scripted target rule on the SAME geometry and looking for a spread.

    M1  target quality     -- highest_elev IS the argmax of the candidate set
    M2  service availability -- does serving quality improve over the horizon?
    M3  unnecessary        -- is 0% reachable, i.e. already attained?
    M4  handover frequency -- is lower unambiguously better?
    M5  concurrency        -- does the C=50 cap ever bind?

The answer is reported as a table of spreads plus an explicit verdict per
metric. "No room" is a legitimate and useful result: it says the decomposition
cannot be carried by that row, which is what the manuscript has to know before
it claims a number.

Usage:
    python scripts/10_probe_headroom.py --terminals 40 --horizon 1200
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gapmoh.constants import Scenario                                # noqa: E402
from gapmoh.metrics_v2 import (concurrency, service_floor_report,    # noqa: E402
                               summarise)
from gapmoh.orbit import build_constellation                         # noqa: E402
from gapmoh.passes import build_pass_table                           # noqa: E402
from gapmoh.policies import SCHEMES                                  # noqa: E402
from gapmoh.sim import simulate_terminal                             # noqa: E402

DT_EXEC_MS = {"rss": 187.3, "dijkstra": 98.5, "cnnlstm": 45.2,
              "madrl_std": 24.7, "grid_only": 12.5, "gapmoh": 3.2}
DT_HO_EXTRA_MS = 2.0
CAPACITY_PER_SAT = 50
ARM_KEY = "gapmoh"

# Every target rule `policies.select_target` implements. These are the reachable
# set: a trained policy can only deliver what some rule over this candidate set
# can deliver, so their best value bounds any policy's.
TARGET_RULES = ("highest_elev", "longest_remaining", "best_quality",
                "first_above_gate")


def make_terminals(n, cfg, seed):
    rng = np.random.default_rng(seed)
    lat = np.deg2rad(rng.uniform(cfg.lat_center_deg - cfg.lat_spread_deg,
                                 cfg.lat_center_deg + cfg.lat_spread_deg, n))
    lon = np.deg2rad(rng.uniform(cfg.lon_min_deg, cfg.lon_max_deg, n))
    spd = np.clip(rng.normal(cfg.speed_mean_kn, cfg.speed_sd_kn, n), 2.0, 40.0)
    return lat, lon, spd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminals", type=int, default=40)
    ap.add_argument("--horizon", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results",
                                                      "headroom"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "headroom_log.txt"), "w",
                encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfg = Scenario()
    rep = service_floor_report(cfg, None)
    floor = rep["service_floor_elev_deg_implied_rate"]
    rain_rate = rep["implied_rain_rate_mm_h"]

    log("=" * 100)
    log("HEADROOM PROBE -- which metric can carry the refinement layer?")
    log(f"  M={args.terminals}  horizon={args.horizon:.0f} s  seed={args.seed}"
        f"  cap C={CAPACITY_PER_SAT}")
    log(f"  rain R={rain_rate:.2f} mm/h -> service floor {floor:.2f} deg")
    log("=" * 100)

    con = build_constellation(cfg.n_planes, cfg.sats_per_plane, cfg.altitude_km,
                              np.deg2rad(cfg.inclination_deg), cfg.phase_factor)
    lats, lons, spds = make_terminals(args.terminals, cfg, args.seed)
    t0 = time.time()
    log(f"building {args.terminals} pass tables ...")
    tables = [build_pass_table(float(lats[i]), float(lons[i]), float(spds[i]),
                               con, cfg, args.horizon, dt_samp=2.0)
              for i in range(args.terminals)]
    n_pass = np.array([t.n_pass for t in tables])
    log(f"  done in {time.time() - t0:.0f} s; passes/term mean "
        f"{n_pass.mean():.0f}")
    if (n_pass == 0).any():
        raise SystemExit("ABORT: terminals with no visible satellite "
                         "(geography/units error, not a result)")

    base = SCHEMES[ARM_KEY]
    dt = DT_EXEC_MS[ARM_KEY] * 1e-3
    out = {}
    for rule in TARGET_RULES:
        sch = replace(base, target=rule)
        trs = [simulate_terminal(
            tables[i], sch, cfg, con, float(lats[i]), float(lons[i]),
            float(spds[i]), dt_exec_s=dt, dt_ho_s=dt + DT_HO_EXTRA_MS * 1e-3,
            horizon_s=args.horizon, trace_serving=True,
            trace_sat=True) for i in range(args.terminals)]
        s = summarise(trs, cfg, args.horizon, rain_rate_mm_h=rain_rate,
                      service_floor_elev_deg=floor)
        s["m5_concurrency"] = concurrency(trs, CAPACITY_PER_SAT)
        out[rule] = s
        log(f"  {rule:<20} done")

    def col(rule, path):
        n = out[rule]
        for step in path:
            n = n.get(step) if isinstance(n, dict) else None
            if n is None:
                return float("nan")
        return float(n)

    rows = [
        ("M1 target elevation [deg]", ["m1_target_quality",
                                       "target_elev_mean"], "higher"),
        ("M1 target SNR [dB]", ["m1_target_quality", "target_snr_mean"],
         "higher"),
        ("M2 A_clear [%]", ["m2_availability", "a_clear"], "higher"),
        ("M2 serving SNR mean [dB]", ["m2_availability",
                                      "serving_snr_mean"], "higher"),
        ("M2 serving elev mean [deg]", ["m2_availability",
                                        "serving_elev_mean_deg"], "higher"),
        ("M3 unnecessary by elev [%]", ["m3_unnecessary",
                                        "share_by_elevation"], "lower"),
        ("M4 frequency [1/h]", ["m4_frequency_per_h"], "n/a"),
        ("M5 peak concurrent [cap 50]", ["m5_concurrency",
                                         "peak_concurrent"], "lower"),
        ("M5 busiest-sat share [%]", ["m5_concurrency", "top1_share"], "lower"),
    ]

    log("")
    log("=" * 100)
    log(f"{'metric':<30}" + "".join(f"{r[:12]:>14}" for r in TARGET_RULES)
        + f"{'spread':>10}  direction")
    log("-" * 100)
    verdicts = {}
    for title, path, better in rows:
        vals = []
        for r in TARGET_RULES:
            v = col(r, path)
            if path[-1] in ("a_clear", "share_by_elevation", "top1_share"):
                v = 100.0 * v
            vals.append(v)
        sp = max(vals) - min(vals)
        log(f"{title:<30}" + "".join(f"{v:>14.4f}" for v in vals)
            + f"{sp:>10.4f}  {better}")
        verdicts[title] = {"values": vals, "spread": sp, "better": better}

    log("")
    log("=" * 100)
    log("VERDICT")
    log("=" * 100)
    for title, v in verdicts.items():
        if not np.isfinite(v["spread"]) or v["spread"] < 1e-9:
            log(f"  {title:<30} NO ROOM    -- every rule ties; metric cannot "
                "separate any policy")
        elif v["better"] == "n/a":
            log(f"  {title:<30} AMBIGUOUS  -- spread {v['spread']:.3f}, but "
                "neither direction is unambiguously better (below the "
                "geometric necessary rate means link loss)")
        else:
            log(f"  {title:<30} SPREAD {v['spread']:.4f} ({v['better']} is "
                "better)")
    log("")
    log("  A metric with NO ROOM cannot carry the framework-vs-refinement")
    log("  decomposition: the framework layer's rule already attains it, so a")
    log("  trained policy cannot move it and a zero is a statement about the")
    log("  metric rather than about the layer.")

    with open(os.path.join(args.out_dir, "headroom.json"), "w",
              encoding="utf-8") as f:
        json.dump({"spec": {"terminals": args.terminals,
                            "horizon_s": args.horizon, "seed": args.seed,
                            "capacity_per_sat": CAPACITY_PER_SAT},
                   "rain": rep, "target_rules": list(TARGET_RULES),
                   "per_rule": out, "verdicts": verdicts},
                  f, indent=2, ensure_ascii=False, default=float)
    log("")
    log(f"wrote {os.path.join(args.out_dir, 'headroom.json')}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
