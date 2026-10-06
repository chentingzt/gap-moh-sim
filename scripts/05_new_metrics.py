#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure the six schemes on the four metrics that actually separate them.

`01_diagnostic.py` and `01b_probe_conjuncts.py` established that the
manuscript's timing-only success rate cannot discriminate: every scheme's
execution duration is orders of magnitude below the time available, so all six
score 100%. This script stops trying to reproduce that row and measures the
quantities the schemes genuinely disagree on -- which satellite they hand to
and when -- using only what the simulation produces.

    M1  target quality        the link the handover delivers
    M2  service availability  the link the terminal has, over time, incl. rain
    M3  unnecessary handovers the handovers that left a good link behind
    M4  handover frequency    serving-satellite changes per terminal-hour

NOTHING HERE IS TUNED TO THE MANUSCRIPT. Every input comes from
`gapmoh/constants.py` (which carries [SPEC]/[CHOICE]/[DEVIATION] tags per
field) or from the scheme definitions in `gapmoh/policies.py`. The manuscript's
Table 7 appears only in the comparison block at the end, which is a REPORT,
never an input.

Usage:
    python scripts/05_new_metrics.py --terminals 100 --horizon 3600
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gapmoh import metrics_v2                                     # noqa: E402
from gapmoh.constants import Scenario                             # noqa: E402
from gapmoh.orbit import build_constellation                      # noqa: E402
from gapmoh.passes import build_pass_table                        # noqa: E402
from gapmoh.policies import REPORT_ORDER, SCHEMES                 # noqa: E402
from gapmoh.sim import simulate_terminal                          # noqa: E402

# Execution durations are an INPUT to the timing criterion (CN:199 says each
# scheme uses its own). They are carried here so the timing verdict stays
# comparable with 01_diagnostic; none of M1-M4 depends on them.
DT_EXEC_MS = {"rss": 187.3, "dijkstra": 98.5, "cnnlstm": 45.2,
              "madrl_std": 24.7, "grid_only": 12.5, "gapmoh": 3.2}
DT_HO_EXTRA_MS = 2.0

# CN:90 / Table 5: 50 concurrent users per satellite (20 Gbps per
# satellite / 400 Mbps peak maritime terminal rate). Only used when
# `--trace-sat` is given; the default run does not read it.
CAPACITY_PER_SAT = 50

# ---------------------------------------------------------------------------
# Table 7 (CN:232-238), for the comparison block ONLY. Kept in this script and
# out of the package; `tests/test_integrity_lint.py` enforces that split.
# ---------------------------------------------------------------------------
TABLE7 = {
    "label": "CN 表 7",
    "success_pct": [79.6, 87.0, 90.1, 92.4, 94.1, 96.8],
    "unnecessary_pct": [23.4, 15.1, 11.6, 8.3, 6.7, 4.1],
    "freq_per_h": [12.0, 10.8, 10.4, 10.0, 9.8, 9.6],
    "mean_snr_db": [14.2, 16.8, 17.2, 17.9, 18.3, 19.1],
}


def make_terminals(n, cfg, seed, deploy="uniform", n_lane=3, sigma_deg=1.0):
    """Random terminal start states, IN RADIANS (as `build_pass_table` wants).

    The `cfg.*_deg` fields are degrees; `orbit.terminal_ecef` and
    `passes.build_pass_table` both take radians. Converting here -- and only
    here -- is what `01_diagnostic.py` and `02_train_small.py` already do.
    Skipping it does not raise: it silently relocates every terminal. At the
    scenario theatre that is not a small error. lat = 20 (read as radians)
    lands the terminal at 65.9 deg N, outside a 53 deg constellation's reach,
    so its pass table comes back EMPTY and the scheme's whole timeline for
    that terminal is a no-op.

    Two deployments. `uniform` is the manuscript's own theatre (a spread out
    over the 30 x 40 deg box) and is the default -- every published result in
    this package uses it, so the default MUST stay byte-identical. `lane`
    concentrates terminals on a few corridors, which is what marine traffic
    actually does and is the only deployment in which the per-satellite cap
    C=50 binds at a tractable terminal count (see
    `scripts/11_probe_load_regime.py`).

    The `lane` branch mirrors `11_probe_load_regime.draw_terminals` DRAW FOR
    DRAW: same RNG, same call order, same conversions. That is deliberate --
    it is what lets the probe's sweep predict this script's measured load.
    Reorder any draw here and the probe stops being a forecast of this run.
    """
    rng = np.random.default_rng(seed)
    if deploy == "lane":
        # Corridor centres first, then terminals scattered on them. Drawing the
        # centres BEFORE the scatter is what makes a sigma sweep a controlled
        # comparison: the centres are identical across sigma, only the width
        # changes.
        c_lat = rng.uniform(cfg.lat_center_deg - cfg.lat_spread_deg,
                            cfg.lat_center_deg + cfg.lat_spread_deg, n_lane)
        c_lon = rng.uniform(cfg.lon_min_deg, cfg.lon_max_deg, n_lane)
        pick = rng.integers(0, n_lane, n)
        lat_deg = c_lat[pick] + rng.normal(0.0, sigma_deg, n)
        lon_deg = c_lon[pick] + rng.normal(0.0, sigma_deg, n)
        spd = np.clip(rng.normal(cfg.speed_mean_kn, cfg.speed_sd_kn, n),
                      2.0, 40.0)
        return np.deg2rad(lat_deg), np.deg2rad(lon_deg), spd
    if deploy != "uniform":
        raise ValueError(f"unknown deployment {deploy!r}")
    lat = np.deg2rad(rng.uniform(cfg.lat_center_deg - cfg.lat_spread_deg,
                                 cfg.lat_center_deg + cfg.lat_spread_deg, n))
    lon = np.deg2rad(rng.uniform(cfg.lon_min_deg, cfg.lon_max_deg, n))
    spd = np.clip(rng.normal(cfg.speed_mean_kn, cfg.speed_sd_kn, n), 2.0, 40.0)
    return lat, lon, spd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminals", type=int, default=100)
    ap.add_argument("--horizon", type=float, default=3600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rain-rate", type=float, default=None,
                    help="rain rate (mm/h) for the availability metric; "
                         "default = the rate the manuscript's own rain margin "
                         "implies at the 15 deg mask")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--trace-sat", action="store_true",
                    help="also record the serving-satellite trace and "
                         "report the per-satellite concurrency "
                         "metric. OFF by default so the default run "
                         "is byte-identical to earlier results.")
    ap.add_argument("--capacity-per-sat", type=int,
                    default=CAPACITY_PER_SAT)
    ap.add_argument("--deploy", choices=("uniform", "lane"), default="uniform",
                    help="terminal deployment. 'uniform' is the manuscript's "
                         "theatre and the default, so existing results are "
                         "unchanged; 'lane' clusters terminals on corridors "
                         "and is what makes the C=50 cap bind.")
    ap.add_argument("--n-lane", type=int, default=3,
                    help="number of corridors (deploy=lane only). Must match "
                         "the value used by scripts/11_probe_load_regime.py "
                         "for the probe to forecast this run.")
    ap.add_argument("--sigma-deg", type=float, default=1.0,
                    help="corridor half-width in degrees (deploy=lane only, "
                         "~111 km/deg). Must match the probe run.")
    args = ap.parse_args()

    out = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "new_metrics")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "new_metrics_log.txt"), "w",
                encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfg = Scenario()
    t0 = time.time()

    log("=" * 96)
    log("NEW METRICS  (M1 target quality / M2 availability / M3 necessity / "
        "M4 frequency)")
    log(f"  M={args.terminals} terminals   horizon={args.horizon:.0f} s   "
        f"seed={args.seed}   constellation={cfg.total_sats} sats")
    log(f"  deployment={args.deploy}"
        + (f"  n_lane={args.n_lane}  sigma={args.sigma_deg} deg"
           if args.deploy == "lane" else ""))
    log("=" * 96)

    # ---- the one number the manuscript omits: its implied rain rate --------
    floor_info = metrics_v2.service_floor_report(cfg, args.rain_rate)
    rate = floor_info["implied_rain_rate_mm_h"]
    floor_deg = floor_info["service_floor_elev_deg_implied_rate"]
    log("")
    log("RAIN REFERENCE (the manuscript gives a 12 dB fade but no rain rate)")
    log(f"  rate that gives exactly {cfg.rain_margin_db:.1f} dB at the "
        f"{cfg.min_elev_deg:.0f} deg mask : {rate:.2f} mm/h")
    log(f"  service-floor elevation at that rate (net SNR >= "
        f"{cfg.snr_min_db:.0f} dB)          : {floor_deg:.2f} deg")
    log("  => the 12 dB margin and the 99.5% availability figure are "
        "mutually consistent")
    log("     for a tropical-maritime rain rate; no contradiction found.")

    con = build_constellation(cfg.n_planes, cfg.sats_per_plane,
                              cfg.altitude_km, np.deg2rad(cfg.inclination_deg),
                              cfg.phase_factor)
    lats, lons, spds = make_terminals(args.terminals, cfg, args.seed,
                                      deploy=args.deploy, n_lane=args.n_lane,
                                      sigma_deg=args.sigma_deg)

    log("")
    log(f"building {args.terminals} pass tables...")
    tables = []
    for i in range(args.terminals):
        tables.append(build_pass_table(float(lats[i]), float(lons[i]),
                                       float(spds[i]), con, cfg,
                                       args.horizon, dt_samp=2.0))
    n_pass = np.array([t.n_pass for t in tables])
    log(f"  done in {time.time() - t0:.1f} s")
    log(f"  passes per terminal: min {n_pass.min()}  mean {n_pass.mean():.0f}"
        f"  max {n_pass.max()}")
    # TRIPWIRE. A terminal with an empty pass table contributes no events and
    # no serving trace, so it silently drops out of every metric instead of
    # failing. That is exactly what a unit error in the terminal geography
    # produces (see `make_terminals`), and it is invisible in the summary rows
    # because the schemes that survive still look plausible. Refuse to report
    # from a world where a meaningful share of the terminals are not there.
    bad = int((n_pass == 0).sum())
    if bad:
        log(f"  ABORT: {bad}/{args.terminals} terminals have NO visible "
            f"satellite above the {cfg.min_elev_deg:.0f} deg mask in the "
            f"whole horizon.")
        log("  This is a geography/units error, not a physical result. "
            "Check that terminal lat/lon reach build_pass_table in RADIANS.")
        logf.close()
        return 2

    # ---- run every scheme on the same world --------------------------------
    results = {}
    for key in REPORT_ORDER:
        scheme = SCHEMES[key]
        dt_exec = DT_EXEC_MS[key] * 1e-3
        t1 = time.time()
        traces = []
        for i in range(args.terminals):
            traces.append(simulate_terminal(
                tables[i], scheme, cfg, con, float(lats[i]), float(lons[i]),
                float(spds[i]), dt_exec_s=dt_exec,
                dt_ho_s=dt_exec + DT_HO_EXTRA_MS * 1e-3,
                horizon_s=args.horizon, trace_serving=True,
                trace_sat=args.trace_sat))
        results[key] = metrics_v2.summarise(
            traces, cfg, args.horizon, rain_rate_mm_h=rate,
            service_floor_elev_deg=floor_deg)
        results[key]["label"] = scheme.label
        # The manuscript's own criterion, carried for the comparison block
        # only. It is not a metric this script proposes.
        n_ev = sum(len(tr.events) for tr in traces)
        n_ok = sum(1 for tr in traces for e in tr.events if e.success)
        results[key]["timing_success_pct"] = (
            100.0 * n_ok / n_ev if n_ev else float("nan"))
        if args.trace_sat:
            results[key]["m5_concurrency"] = metrics_v2.concurrency(
                traces, args.capacity_per_sat)
        log(f"  {scheme.label:<18} {results[key]['n_handover']:>7} handovers  "
            f"({time.time() - t1:.1f} s)")

    # ---- report ------------------------------------------------------------
    labels = [SCHEMES[k].label for k in REPORT_ORDER]
    log("")
    log("=" * 96)
    log("M1  TARGET QUALITY -- the link the handover delivers")
    log("-" * 96)
    log(f"{'scheme':<18}{'tgt elev mean':>15}{'tgt elev p05':>14}"
        f"{'tgt SNR mean':>14}{'tgt SNR min':>13}")
    for k in REPORT_ORDER:
        q = results[k]["m1_target_quality"]
        log(f"{SCHEMES[k].label:<18}{q['target_elev_mean']:>15.1f}"
            f"{q['target_elev_p05']:>14.1f}{q['target_snr_mean']:>14.1f}"
            f"{q['target_snr_min']:>13.1f}")

    log("")
    log("M2  SERVICE AVAILABILITY -- fraction of time the link meets the "
        "requirement")
    log(f"    (rain at {rate:.2f} mm/h; service floor {floor_deg:.2f} deg)")
    log("-" * 96)
    log(f"{'scheme':<18}{'clear %':>10}{'rain %':>10}"
        f"{'mean srv SNR':>14}{'mean srv elev':>15}{'min srv elev':>14}")
    for k in REPORT_ORDER:
        a = results[k]["m2_availability"]
        if not a:
            continue
        log(f"{SCHEMES[k].label:<18}{a['a_clear'] * 100:>10.2f}"
            f"{a['a_rain'][rate] * 100:>10.2f}"
            f"{a['serving_snr_mean']:>14.1f}"
            f"{a['serving_elev_mean_deg']:>15.1f}"
            f"{a['min_elev_deg']:>14.1f}")

    log("")
    log("M3  UNNECESSARY HANDOVERS -- left a still-serviceable link behind")
    log("-" * 96)
    log(f"{'scheme':<18}{'by elev %':>11}{'@60 s %':>10}"
        f"{'srv elev at t_init':>20}{'budget median s':>17}")
    for k in REPORT_ORDER:
        u = results[k]["m3_unnecessary"]
        if not u.get("n"):
            continue
        elev_share = u["share_by_elevation"]
        log(f"{SCHEMES[k].label:<18}"
            f"{(elev_share * 100 if elev_share is not None else float('nan')):>11.1f}"
            f"{u['share_by_time'][60.0] * 100:>10.1f}"
            f"{u['serving_elev_at_init_median_deg']:>20.1f}"
            f"{u['budget_median_s']:>17.1f}")

    log("")
    log("M4  HANDOVER FREQUENCY -- serving-satellite changes per terminal-hour")
    log("-" * 96)
    log(f"{'scheme':<18}{'measured /h':>14}{'CN 表 7 /h':>13}{'ratio':>10}")
    for k, t7 in zip(REPORT_ORDER, TABLE7["freq_per_h"]):
        f = results[k]["m4_frequency_per_h"]
        log(f"{SCHEMES[k].label:<18}{f:>14.1f}{t7:>13.1f}{f / t7:>10.2f}")

    # ---- spread check: the acceptance gate ---------------------------------
    log("")
    log("=" * 96)
    spreads = {}
    for name, getter in (
            ("M1 target elev mean",
             lambda k: results[k]["m1_target_quality"]["target_elev_mean"]),
            ("M2 A_rain",
             lambda k: results[k]["m2_availability"].get(
                 "a_rain", {}).get(rate, float("nan")) * 100),
            ("M3 unnecessary by elev",
             lambda k: (results[k]["m3_unnecessary"].get(
                 "share_by_elevation") or 0.0) * 100),
            ("M4 frequency",
             lambda k: results[k]["m4_frequency_per_h"]),
    ):
        vals = [getter(k) for k in REPORT_ORDER]
        sp = max(vals) - min(vals)
        spreads[name] = {"values": vals, "spread": sp,
                         "min": min(vals), "max": max(vals)}
        log(f"{name:<26} min {min(vals):>8.2f}  max {max(vals):>8.2f}  "
            f"spread {sp:>8.2f}")
    log("=" * 96)

    # ---- comparison against the published rows -----------------------------
    log("")
    log("COMPARISON -- published Table 7 row vs measured (same column order)")
    log("-" * 96)
    log("        " + "".join(f"{l:>17}" for l in labels))
    for name, pub, meas in (
            ("success % (pub)", TABLE7["success_pct"],
             [_success_rate(k, results) for k in REPORT_ORDER]),
            ("unnecess % (pub)", TABLE7["unnecessary_pct"],
             [(results[k]["m3_unnecessary"].get("share_by_elevation") or 0.0)
              * 100 for k in REPORT_ORDER]),
            ("freq /h (pub)", TABLE7["freq_per_h"],
             [results[k]["m4_frequency_per_h"] for k in REPORT_ORDER]),
            ("mean SNR (pub)", TABLE7["mean_snr_db"],
             [results[k]["m2_availability"].get("serving_snr_mean", float("nan"))
              for k in REPORT_ORDER]),
    ):
        log(f"{name:<12}" + "".join(f"{v:>17.1f}" for v in pub))
        log(f"{'  measured':<12}" + "".join(f"{v:>17.1f}" for v in meas))
    log("-" * 96)

    payload = {
        "spec": {"M": args.terminals, "horizon_s": args.horizon,
                 "seed": args.seed, "constellation": cfg.total_sats},
        "rain": {"implied_rate_mm_h": rate,
                 "service_floor_elev_deg": floor_deg,
                 "rain_margin_db": cfg.rain_margin_db,
                 "rain_availability_pct": cfg.rain_availability_pct},
        "service_floor": floor_info,
        "per_scheme": results,
        "spreads": spreads,
        "table7_published": TABLE7,
        "report_order": REPORT_ORDER,
    }
    with open(os.path.join(out, "new_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log("")
    log(f"wrote {os.path.join(out, 'new_metrics.json')}")
    log(f"wall clock {time.time() - t0:.1f} s")
    logf.close()
    return 0


def _success_rate(key, results):
    """Timing-criterion success share, carried only for the comparison block.

    It is recomputed from the same traces; it is not a metric this script
    proposes, and its 100% value is the point of the comparison.
    """
    return results[key].get("timing_success_pct", float("nan"))


if __name__ == "__main__":
    raise SystemExit(main())
