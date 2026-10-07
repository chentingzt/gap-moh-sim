#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""X-10 closure: the DISTRIBUTION of the handover target's elevation.

WHY. The manuscript reports a single number per scheme for the target link --
its mean elevation -- and a first-round review asked for the distribution
(A-4 / X-10). Two quantities are reported here, because the phrase "the
handover target's elevation" admits both readings and they answer different
questions:

    target_elev       elevation of the handed-to link AT THE HANDOVER INSTANT
                      (this is the quantity whose mean the manuscript prints;
                       its p05/min are what has been missing)
    target_peak_elev  elevation that SAME pass reaches at its own peak -- how
                      good the pass is, not where it happens to stand when the
                      handover fires

Both are per-handover quantities, pooled over the terminals of a seed and then
averaged over the manuscript's own 10 seeds with the same t_{0.975,9} = 2.262
protocol every other row of Table 7 uses. That protocol is why this is a
re-run rather than a post-hoc read: only the mean was ever aggregated into
`multiseed_summary.json`, and the peak was not recorded at all.

NOTHING IS TUNED. The world is `05_new_metrics.py`'s world, unchanged --
`make_terminals` is imported from it so the terminals are drawn by the same
code, and the schemes come from `gapmoh/policies.py`. The script re-derives
`target_elev_mean` and prints it next to the value already in
`results/multiseed/seed<k>/new_metrics.json`; if those ever disagree the world
has drifted and the new numbers must not be used.

    python scripts/13_target_elev.py                 # seeds 0-9, 4 workers
    python scripts/13_target_elev.py --seeds 0       # one seed
    python scripts/13_target_elev.py --aggregate-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def t95(n):
    return _T95.get(n - 1, 1.96) if n >= 2 else float("nan")


def ci(vals):
    """(mean, half-width of the 95% CI) over per-seed values."""
    x = [v for v in vals if v is not None and v == v]
    if not x:
        return float("nan"), float("nan")
    m = sum(x) / len(x)
    if len(x) < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in x) / (len(x) - 1)
    return m, t95(len(x)) * (var ** 0.5) / (len(x) ** 0.5)


def one_seed(seed, terminals, horizon, out_dir):
    """Run the six schemes on this seed's world; return the two distributions."""
    from gapmoh.constants import Scenario
    from gapmoh.metrics_v2 import target_quality
    from gapmoh.orbit import build_constellation
    from gapmoh.passes import build_pass_table
    from gapmoh.policies import REPORT_ORDER, SCHEMES
    from gapmoh.sim import simulate_terminal
    # `05_new_metrics.py` cannot be imported by name (the module name starts
    # with a digit), so it is loaded from its path. Its `main()` is guarded, so
    # this only pulls in the terminal-drawing function -- which is the point:
    # the terminals must be drawn by the same code, or this is not the same
    # world the published rows came from.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "new_metrics_05", os.path.join(ROOT, "scripts", "05_new_metrics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    make_terminals = mod.make_terminals

    cfg = Scenario()
    con = build_constellation(cfg.n_planes, cfg.sats_per_plane,
                              cfg.altitude_km, np.deg2rad(cfg.inclination_deg),
                              cfg.phase_factor)
    lats, lons, spds = make_terminals(terminals, cfg, seed)
    tables = [build_pass_table(float(lats[i]), float(lons[i]),
                               float(spds[i]), con, cfg, horizon, dt_samp=2.0)
              for i in range(terminals)]

    # Same execution durations as 05_new_metrics.py. The target rules do not
    # read them; they are carried so the run is the same command line.
    dt_exec_ms = {"rss": 187.3, "dijkstra": 98.5, "cnnlstm": 45.2,
                  "madrl_std": 24.7, "grid_only": 12.5, "gapmoh": 3.2}

    out = {}
    for key in REPORT_ORDER:
        scheme = SCHEMES[key]
        dt_exec = dt_exec_ms[key] * 1e-3
        events = []
        for i in range(terminals):
            tr = simulate_terminal(
                tables[i], scheme, cfg, con, float(lats[i]), float(lons[i]),
                float(spds[i]), dt_exec_s=dt_exec,
                dt_ho_s=dt_exec + 2.0e-3, horizon_s=horizon)
            events.extend(tr.events)
        q = target_quality(events)
        q["label"] = scheme.label
        out[key] = q

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "target_elev.json"), "w",
              encoding="utf-8") as f:
        json.dump({"seed": seed, "terminals": terminals, "horizon_s": horizon,
                   "per_scheme": out}, f, indent=2, ensure_ascii=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--terminals", type=int, default=100)
    ap.add_argument("--horizon", type=float, default=3600.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base", default=os.path.join(ROOT, "results",
                                                   "target_elev"))
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--one", type=int, default=None,
                    help="run one seed IN-PROCESS (no subprocess pool)")
    args = ap.parse_args()

    from gapmoh.policies import REPORT_ORDER

    if args.one is not None:
        # `one_seed` takes the per-seed OUTPUT DIRECTORY, not the run base.
        # Passing `args.base` here wrote every worker to a single
        # `<base>/target_elev.json`, which the aggregate step (and `job`'s
        # cache probe) then reads as `--base/seed<k>/target_elev.json`; every
        # completed seed was reported FAILED and the file was clobbered.
        one_seed(args.one, args.terminals, args.horizon,
                 os.path.join(args.base, "seed%d" % args.one))
        return 0

    os.makedirs(args.base, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not args.aggregate_only:
        def job(k):
            d = os.path.join(args.base, f"seed{k}")
            if os.path.exists(os.path.join(d, "target_elev.json")):
                return k, "cached"
            os.makedirs(d, exist_ok=True)
            t0 = time.time()
            subprocess.run([sys.executable,
                            os.path.join(ROOT, "scripts", "13_target_elev.py"),
                            "--one", str(k), "--terminals", str(args.terminals),
                            "--horizon", str(args.horizon), "--base", args.base],
                           check=False)
            ok = os.path.exists(os.path.join(d, "target_elev.json"))
            return k, ("ok %.0f s" % (time.time() - t0)) if ok else "FAILED"

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for k, st in pool.map(job, seeds):
                print(f"  seed {k}: {st}   elapsed {time.time() - t0:.0f} s",
                      flush=True)

    # ---- aggregate -------------------------------------------------------
    per = []
    for k in seeds:
        p = os.path.join(args.base, f"seed{k}", "target_elev.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                per.append(json.load(f))
    if not per:
        print("no completed seeds")
        return 1

    fields = ["target_elev_mean", "target_elev_p05", "target_elev_min",
              "target_peak_elev_mean", "target_peak_elev_p05",
              "target_peak_elev_min", "target_snr_mean"]
    rows = {}
    for key in REPORT_ORDER:
        rows[key] = {"label": per[0]["per_scheme"][key]["label"],
                     "n_handover_per_seed": [d["per_scheme"][key]["n"]
                                             for d in per]}
        for f in fields:
            rows[key][f] = ci([d["per_scheme"][key].get(f) for d in per])

    # ---- consistency gate against the cached 10-seed run ------------------
    print("")
    print("=" * 92)
    print("CONSISTENCY CHECK -- target_elev_mean here vs results/multiseed "
          "(same world?)")
    print("=" * 92)
    worst = 0.0
    for key in REPORT_ORDER:
        old = []
        for k in seeds:
            p = os.path.join(ROOT, "results", "multiseed", f"seed{k}",
                             "new_metrics.json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                old.append(json.load(f)["per_scheme"][key]
                           ["m1_target_quality"]["target_elev_mean"])
        if not old:
            print(f"  {rows[key]['label']:<18} (no cached run)")
            continue
        d = abs(sum(old) / len(old) - rows[key]["target_elev_mean"][0])
        worst = max(worst, d)
        print(f"  {rows[key]['label']:<18} cached {sum(old) / len(old):8.4f}"
              f"   here {rows[key]['target_elev_mean'][0]:8.4f}   "
              f"|diff| {d:.2e}")
    print(f"  worst |diff| = {worst:.2e}  -- the world is "
          f"{'IDENTICAL' if worst < 0.02 else 'DIFFERENT (DO NOT USE)'}")
    print("=" * 92)

    def show(title, f_mean, f_p05, f_min, nd=2):
        print("")
        print(title)
        print(f"{'scheme':<18}{'mean':>10}{'±95% CI':>10}"
              f"{'p05':>10}{'min':>10}")
        for key in REPORT_ORDER:
            m, h = rows[key][f_mean]
            p, _ = rows[key][f_p05]
            mn, _ = rows[key][f_min]
            print(f"{rows[key]['label']:<18}{m:>10.{nd}f}{h:>10.{nd}f}"
                  f"{p:>10.{nd}f}{mn:>10.{nd}f}")

    show("X-10  TARGET-LINK ELEVATION AT THE HANDOVER INSTANT  [deg] "
         "(per-handover, pooled per seed, averaged over seeds)",
         "target_elev_mean", "target_elev_p05", "target_elev_min")
    show("X-10  TARGET PASS PEAK ELEVATION  [deg] "
         "(the best this pass ever gets)",
         "target_peak_elev_mean", "target_peak_elev_p05",
         "target_peak_elev_min")

    with open(os.path.join(args.base, "target_elev_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"n_seeds": len(per), "t_multiplier": t95(len(per)),
                   "order": REPORT_ORDER, "rows": rows}, f, indent=2,
                  ensure_ascii=False)
    print("")
    print(f"wrote {os.path.join(args.base, 'target_elev_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
