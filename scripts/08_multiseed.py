#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run `05_new_metrics.py` over the manuscript's 10 seeds and aggregate.

WHY 10. The manuscript states its whole results table is 10 independent seeds
with 95% confidence intervals (t_{0.975,9} = 2.262), and every other table in
the paper is quoted that way. Rebuilding one row from a single seed would leave
that row in a different statistical protocol from the rest of the paper, which
is exactly the kind of silent inconsistency this work is trying to remove.

RESUMABLE BY DESIGN. Each seed writes its own `new_metrics.json` under
`results/multiseed/seed<k>/`. A seed whose file already exists is skipped, so
an interrupted run costs only the seeds still in flight. Re-run the same
command to continue.

    python scripts/08_multiseed.py                 # seeds 0-9, 4 workers
    python scripts/08_multiseed.py --seeds 0,1,2   # a subset
    python scripts/08_multiseed.py --aggregate-only

The aggregation reports, per scheme, the mean and half-width of the 95% CI
across seeds, in the shape the manuscript's tables use.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Two-sided 95% t multipliers by degrees of freedom (n - 1). Falls back to the
# normal quantile, which is the correct large-sample limit.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042}


def t95(n):
    if n < 2:
        return float("nan")
    if n - 1 in _T95:
        return _T95[n - 1]
    floor = max(k for k in _T95 if k <= n - 1)
    return _T95[floor]


def seed_dir(base, k):
    return os.path.join(base, f"seed{k}")


def run_seed(k, base, terminals, horizon, workers_note="",
             trace_sat=False, deploy="uniform", n_lane=3, sigma_deg=1.0):
    """One seed, via the already-validated `05_new_metrics.py`."""
    out = seed_dir(base, k)
    target = os.path.join(out, "new_metrics.json")
    if os.path.exists(target):
        return k, "cached", 0.0
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "05_new_metrics.py"),
           "--terminals", str(terminals), "--horizon", str(horizon),
           "--seed", str(k), "--out-dir", out]
    if trace_sat:
        cmd.append("--trace-sat")
    if deploy != "uniform":
        # Only emitted off the default, so every existing run's command line
        # is unchanged and its cached output stays valid.
        cmd += ["--deploy", deploy, "--n-lane", str(n_lane),
                "--sigma-deg", str(sigma_deg)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    ok = os.path.exists(target)
    return k, ("ok" if ok else "FAILED"), time.time() - t0


def _ci(vals):
    """(mean, half-width of the 95% CI) over a list of per-seed values."""
    x = [v for v in vals if v is not None and v == v]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(x) / n
    if n < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in x) / (n - 1)
    return m, t95(n) * (var ** 0.5) / (n ** 0.5)


def aggregate(base, order, out_path):
    seeds = []
    for name in sorted(os.listdir(base)):
        if not name.startswith("seed"):
            continue
        p = os.path.join(base, name, "new_metrics.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                seeds.append((name, json.load(f)))
    if not seeds:
        return None, []

    def series(path):
        """Per-seed values of one scalar, in seed order."""
        out = []
        for _, d in seeds:
            node = d
            try:
                for step in path:
                    node = node[step]
            except (KeyError, TypeError):
                node = None
            out.append(node)
        return out

    def has(path):
        """True if any seed carries this scalar (m5 is opt-in)."""
        return any(v is not None for v in series(path))

    rows = {}
    for key in order:
        rows[key] = {
            "label": seeds[0][1]["per_scheme"][key]["label"],
            "target_elev_mean": _ci(series(["per_scheme", key,
                                            "m1_target_quality",
                                            "target_elev_mean"])),
            "target_snr_mean": _ci(series(["per_scheme", key,
                                           "m1_target_quality",
                                           "target_snr_mean"])),
            "serving_snr_mean": _ci(series(["per_scheme", key,
                                            "m2_availability",
                                            "serving_snr_mean"])),
            "serve_elev_mean": _ci(series(["per_scheme", key,
                                           "m2_availability",
                                           "serving_elev_mean_deg"])),
            "unnecessary_pct": _ci([100.0 * v if v is not None else None
                                    for v in series(
                                        ["per_scheme", key, "m3_unnecessary",
                                         "share_by_elevation"])]),
            "freq_per_h": _ci(series(["per_scheme", key,
                                      "m4_frequency_per_h"])),
        }
        # A_rain at the manuscript-implied rate, keyed by a stringified float.
        rate = seeds[0][1]["rain"]["implied_rate_mm_h"]
        a_rain = []
        for _, d in seeds:
            m = d["per_scheme"][key]["m2_availability"].get("a_rain", {})
            v = None
            for kk, vv in m.items():
                try:
                    if abs(float(kk) - float(rate)) < 1e-9:
                        v = vv
                except (TypeError, ValueError):
                    continue
            a_rain.append(100.0 * v if v is not None else None)
        rows[key]["a_rain_pct"] = _ci(a_rain)

        # Opt-in load metrics. Present only when every seed was run
        # with --trace-sat; absent otherwise, so the default summary
        # keeps exactly its old shape.
        if has(["per_scheme", key, "m5_concurrency",
                "peak_concurrent"]):
            rows[key]["peak_concurrent"] = _ci(series(
                ["per_scheme", key, "m5_concurrency",
                 "peak_concurrent"]))
            rows[key]["peak_over_capacity"] = _ci(series(
                ["per_scheme", key, "m5_concurrency",
                 "peak_over_capacity"]))
            rows[key]["overflow_terminal_seconds"] = _ci(series(
                ["per_scheme", key, "m5_concurrency",
                 "overflow_terminal_seconds"]))
            rows[key]["top1_share_pct"] = _ci(
                [100.0 * v if v is not None else None
                 for v in series(["per_scheme", key, "m5_concurrency",
                                  "top1_share"])])

    summary = {"n_seeds": len(seeds), "seeds": [n for n, _ in seeds],
               "rain_implied_rate_mm_h": seeds[0][1]["rain"]
               ["implied_rate_mm_h"],
               "service_floor_elev_deg": seeds[0][1]["rain"]
               ["service_floor_elev_deg"],
               "t_multiplier": t95(len(seeds)), "order": order, "rows": rows}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--terminals", type=int, default=100)
    ap.add_argument("--horizon", type=float, default=3600.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base", default=os.path.join(ROOT, "results",
                                                   "multiseed"))
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--trace-sat", action="store_true",
                    help="record the serving-satellite trace and "
                         "aggregate the per-satellite concurrency "
                         "metric (needs the 05 --trace-sat data)")
    ap.add_argument("--deploy", choices=("uniform", "lane"), default="uniform",
                    help="terminal deployment, passed through to 05. "
                         "'uniform' (default) keeps every existing result "
                         "valid; 'lane' is the high-load scenario.")
    ap.add_argument("--n-lane", type=int, default=3)
    ap.add_argument("--sigma-deg", type=float, default=1.0)
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    from gapmoh.policies import REPORT_ORDER

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.base, exist_ok=True)

    if not args.aggregate_only:
        t0 = time.time()
        print(f"running seeds {seeds} at M={args.terminals} "
              f"H={args.horizon:.0f} deploy={args.deploy} "
              f"with {args.workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_seed, k, args.base, args.terminals,
                                args.horizon,
                                trace_sat=args.trace_sat,
                                deploy=args.deploy, n_lane=args.n_lane,
                                sigma_deg=args.sigma_deg)
                    for k in seeds]
            for fut in futs:
                k, status, dt = fut.result()
                print(f"  seed {k}: {status}  ({dt:.0f} s)  "
                      f"elapsed {time.time() - t0:.0f} s", flush=True)

    summary, done = aggregate(args.base, REPORT_ORDER,
                              os.path.join(args.base, "multiseed_summary.json"))
    if summary is None:
        print("no completed seeds to aggregate")
        return 1

    print("")
    print("=" * 100)
    print(f"AGGREGATE over {summary['n_seeds']} seeds "
          f"(t = {summary['t_multiplier']:.3f})")
    print("=" * 100)

    def show(title, field, unit, nd=2):
        print("")
        print(f"{title}  [{unit}]")
        print(f"{'scheme':<18}{'mean':>10}{'±95% CI':>10}")
        for key in REPORT_ORDER:
            m, h = summary["rows"][key][field]
            print(f"{summary['rows'][key]['label']:<18}{m:>10.{nd}f}"
                  f"{h:>10.{nd}f}")

    show("M1 target elevation", "target_elev_mean", "deg")
    show("M1 target SNR", "target_snr_mean", "dB")
    show("M2 serving SNR (clear)", "serving_snr_mean", "dB")
    show("M2 A_rain", "a_rain_pct", "%")
    show("M3 unnecessary (by elevation)", "unnecessary_pct", "%")
    show("M4 frequency", "freq_per_h", "events/h")

    if "peak_concurrent" in summary["rows"][REPORT_ORDER[0]]:
        show("M5 peak concurrent per satellite", "peak_concurrent",
             "users")
        show("M5 peak over capacity", "peak_over_capacity", "users")
        show("M5 overflow", "overflow_terminal_seconds",
             "terminal-s")
        show("M5 busiest-satellite share", "top1_share_pct", "%")

    print("")
    print(f"wrote {os.path.join(args.base, 'multiseed_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
