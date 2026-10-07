#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R-B 探针：为「非必要切换」找一个**非退化**的替代比较量。

WHY. 复审 C-4 判定表 7 的「非必要切换」行退化：六方案 100/100/100/95.35/0/0，
5/6 落在端点、端点值跨种子零方差 —— 与被删除的「时序可行性成功率」行同病。

根因是**结构性**的：该量把「发起切换时服务链路相对雨衰服务下限的余量」二值化
（> 16.67° 记非必要）。GBPT 按定义在边界触发 ⇒ 服务链路恰好降到下限 ⇒ 恒 0；
门限式方案的门限远高于下限 ⇒ 恒 100%。**任何同构的二值化都救不回来**，因为
触发时机是各方案规则里写死的。

所以本探针同时评估几个**非二值、且不是任何方案目标函数**的候选量，报告各方案
之间的极差与端点占比，供选择。本脚本只做测量，不改任何稿件、不调参。

    python scripts/12_probe_rb.py --terminals 20 --horizon 1800 --seeds 0,1,2

产出 results/rb_probe/rb_probe.json 与一张控制台表。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gapmoh.channel import snr_from_elev_db
from gapmoh.constants import Scenario
from gapmoh.metrics_v2 import DEFAULT_RAIN_RATES_MM_H
from gapmoh.channel import rain_loss_db
from gapmoh.orbit import build_constellation
from gapmoh.passes import build_pass_table
from gapmoh.policies import REPORT_ORDER, SCHEMES
from gapmoh.sim import simulate_terminal

# Table 7 row 2 (execution durations, ms) -- an INPUT to the timing criterion,
# allowed under scripts/ (same as 01_diagnostic.py), forbidden under gapmoh/.
TABLE7_DT_EXEC_MS = {
    "rss": 187.3, "dijkstra": 98.5, "cnnlstm": 45.2,
    "madrl_std": 24.7, "grid_only": 12.5, "gapmoh": 3.2,
}
DT_HO_EXTRA_MS = 2.0

# 雨衰服务下限的仰角（°），由 12 dB 余量反解，见 results/new_metrics。
SERVICE_FLOOR_DEG = 16.667509547656717
# 与之配套的隐含雨强（mm/h）
IMPLIED_RATE = 7.644228019779442


def make_terminals(n, cfg, seed):
    rng = np.random.default_rng(seed)
    lat = np.deg2rad(rng.uniform(cfg.lat_center_deg - cfg.lat_spread_deg,
                                 cfg.lat_center_deg + cfg.lat_spread_deg, n))
    lon = np.deg2rad(rng.uniform(cfg.lon_min_deg, cfg.lon_max_deg, n))
    spd = np.clip(rng.normal(cfg.speed_mean_kn, cfg.speed_sd_kn, n), 2.0, 40.0)
    return lat, lon, spd


def _degeneracy(vals):
    """端点占比：取值落在 [0,0.5] 或 [99.5,100] 的方案比例（百分比量）。"""
    n = len(vals)
    at_end = sum(1 for v in vals if v <= 0.5 or v >= 99.5)
    return at_end / n if n else float("nan")


def _spread(vals):
    v = [x for x in vals if x == x]
    return (max(v) - min(v)) if len(v) > 1 else float("nan")


def metrics_for(events, traces):
    """一个方案的全部候选量。events 为该方案全部终端的切换事件。"""
    if not events:
        return None
    ev = sorted(events, key=lambda e: e.t_init)
    budget = np.array([e.budget_s for e in ev], dtype=float)
    tremain = np.array([e.target_remaining_s for e in ev], dtype=float)
    serv_elev = np.array([e.serving_elev_deg for e in ev], dtype=float)
    serv_snr = np.array([e.serving_snr_db for e in ev], dtype=float)
    tgt_elev = np.array([e.target_elev_deg for e in ev], dtype=float)

    out = {}

    # ---- 现口径（预期退化）：发起时服务仰角 > 服务下限 ----
    out["m3_elev_share_pct"] = float(np.mean(serv_elev > SERVICE_FLOOR_DEG) * 100)

    # ---- 按剩余可见时间 ----
    for t in (30.0, 60.0, 120.0):
        out["m3_time_ge_%ds_pct" % int(t)] = float(np.mean(budget >= t) * 100)

    # ---- 连续量：发起时服务链路的几何余量 ----
    out["serving_margin_median_deg"] = float(np.median(serv_elev)
                                             - SERVICE_FLOOR_DEG)
    out["serving_margin_p05_deg"] = float(np.percentile(serv_elev, 5)
                                          - SERVICE_FLOOR_DEG)

    # ---- 后悔率：把更长的链路换成更短的 ----
    out["regret_pct"] = float(np.mean(tremain < budget) * 100)

    # ---- 目标驻留（连续）----
    out["target_dwell_median_s"] = float(np.median(tremain))
    out["target_dwell_p05_s"] = float(np.percentile(tremain, 5))

    # ---- 乒乓：同一条终端上两次切换相隔 < T ----
    for T in (60.0, 120.0):
        gaps = []
        for tr in traces:
            ts = sorted(e.t_init for e in tr.events)
            gaps.extend(np.diff(ts).tolist())
        gaps = np.array(gaps, dtype=float)
        out["pingpong_lt_%ds_pct" % int(T)] = (
            float(np.mean(gaps < T) * 100) if gaps.size else float("nan"))

    # ---- 代价侧：服务链路最坏质量（连续、非退化）----
    snr_all, elev_all = [], []
    for tr in traces:
        if tr.serving_trace is None:
            continue
        a = np.asarray(tr.serving_trace, dtype=float)
        snr_all.append(a[:, 2])
        elev_all.append(np.rad2deg(a[:, 1]))
    if snr_all:
        snr_all = np.concatenate(snr_all)
        elev_all = np.concatenate(elev_all)
        out["serving_snr_p05_db"] = float(np.percentile(snr_all, 5))
        out["serving_elev_p05_deg"] = float(np.percentile(elev_all, 5))
        out["serving_elev_min_deg"] = float(elev_all.min())
        rain = rain_loss_db(np.deg2rad(elev_all), Scenario(), IMPLIED_RATE)
        out["a_rain_pct"] = float(np.mean((snr_all - rain)
                                          >= Scenario().snr_min_db) * 100)
    out["n_handover"] = int(len(ev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminals", type=int, default=20)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else root / "results" / "rb_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Scenario()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()

    con = build_constellation(cfg.n_planes, cfg.sats_per_plane,
                              cfg.altitude_km, np.deg2rad(cfg.inclination_deg),
                              cfg.phase_factor)

    per_seed = {}
    for seed in seeds:
        lats, lons, spds = make_terminals(args.terminals, cfg, seed)
        tables = [build_pass_table(float(lats[i]), float(lons[i]),
                                   float(spds[i]), con, cfg, args.horizon,
                                   dt_samp=2.0)
                  for i in range(args.terminals)]
        per_seed[seed] = {}
        for key in REPORT_ORDER:
            scheme = SCHEMES[key]
            dt_exec = TABLE7_DT_EXEC_MS[key] * 1e-3
            dt_ho = dt_exec + DT_HO_EXTRA_MS * 1e-3
            traces = [simulate_terminal(
                tables[i], scheme, cfg, con, float(lats[i]), float(lons[i]),
                float(spds[i]), dt_exec_s=dt_exec, dt_ho_s=dt_ho,
                horizon_s=args.horizon, trace_serving=True)
                for i in range(args.terminals)]
            events = [e for tr in traces for e in tr.events]
            per_seed[seed][key] = metrics_for(events, traces)
        print("seed %d done  elapsed %.0f s" % (seed, time.time() - t0),
              flush=True)

    # ---- 聚合：跨种子均值 + 方案间极差 + 端点占比 ----
    fields = sorted({k for s in per_seed.values() for v in s.values()
                     if v for k in v})
    agg = {}
    for key in REPORT_ORDER:
        agg[key] = {"label": SCHEMES[key].label}
        for f in fields:
            vals = [per_seed[s][key][f] for s in seeds
                    if per_seed[s].get(key) and f in per_seed[s][key]]
            vals = [v for v in vals if v is not None and v == v]
            agg[key][f] = float(np.mean(vals)) if vals else float("nan")

    lines = []
    lines.append("R-B 候选比较量探针  M=%d  horizon=%.0f s  seeds=%s"
                 % (args.terminals, args.horizon, seeds))
    lines.append("（数值为跨种子均值；极差=方案间 max-min；端点=落在 0/100 "
                 "两端的方案占比）")
    lines.append("")
    hdr = "%-26s" % "候选量"
    for key in REPORT_ORDER:
        hdr += "%14s" % SCHEMES[key].label[:12]
    hdr += "%10s%8s" % ("极差", "端点")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for f in fields:
        vals = [agg[k][f] for k in REPORT_ORDER]
        row = "%-26s" % f
        for v in vals:
            row += "%14.2f" % v if v == v else "%14s" % "nan"
        is_pct = f.endswith("_pct")
        row += "%10.2f" % _spread(vals)
        row += "%8.2f" % (_degeneracy(vals) if is_pct else float("nan"))
        lines.append(row)

    text = "\n".join(lines)
    (out_dir / "rb_probe.txt").write_text(text, encoding="utf-8")
    (out_dir / "rb_probe.json").write_text(
        json.dumps({"config": vars(args), "per_seed": per_seed, "agg": agg},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    print("\nwrote %s" % (out_dir / "rb_probe.json"))
    print("total wall clock: %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
