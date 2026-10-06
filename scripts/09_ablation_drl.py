#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation of the DRL refinement layer under metrics that do not degenerate.

WHY THIS SCRIPT EXISTS. Table 7's decomposition ("deterministic framework layer
+14.5 pp, DRL refinement layer +2.7 pp") rests on row 1, whose criterion is
identically true -- GBPT triggers at t_exit - dt_ho with dt_ho = dt_exec + 2 ms
(CN:140), so `dt_exec < t_exit - t_init` holds by construction for any data.
Delete that row -- which is what the Option-A restructure does -- and the
decomposition loses its metric.

`gapmoh/policies.py` holds two LITERALLY IDENTICAL schemes:

    "grid_only": Scheme("grid_only", "Grid-Only", "gbpt", "highest_elev"),
    "gapmoh":    Scheme("gapmoh",    "GAP-MOH",   "gbpt", "highest_elev"),

so as released, the framework layer and the framework+DRL layer give the same
number in every column. `select_target`'s `"policy"` branch says as much --
"[DEVIATION] ... Stage 1 replaces it with the trained net" -- and `gapmoh/rl/`
is that net (pure numpy; there is no torch on this box). This script is the
replacement: it trains the refinement layer and evaluates it through the SAME
`sim.simulate_terminal` path that produces every other row, via the `policy_fn`
hook added there.

WHAT IT REPORTS -- INCLUDING THE ANSWER "NO HEADROOM".

    framework    GBPT + highest_elev        (the deterministic layer)
    refined      GBPT + trained DQN         (framework + refinement layer)
    long_rem     GBPT + longest_remaining   } scripted references that bound
    best_qual    GBPT + best_quality        } what any target rule can reach

The scripted references matter. On M1 (target elevation) `highest_elev` IS the
argmax over the candidate set, so the framework layer sits AT the metric's
optimum and no policy -- trained or otherwise -- can beat it there. A tie on M1
is a property of the metric, not a failure of the training, and the script says
so in the output instead of leaving a tie to be misread. Where the refinement
layer CAN move the number (multi-objective target quality, handover frequency)
the values are reported as they come out, with 95% CIs over seeds.

No Table 7 number appears in this file.

Usage:
    python scripts/09_ablation_drl.py                       # defaults below
    python scripts/09_ablation_drl.py --terminals 100 --seeds 0,1,2
    python scripts/09_ablation_drl.py --skip-train          # eval a checkpoint
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gapmoh.constants import RLConfig, Reward, Scenario              # noqa: E402
from gapmoh.metrics_v2 import (concurrency, service_floor_report,    # noqa: E402
                               summarise)
from gapmoh.orbit import build_constellation                        # noqa: E402
from gapmoh.passes import build_pass_table                          # noqa: E402
from gapmoh.policies import SCHEMES                                 # noqa: E402
from gapmoh.rl import DQN, HandoverEnv, build_track                 # noqa: E402
from gapmoh.rl.policy_target import (PolicySelector, load_checkpoint,  # noqa: E402
                                     save_checkpoint)
from gapmoh.sim import simulate_terminal                            # noqa: E402

DT_EXEC_MS = {"rss": 187.3, "dijkstra": 98.5, "cnnlstm": 45.2,
              "madrl_std": 24.7, "grid_only": 12.5, "gapmoh": 3.2}
DT_HO_EXTRA_MS = 2.0

# The manuscript's per-satellite concurrency cap (CN:90, C = 50 concurrent
# users/satellite). The load-balance metric is measured against this.
CAPACITY_PER_SAT = 50

# The DRL layer is trained and evaluated at the scheme's own execution duration,
# so the refinement is measured on the arm it will be reported under.
ARM_KEY = "gapmoh"

# Two-sided 95% t multipliers (df -> t), as in `08_multiseed.py`.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t95(n):
    if n < 2:
        return float("nan")
    return _T95.get(n - 1, 2.0)


def make_terminals(n, cfg, seed):
    """Radians -- see `tests/test_theatre_geography.py` for why this matters."""
    rng = np.random.default_rng(seed)
    lat = np.deg2rad(rng.uniform(cfg.lat_center_deg - cfg.lat_spread_deg,
                                 cfg.lat_center_deg + cfg.lat_spread_deg, n))
    lon = np.deg2rad(rng.uniform(cfg.lon_min_deg, cfg.lon_max_deg, n))
    spd = np.clip(rng.normal(cfg.speed_mean_kn, cfg.speed_sd_kn, n), 2.0, 40.0)
    return lat, lon, spd


def build_world(cfg, n_terms, horizon, seed, log):
    """Constellation + pass tables + tracks. Tables and tracks share a table so
    the training geometry and the evaluation geometry are the same object."""
    con = build_constellation(cfg.n_planes, cfg.sats_per_plane, cfg.altitude_km,
                              np.deg2rad(cfg.inclination_deg), cfg.phase_factor)
    lats, lons, spds = make_terminals(n_terms, cfg, seed)
    tables, tracks = [], []
    t0 = time.time()
    log(f"  building {n_terms} pass tables ...")
    for i in range(n_terms):
        tb = build_pass_table(float(lats[i]), float(lons[i]), float(spds[i]),
                              con, cfg, horizon, dt_samp=2.0)
        tables.append(tb)
        tracks.append(build_track(tb, con, cfg, float(lats[i]), float(lons[i]),
                                  float(spds[i]), horizon_s=horizon))
    n_pass = np.array([t.n_pass for t in tables])
    log(f"  done in {time.time() - t0:.1f} s; passes/terminal "
        f"min {n_pass.min()} mean {n_pass.mean():.0f} max {n_pass.max()}")
    if (n_pass == 0).any():
        raise SystemExit(
            f"ABORT: {int((n_pass == 0).sum())}/{n_terms} terminals see no "
            "satellite above the mask. Geography/units error, not a result.")
    return con, lats, lons, spds, tables, tracks


def train_layer(tracks, cfg, args, log):
    """Train the refinement layer in `HandoverEnv` and return the net."""
    rl = RLConfig()
    rng = np.random.default_rng(1000 + args.seed)
    dt_exec = DT_EXEC_MS[ARM_KEY] * 1e-3
    env = HandoverEnv(tracks, cfg, Reward(), dt_exec_s=dt_exec,
                      dt_ho_s=dt_exec + DT_HO_EXTRA_MS * 1e-3, rng=rng)
    net = DQN(env.s_dim, env.a_dim, hidden=rl.actor_hidden, lr=args.lr,
              gamma=rl.gamma, tau=rl.tau, replay_size=args.replay,
              batch=rl.batch_size, rng=rng)

    log(f"  env s_dim={env.s_dim} a_dim={env.a_dim} K={env.K}; "
        f"training {args.episodes} episodes")
    for ep in range(args.episodes):
        frac = min(1.0, ep / max(1, args.episodes))
        eps = rl.eps_start + frac * (rl.eps_end - rl.eps_start)
        start = int(rng.integers(0, max(1, env.n_step - args.ep_len - 1)))
        obs = env.reset(start_k=start, ep_len=args.ep_len)
        done = False
        losses = []
        while not done:
            greedy = net.act_greedy_batch(obs)
            rand = rng.random(env.M) < eps
            if rand.any():
                greedy = greedy.copy()
                greedy[rand] = rng.integers(0, env.a_dim, int(rand.sum()))
            nxt, reward, done, info = env.step(greedy)
            for i in range(env.M):
                net.store(obs[i], int(greedy[i]), float(reward[i]), nxt[i],
                          float(done))
            obs = nxt
            if net.buf.size >= 256 and env.k % args.update_every == 0:
                for _ in range(args.updates_per_step):
                    l = net.update()
                    if l is not None:
                        losses.append(l)
        if (ep + 1) % max(1, args.episodes // 5) == 0 or ep == args.episodes - 1:
            log(f"    ep {ep + 1:4d}/{args.episodes}  eps={eps:.3f}  "
                f"succ={100.0 * info['success_rate']:6.2f}%  "
                f"ho={info['handovers']:5d}  lost={info['lost']:3d}  "
                f"td={np.mean(losses) if losses else float('nan'):.4f}")
    return net


def run_arm(tables, lats, lons, spds, cfg, con, scheme, horizon, key,
            policy_fn=None):
    """One arm's traces through the shared simulator."""
    dt_exec = DT_EXEC_MS[key] * 1e-3
    return [simulate_terminal(
        tables[i], scheme, cfg, con, float(lats[i]), float(lons[i]),
        float(spds[i]), dt_exec_s=dt_exec,
        dt_ho_s=dt_exec + DT_HO_EXTRA_MS * 1e-3, horizon_s=horizon,
        trace_sat=True, policy_fn=policy_fn) for i in range(len(tables))]


def evaluate(tables, lats, lons, spds, cfg, con, horizon, selector, rain_rate,
             floor):
    """All four arms, summarised on the same metric definitions."""
    arms = {}
    base = SCHEMES[ARM_KEY]

    specs = [
        ("framework", base, None),
        ("refined", base, selector),
    ]
    for name, target_rule in (("long_rem", "longest_remaining"),
                              ("best_qual", "best_quality")):
        from dataclasses import replace
        specs.append((name, replace(base, target=target_rule), None))

    for name, scheme, pf in specs:
        traces = run_arm(tables, lats, lons, spds, cfg, con, scheme, horizon,
                         ARM_KEY, policy_fn=pf)
        s = summarise(traces, cfg, horizon, rain_rate_mm_h=rain_rate,
                      service_floor_elev_deg=floor)
        s["m5_concurrency"] = concurrency(traces, CAPACITY_PER_SAT)
        s["label"] = name
        s["target_rule"] = scheme.target
        arms[name] = s
    return arms


def _ci(vals):
    x = [v for v in vals if v is not None and v == v]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(x) / n
    if n < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in x) / (n - 1)
    return m, t95(n) * (var ** 0.5) / (n ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminals", type=int, default=100)
    ap.add_argument("--horizon", type=float, default=3600.0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--ep-len", type=int, default=1200)
    ap.add_argument("--update-every", type=int, default=4)
    ap.add_argument("--updates-per-step", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--replay", type=int, default=200_000)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results",
                                                      "ablation_drl"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "ablation_log.txt"), "w",
                encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfg = Scenario()
    floor_rep = service_floor_report(cfg, None)
    floor = floor_rep["service_floor_elev_deg_implied_rate"]
    rain_rate = floor_rep["implied_rain_rate_mm_h"]

    log("=" * 96)
    log("DRL REFINEMENT-LAYER ABLATION (non-degenerate metrics)")
    log(f"  M={args.terminals}  horizon={args.horizon:.0f} s  seeds={seeds}"
        f"  episodes={args.episodes}")
    log(f"  rain R={rain_rate:.2f} mm/h -> service floor {floor:.2f} deg "
        f"(mask {cfg.min_elev_deg:.0f} deg)")
    log("=" * 96)

    per_seed = {}
    for k in seeds:
        log("")
        log(f"--- seed {k} ---")
        con, lats, lons, spds, tables, tracks = build_world(
            cfg, args.terminals, args.horizon, k, log)

        ckpt = os.path.join(args.out_dir, f"drl_seed{k}.npz")
        args.seed = k
        if args.skip_train and os.path.exists(ckpt):
            log(f"  using checkpoint {ckpt}")
        else:
            t0 = time.time()
            net = train_layer(tracks, cfg, args, log)
            save_checkpoint(ckpt, net, meta={
                "scheme": ARM_KEY, "seed": k, "terminals": args.terminals,
                "horizon_s": args.horizon, "episodes": args.episodes,
                "dt_exec_ms": DT_EXEC_MS[ARM_KEY],
                "s_dim": int(net.q.dims[0]), "a_dim": int(net.q.dims[-1]),
            })
            log(f"  trained in {time.time() - t0:.1f} s -> {ckpt}")

        selector, _ = load_checkpoint(ckpt, cfg)
        arms = evaluate(tables, lats, lons, spds, cfg, con, args.horizon,
                        selector, rain_rate, floor)
        per_seed[k] = {"arms": arms, "selector": selector.stats()}
        log("  selector: " + json.dumps(selector.stats(), ensure_ascii=False))

    # ---- aggregate --------------------------------------------------------
    log("")
    log("=" * 96)
    log(f"AGGREGATE over {len(seeds)} seed(s)")
    log("=" * 96)

    def field(arm, path):
        """Per-seed values of one scalar, or [] if the path is absent."""
        out = []
        for k in seeds:
            n = per_seed[k]["arms"][arm]
            for step in path:
                n = n.get(step) if isinstance(n, dict) else None
                if n is None:
                    break
            out.append(n)
        return out

    rows = ["framework", "refined", "long_rem", "best_qual"]
    metrics = [
        ("M1 target elevation [deg]", ["m1_target_quality",
                                       "target_elev_mean"], 2),
        ("M1 target SNR [dB]", ["m1_target_quality", "target_snr_mean"], 2),
        ("M3 unnecessary by elev [%]", ["m3_unnecessary",
                                        "share_by_elevation"], 1),
        ("M4 frequency [1/h]", ["m4_frequency_per_h"], 2),
        (f"M5 peak concurrent per sat [cap {CAPACITY_PER_SAT}]",
         ["m5_concurrency", "peak_concurrent"], 2),
        ("M5 load held by busiest sat [share]",
         ["m5_concurrency", "top1_share"], 4),
    ]
    for title, path, nd in metrics:
        log("")
        log(f"{title}")
        log(f"{'arm':<14}{'mean':>12}{'95% CI half-width':>20}")
        for arm in rows:
            vals = field(arm, path)
            if path[-1] == "share_by_elevation":
                vals = [None if v is None else 100.0 * v for v in vals]
            m, h = _ci(vals)
            log(f"{arm:<14}{m:>{12}.{nd}f}{h:>{20}.{nd}f}")

    # ---- headroom ---------------------------------------------------------
    log("")
    log("-" * 96)
    log("HEADROOM")
    log("-" * 96)
    m1 = {a: _ci(field(a, ["m1_target_quality", "target_elev_mean"]))[0]
          for a in rows}
    best_ref = max(m1["long_rem"], m1["best_qual"])
    log(f"  M1: framework={m1['framework']:.2f} deg, refined={m1['refined']:.2f}"
        f" deg, best scripted reference={best_ref:.2f} deg")
    if m1["framework"] >= best_ref - 1e-9:
        log("  -> The framework layer's rule (`highest_elev`) is the ARGMAX of")
        log("     the candidate set, so on target elevation it sits AT the")
        log("     metric's optimum. A tie here is a property of the metric, not")
        log("     a training failure; the refinement layer has NO room on M1 and")
        log("     its contribution cannot be claimed from this row.")
    else:
        log("  -> The refinement layer has room on M1; report the gap.")

    m3 = {a: _ci(field(a, ["m3_unnecessary", "share_by_elevation"]))[0]
          for a in rows}
    log("")
    log(f"  M3: framework={100.0 * m3['framework']:.2f}% unnecessary")
    if m3["framework"] <= 1e-9:
        log("  -> 0% IS the floor of this metric: no policy can abandon a")
        log("     still-serviceable link fewer than zero times. No room here")
        log("     either; M3 cannot carry the refinement layer's contribution.")

    log("")
    log(f"  M5: THE ONLY METRIC WITH HEADROOM, if any is to be found.")
    log("     `highest_elev` selects each terminal's target INDEPENDENTLY, so")
    log("     terminals in the same cell all name the same satellite and the")
    log("     load concentrates. A load-aware policy trades some target")
    log("     elevation for a flatter distribution -- which is the trade the")
    log("     manuscript's own contribution (2) names ('优化目标选择与负载均衡').")
    log("     Compare `top1_share` and `peak_concurrent` across the arms above:")
    log("     if they do not separate here, the refinement layer has no room on")
    log("     M5 either and contribution (3) cannot be claimed from it.")

    with open(os.path.join(args.out_dir, "ablation_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"spec": {"terminals": args.terminals,
                            "horizon_s": args.horizon, "seeds": seeds,
                            "episodes": args.episodes},
                   "rain": floor_rep, "per_seed": {
                       str(k): per_seed[k] for k in seeds}},
                  f, indent=2, ensure_ascii=False, default=float)
    log("")
    log(f"wrote {os.path.join(args.out_dir, 'ablation_summary.json')}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
