# -*- coding: utf-8 -*-
"""MADRL-Standard training run producing the per-episode learning curves.

WHAT THIS IS FOR. The manuscript reports, for MADRL-Standard, that the
100-episode moving-average success rate of 10 seeds converges and stabilises at
92.4% +/- 0.6%. No training code, log or checkpoint for that run exists anywhere
under the author's directory, so this script trains the scheme from the
specification (constants.RLConfig + Scenario) and emits the curve that the
manuscript's supplementary material says should exist.

IT DOES NOT TRY TO REPRODUCE 92.4%. Every hyper-parameter comes from
`constants.py`, which carries [SPEC]/[CHOICE]/[DEVIATION] tags per field. The
comparison against the manuscript's number happens later, in a separate script,
so that no number can leak backwards into the training setup.

Honest expectations, stated before the run: the timing criterion
`dt_exec < t_exit - t_init` is a property of WHEN the trigger fires, not of
which satellite is chosen, so a policy that hands over with any lead time scores
near 100%. If that is what comes out, that is the result.

Usage:
    python scripts/02_train_small.py --seeds 3 --episodes 60
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gapmoh.constants import Reward, RLConfig, Scenario             # noqa: E402
from gapmoh.orbit import build_constellation                        # noqa: E402
from gapmoh.passes import build_pass_table                          # noqa: E402
from gapmoh.rl import DQN, HandoverEnv, build_track                 # noqa: E402

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


def build_world(cfg, n_terms, horizon, seed):
    """Constellation + pass tables + tracks. Done once, reused by every seed."""
    con = build_constellation(cfg.n_planes, cfg.sats_per_plane, cfg.altitude_km,
                              np.deg2rad(cfg.inclination_deg), cfg.phase_factor)
    lats, lons, spds = make_terminals(n_terms, cfg, seed)
    tables, tracks = [], []
    for i in range(n_terms):
        tb = build_pass_table(float(lats[i]), float(lons[i]), float(spds[i]),
                              con, cfg, horizon, dt_samp=2.0)
        tables.append(tb)
        tracks.append(build_track(tb, con, cfg, float(lats[i]), float(lons[i]),
                                  float(spds[i]), horizon_s=horizon))
    return con, tracks


def epsilon_at(ep, rl, total_episodes):
    """Linear epsilon decay over the run.

    [DEVIATION] The spec (RLConfig.eps_decay_episodes = 5000) decays epsilon
    over 5000 episodes. This run is far shorter, so decaying on the spec's
    schedule would leave epsilon pinned at ~1.0 -- pure random actions, nothing
    learned. The schedule is therefore rescaled to the episode budget actually
    run, and eps_start / eps_end are kept at their specified values.
    """
    frac = min(1.0, ep / max(1, total_episodes))
    return rl.eps_start + frac * (rl.eps_end - rl.eps_start)


def run_seed(tracks, cfg, rw, rl, seed, args, log):
    rng = np.random.default_rng(1000 + seed)
    env = HandoverEnv(tracks, cfg, rw, dt_exec_s=args.dt_exec_ms * 1e-3,
                      dt_ho_s=(args.dt_exec_ms + DT_HO_EXTRA_MS) * 1e-3, rng=rng)
    agent = DQN(env.s_dim, env.a_dim, hidden=rl.actor_hidden,
                lr=args.lr, gamma=rl.gamma, tau=rl.tau,
                replay_size=args.replay, batch=rl.batch_size, rng=rng)

    hist = []
    for ep in range(args.episodes):
        start = int(rng.integers(0, max(1, env.n_step - args.ep_len - 1)))
        obs = env.reset(start_k=start, ep_len=args.ep_len)
        eps = epsilon_at(ep, rl, args.episodes)
        done = False
        losses = []
        while not done:
            # one batched forward for all M terminals, then epsilon-greedy
            greedy = agent.act_greedy_batch(obs)
            rand = rng.random(env.M) < eps
            if rand.any():
                greedy = greedy.copy()
                greedy[rand] = rng.integers(0, env.a_dim, int(rand.sum()))
            nxt, reward, done, info = env.step(greedy)
            for i in range(env.M):
                agent.store(obs[i], int(greedy[i]), float(reward[i]), nxt[i],
                            float(done))
            obs = nxt
            if agent.buf.size >= 256 and env.k % args.update_every == 0:
                for _ in range(args.updates_per_step):
                    l = agent.update()
                    if l is not None:
                        losses.append(l)
        hist.append({
            "episode": ep, "epsilon": eps,
            "success_pct": 100.0 * info["success_rate"],
            "reward_per_terminal": float(env.ep_reward.mean()),
            "handovers": info["handovers"], "lost": info["lost"],
            "td_loss": float(np.mean(losses)) if losses else None,
        })
        if (ep + 1) % max(1, args.episodes // 10) == 0 or ep == args.episodes - 1:
            log(f"  seed {seed} ep {ep + 1:4d}/{args.episodes}  eps={eps:.3f}  "
                f"succ={hist[-1]['success_pct']:6.2f}%  "
                f"ho={info['handovers']:5d}  lost={info['lost']:3d}  "
                f"R={hist[-1]['reward_per_terminal']:8.2f}")
    return hist


def smooth(x, w=100):
    x = np.asarray(x, dtype=float)
    if x.size < w:
        w = max(1, x.size)
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def moving_average_stabilised(x, w=100):
    """Last value of the w-episode moving average -- the manuscript's statistic."""
    s = smooth(x, w)
    return float(s[-1]) if s.size else float("nan")


def run_baselines(tracks, cfg, rw, args, log):
    """Scripted policies on the same world, evaluated on the same criterion.

    These exist to show WHERE the ceiling is. If a random policy already scores
    what the trained policy scores, then the learning is not what produces the
    metric, and no amount of training can manufacture the spread the manuscript
    reports.
    """
    rng = np.random.default_rng(7)
    M = len(tracks)
    out = {}
    policies = {
        "RANDOM (uniform over 7 actions)": lambda k: rng.integers(0, 7, M),
        "HOLD always (never hand over)": lambda k: np.full(M, 5),
        "MAX-REMAINING every step": lambda k: np.full(M, 6),
    }

    for name, pol in policies.items():
        succ, hos, losts = [], [], []
        for _ in range(args.baseline_episodes):
            env = HandoverEnv(tracks, cfg, rw, dt_exec_s=args.dt_exec_ms * 1e-3,
                              dt_ho_s=(args.dt_exec_ms + DT_HO_EXTRA_MS) * 1e-3,
                              rng=np.random.default_rng(11))
            start = int(rng.integers(0, max(1, env.n_step - args.ep_len - 1)))
            env.reset(start_k=start, ep_len=args.ep_len)
            done = False
            k = 0
            while not done:
                _, _, done, info = env.step(pol(k))
                k += 1
            succ.append(100.0 * info["success_rate"])
            hos.append(info["handovers"])
            losts.append(info["lost"])
        out[name] = {"success_pct_mean": float(np.nanmean(succ)),
                     "handovers_mean": float(np.mean(hos)),
                     "lost_mean": float(np.mean(losts))}
        log(f"  baseline {name:<32} success="
            f"{out[name]['success_pct_mean']:6.2f}%  "
            f"handovers/ep={out[name]['handovers_mean']:7.1f}  "
            f"lost/ep={out[name]['lost_mean']:6.1f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminals", type=int, default=20)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ep-len", type=int, default=1500,
                    help="steps per episode (dt_dec=0.1 s -> 1500 = 150 s)")
    ap.add_argument("--dt-exec-ms", type=float,
                    default=DT_EXEC_MS["madrl_std"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--replay", type=int, default=100_000)
    ap.add_argument("--update-every", type=int, default=10)
    ap.add_argument("--updates-per-step", type=int, default=1)
    ap.add_argument("--baseline-episodes", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "madrl_standard")
    os.makedirs(out, exist_ok=True)

    logf = open(os.path.join(out, "train_log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfg, rw, rl = Scenario(), Reward(), RLConfig()
    log("=" * 78)
    log("MADRL-Standard training (numpy backend)")
    log(f"  M={args.terminals}  horizon={args.horizon:.0f}s  "
        f"episode={args.ep_len * cfg.dt_dec_s:.0f}s  episodes={args.episodes}  "
        f"seeds={args.seeds}")
    log(f"  dt_exec={args.dt_exec_ms:.1f} ms  lr={args.lr:g}  "
        f"replay={args.replay}  update_every={args.update_every}")
    log("=" * 78)

    t0 = time.time()
    log("building constellation, pass tables and tracks (once)...")
    con, tracks = build_world(cfg, args.terminals, args.horizon, seed=0)
    log(f"world ready in {time.time() - t0:.1f} s")

    log("--- scripted baselines (same world, same criterion) ---")
    baselines = run_baselines(tracks, cfg, rw, args, log)
    log("")

    per_seed = {}
    for s in range(args.seeds):
        t1 = time.time()
        log(f"--- seed {s} ---")
        hist = run_seed(tracks, cfg, rw, rl, s, args, log)
        per_seed[str(s)] = hist
        log(f"  seed {s} done in {time.time() - t1:.1f} s  "
            f"final moving-avg succ = "
            f"{moving_average_stabilised([h['success_pct'] for h in hist]):.2f}%")

    # --------------------------------------------------- aggregate + export
    succ = np.array([[h["success_pct"] for h in per_seed[str(s)]]
                     for s in range(args.seeds)], dtype=float)
    rew = np.array([[h["reward_per_terminal"] for h in per_seed[str(s)]]
                    for s in range(args.seeds)], dtype=float)
    mean_curve, sd_curve = succ.mean(axis=0), succ.std(axis=0)

    finals = [moving_average_stabilised(succ[s]) for s in range(args.seeds)]
    summary = {
        "scheme": "MADRL-Standard",
        "baselines": baselines,
        "spec": {"M": args.terminals, "horizon_s": args.horizon,
                 "episode_steps": args.ep_len, "episodes": args.episodes,
                 "seeds": args.seeds, "dt_exec_ms": args.dt_exec_ms,
                 "lr": args.lr, "replay": args.replay,
                 "update_every": args.update_every,
                 "backend": "numpy"},
        "per_seed_final_moving_avg_success_pct": finals,
        "mean_final_moving_avg_success_pct": float(np.mean(finals)),
        "sd_final_moving_avg_success_pct": float(np.std(finals, ddof=1))
        if len(finals) > 1 else 0.0,
        "mean_curve_pct": mean_curve.tolist(),
        "sd_curve_pct": sd_curve.tolist(),
        "mean_reward_curve": rew.mean(axis=0).tolist(),
        "episodes": [h["episode"] for h in per_seed["0"]],
        "per_seed": per_seed,
    }
    with open(os.path.join(out, "training_curves.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log("=" * 78)
    log(f"FINAL: {args.seeds}-seed mean moving-average success rate = "
        f"{summary['mean_final_moving_avg_success_pct']:.2f}% "
        f"+/- {summary['sd_final_moving_avg_success_pct']:.2f}%")
    log(f"       per-seed: {['%.2f' % v for v in finals]}")
    log(f"       manuscript reports 92.4% +/- 0.6% for this scheme")
    log("=" * 78)

    write_plot(out, summary, args)
    log(f"wrote {os.path.join(out, 'training_curves.json')}")
    log(f"wrote {os.path.join(out, 'training_curves.png')}")
    logf.close()
    return summary


def write_plot(out, summary, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                   # pragma: no cover
        print("matplotlib unavailable, skipping plot:", e)
        return
    ep = np.array(summary["episodes"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    for s, hist in summary["per_seed"].items():
        y = [h["success_pct"] for h in hist]
        ax[0].plot(ep, y, alpha=0.30, lw=1)
    m = np.array(summary["mean_curve_pct"])
    ax[0].plot(ep, m, "k-", lw=2, label="mean over seeds")
    ax[0].axhline(92.4, color="crimson", ls="--", lw=1.5,
                  label="manuscript: 92.4%")
    ax[0].set_xlabel("episode")
    ax[0].set_ylabel("timing-feasibility success rate (%)")
    ax[0].set_title("(a) Success rate")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for s, hist in summary["per_seed"].items():
        ax[1].plot(ep, [h["reward_per_terminal"] for h in hist], alpha=0.35, lw=1)
    ax[1].plot(ep, summary["mean_reward_curve"], "k-", lw=2)
    ax[1].set_xlabel("episode")
    ax[1].set_ylabel("episode reward per terminal")
    ax[1].set_title("(b) Reward")
    ax[1].grid(alpha=0.3)

    for s, hist in summary["per_seed"].items():
        y = [h["td_loss"] if h["td_loss"] is not None else np.nan for h in hist]
        ax[2].plot(ep, y, alpha=0.35, lw=1)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("episode")
    ax[2].set_ylabel("mean TD loss")
    ax[2].set_title("(c) Critic loss")
    ax[2].grid(alpha=0.3)

    fig.suptitle("MADRL-Standard learning curves (numpy backend, "
                 f"M={args.terminals}, {args.seeds} seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
