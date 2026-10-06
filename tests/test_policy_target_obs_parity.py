# -*- coding: utf-8 -*-
"""`PolicySelector._obs` must reproduce `HandoverEnv._obs` exactly.

WHY THIS TEST EXISTS. `PolicySelector` rebuilds the training observation from a
candidate array, because `sim.simulate_terminal` hands it candidates where
`HandoverEnv` has a `TerminalTrack`. Two hand-written builders of the same
vector is a silent-failure hazard: get a slot order, a normalisation constant or
the candidate ordering wrong and the net is fed a vector it never saw. There is
no exception and no obviously wrong number -- the policy simply returns worse
targets than it learned to, and the ablation under-reports the layer's
contribution with nothing to point at.

So the two are compared on a shared track. If `HandoverEnv._obs` changes, this
test fails and `policy_target._obs` has to change with it.

The comparison is tolerant to ~1e-6 because the track stores elevations as
float32 while the selector receives float64 geometry; the deployed path has the
same small difference.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gapmoh.constants import Reward, Scenario                        # noqa: E402
from gapmoh.orbit import build_constellation                         # noqa: E402
from gapmoh.passes import build_pass_table                           # noqa: E402
from gapmoh.rl import DQN, HandoverEnv, build_track                  # noqa: E402
from gapmoh.rl.policy_target import PolicySelector                   # noqa: E402


def _fixture(horizon=600.0):
    cfg = Scenario()
    con = build_constellation(cfg.n_planes, cfg.sats_per_plane, cfg.altitude_km,
                              np.deg2rad(cfg.inclination_deg), cfg.phase_factor)
    lat, lon, spd = np.deg2rad(20.0), np.deg2rad(130.0), 15.0
    tb = build_pass_table(lat, lon, spd, con, cfg, horizon, dt_samp=2.0)
    track = build_track(tb, con, cfg, lat, lon, spd, horizon_s=horizon)
    env = HandoverEnv([track], cfg, Reward(), dt_exec_s=3.2e-3,
                      dt_ho_s=5.2e-3, rng=np.random.default_rng(0))
    return cfg, track, env


def test_selector_obs_matches_env_obs():
    cfg, track, env = _fixture()
    K = env.K

    # A step where the full candidate set exists, so the comparison exercises
    # every slot rather than just the padding.
    k = int(np.argmax([np.count_nonzero(track.top_idx[i] >= 0)
                       for i in range(track.n_step)]))
    env.reset(start_k=k)
    env.serving_pid = [int(track.top_idx[k, 0])]
    serving = env.serving_pid[0]
    want = env._obs(k)[0]

    cols = env._cand_slots(track, k, serving)
    cand = track.top_idx[k, cols].astype(np.int64)
    ce = track.top_elev[k, cols].astype(np.float64)
    crem = track.top_rem[k, cols].astype(np.float64)
    ce_rate = track.top_rate[k, cols].astype(np.float64)
    s_elev = track.elev_of(serving, k)
    s_rate = track.rate_of(serving, k)
    s_rem = float(track.t_exit[serving]) - k * track.dt_dec

    net = DQN(env.s_dim, env.a_dim, hidden=(8, 8), rng=np.random.default_rng(0))
    sel = PolicySelector(net, cfg, K=K)
    ctx = {"lat_rad": track.lat_rad, "lon_rad": track.lon_rad,
           "speed_kn": track.speed_kn, "horizon_s": track.horizon_s,
           "t": k * track.dt_dec, "n_ho": 0}

    got, order = sel._obs(cand, ce, crem, ce_rate, s_elev, s_rate, s_rem, ctx)

    assert got.size == want.size, (
        f"observation width drifted: selector built {got.size}, env wants "
        f"{want.size}")
    assert np.allclose(got, want, atol=1e-6), (
        "PolicySelector._obs no longer matches HandoverEnv._obs.\n"
        f"  max abs diff = {np.max(np.abs(got - want)):.3e}\n"
        f"  selector = {np.round(got, 4)}\n"
        f"  env      = {np.round(want, 4)}")
    # Candidates must come out in descending elevation: the action index means
    # "the a-th highest-elevation candidate", not "the a-th as listed".
    assert np.all(np.diff(ce[order]) <= 1e-12), \
        "candidate ordering is not descending by elevation"


def test_selector_layout_guard_fires_on_a_mismatched_net():
    """A net built for a different K must fail loudly, not silently."""
    cfg, track, env = _fixture()
    k = int(np.argmax([np.count_nonzero(track.top_idx[i] >= 0)
                       for i in range(track.n_step)]))
    env.reset(start_k=k)
    serving = env.serving_pid[0]
    cols = env._cand_slots(track, k, serving)

    wrong = DQN(env.s_dim + 7, env.a_dim, hidden=(8, 8),
                rng=np.random.default_rng(0))
    sel = PolicySelector(wrong, cfg, K=env.K)
    ctx = {"lat_rad": track.lat_rad, "lon_rad": track.lon_rad,
           "speed_kn": track.speed_kn, "horizon_s": track.horizon_s,
           "t": k * track.dt_dec, "n_ho": 0}
    args = (track.top_idx[k, cols].astype(np.int64),
            track.top_elev[k, cols].astype(np.float64),
            track.top_rem[k, cols].astype(np.float64),
            track.top_rate[k, cols].astype(np.float64),
            track.elev_of(serving, k), track.rate_of(serving, k),
            float(track.t_exit[serving]) - k * track.dt_dec, ctx)

    try:
        sel._obs(*args)
    except ValueError as e:
        assert "observation layout drift" in str(e)
    else:
        raise AssertionError(
            "a net with a mismatched input dimension was accepted; the layout "
            "guard in PolicySelector._obs is not firing")
