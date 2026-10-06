# -*- coding: utf-8 -*-
"""Discrete-action multi-terminal handover environment (numpy, vectorized).

WHY NOT torch / OpenAI-Gym: the manuscript describes a custom simulator with
PyTorch 2.0 + CUDA (EN:1126 region). Neither torch nor a GPU exists here, so the
backend is numpy and the environment is hand-rolled. This is a backend
substitution, not a change of model.

THE LOAD-BEARING DESIGN CHOICE. The geometry is EPISODE-INDEPENDENT: the
constellation, the terminal tracks and the pass table do not depend on the
policy, and only the policy changes between episodes or seeds. So the expensive
part -- which satellites are visible, at what elevation, and for how long -- is
computed ONCE per terminal in `build_track` and reused by every episode of every
seed. The rollout itself is then pure array indexing. Without this, one episode
would cost minutes and the 10-seed run the manuscript promises would be
unreachable here.

Model simplifications, all recorded rather than hidden:

  [DEVIATION] The serving satellite is always one of the tracked top candidates.
      The paper's agent selects among a K-candidate set, so this is the same
      restriction the policy operates under; what it loses is the ability to
      serve a satellite that has dropped out of the top-(K+2) by elevation.
  [DEVIATION] Doppler is represented by the elevation rate. The elevation rate
      is monotone-related to the range-rate Doppler over a pass and carries the
      same sign information; the exact Doppler needs satellite range rate, which
      the observation is not used to decide the headline metric.
  [CHOICE]    Candidate set = the K highest-elevation visible satellites above
      the mask (the paper says "top-K by score" without fixing the score).
  [CHOICE]    Action K is HOLD; action K+1 hands over to the candidate with the
      longest remaining visibility. The paper fixes |A| = K+2 without naming the
      two extras.
  [CHOICE]    Observation layout (43 = 6K+13): 5*K candidate features
      (elevation, elevation rate, remaining visibility, SNR, load), 5 serving
      features, 2 serving scalars (remaining, elevation), 5 terminal scalars,
      6 validity mask. The paper gives the dimension, not the split.

No Table 7 number appears in this file.
"""

from dataclasses import dataclass

import numpy as np

from ..channel import snr_from_elev_db

__all__ = ["TerminalTrack", "build_track", "HandoverEnv"]

_EPS = 1e-9


@dataclass
class TerminalTrack:
    top_idx: np.ndarray    # (n_step, K+2) int32, pass ids; -1 = no candidate
    top_elev: np.ndarray   # (n_step, K+2) float32, radians
    top_rem: np.ndarray    # (n_step, K+2) float32, seconds of visibility left
    top_rate: np.ndarray   # (n_step, K+2) float32, rad/s
    t_exit: np.ndarray     # (n_pass,) float64, exit time per pass id
    sat: np.ndarray        # (n_pass,) satellite index of each pass id
    lat_rad: float
    lon_rad: float
    speed_kn: float
    horizon_s: float
    dt_dec: float
    ts: np.ndarray = None      # (n_step,) decision-grid times
    terms: np.ndarray = None   # (n_step, 3) terminal ECEF
    con: object = None         # constellation, for on-demand geometry
    min_elev_rad: float = 0.26179938779914946   # 15 deg

    def __post_init__(self):
        # Lazy per-pass elevation cache. The SERVING satellite must be
        # readable at its true elevation even when it has dropped out of the
        # elevation-ranked candidate columns; without this, a perfectly healthy
        # serving link gets scored as a link loss the moment K+2 better-looking
        # satellites appear, and the environment hands out free emergency
        # handovers. Keyed by pass id, so the cache is shared by every episode.
        self._elev_cache = {}

    @property
    def n_step(self):
        return self.top_idx.shape[0]

    def elev_of(self, pid, k):
        """Elevation (rad) of PASS `pid` at step k, computed on demand.

        `pid` indexes the pass table. `sat_elev_series` takes a SATELLITE index,
        so the pass -> satellite mapping has to be applied here; passing the
        pass id straight through evaluates an unrelated satellite and makes
        every healthy serving link look dead.
        """
        if pid < 0 or k >= self.n_step:
            return 0.0
        arr = self._elev_cache.get(pid)
        if arr is None:
            from ..orbit import sat_elev_series
            arr = sat_elev_series(self.ts, self.terms, self.con,
                                  np.asarray([self.sat[pid]]))[:, 0]
            self._elev_cache[pid] = arr
        return float(arr[k])

    def rate_of(self, pid, k, ch=40):
        """Elevation rate (rad/s), finite-differenced from the cached series."""
        k2 = min(k + ch, self.n_step - 1)
        if k2 <= k:
            return 0.0
        return (self.elev_of(pid, k2) - self.elev_of(pid, k)) / (
            (k2 - k) * self.dt_dec)

    def is_live(self, pid, k):
        return (pid >= 0 and k < self.n_step
                and k * self.dt_dec < self.t_exit[pid]
                and self.elev_of(pid, k) > self.min_elev_rad)


def build_track(table, con, cfg, lat, lon0, speed_kn, horizon_s=None,
                dt_dec=None, K=None):
    """Precompute one terminal's top-(K+2) candidates on the decision grid.

    `PassTable` carries only the pass geometry, not the terminal's own state
    (lat/lon0/speed are not fields on it), so the caller passes them in.
    """
    from ..orbit import sat_elev_series, terminal_series

    horizon = float(table.horizon_s if horizon_s is None else horizon_s)
    dt = float(cfg.dt_dec_s if dt_dec is None else dt_dec)
    K = cfg.n_candidates if K is None else K
    n_step = int(horizon / dt) + 1
    ts = np.arange(n_step) * dt

    if (getattr(table, "_idx_dt", None) != dt
            or table._offsets is None or table._offsets.size != n_step + 1):
        table.build_active_index(n_step, dt)
        table._idx_dt = dt

    terms = terminal_series(ts, float(lat), float(lon0), float(speed_kn))
    min_elev = np.deg2rad(cfg.min_elev_deg)
    ncol = K + 2

    top_idx = np.full((n_step, ncol), -1, dtype=np.int32)
    top_elev = np.zeros((n_step, ncol), dtype=np.float32)
    top_rem = np.zeros((n_step, ncol), dtype=np.float32)
    top_rate = np.zeros((n_step, ncol), dtype=np.float32)

    ch = max(1, int(round(4.0 / dt)))   # 4 s window for the elevation rate
    for k in range(n_step):
        live = table.active_at(k)
        if live.size == 0:
            continue
        sats = table.sat[live]
        e = sat_elev_series(ts[k:k + 1], terms[k:k + 1], con, sats)[0]
        ok = e > min_elev
        if not ok.any():
            continue
        cand, ce = live[ok], e[ok]
        order = np.argsort(ce)[::-1][:ncol]
        cand, ce = cand[order], ce[order]

        k2 = min(k + ch, n_step - 1)
        if k2 > k:
            e2 = sat_elev_series(ts[k2:k2 + 1], terms[k2:k2 + 1], con,
                                 table.sat[cand])[0]
            rate = (e2 - ce) / max(ts[k2] - ts[k], _EPS)
        else:
            rate = np.zeros_like(ce)

        m = cand.size
        top_idx[k, :m] = cand
        top_elev[k, :m] = ce
        top_rem[k, :m] = table.t_exit[cand] - ts[k]
        top_rate[k, :m] = rate

    return TerminalTrack(top_idx=top_idx, top_elev=top_elev, top_rem=top_rem,
                         top_rate=top_rate, t_exit=np.asarray(table.t_exit,
                                                              dtype=np.float64),
                         sat=np.asarray(table.sat, dtype=np.int64),
                         lat_rad=float(lat), lon_rad=float(lon0),
                         speed_kn=float(speed_kn), horizon_s=horizon,
                         dt_dec=dt, ts=ts, terms=terms, con=con,
                         min_elev_rad=float(min_elev))


class HandoverEnv:
    """M independent terminals, one shared discrete-action policy."""

    def __init__(self, tracks, cfg, rw, dt_exec_s, dt_ho_s, rng=None,
                 snr_min_db=None, dt_exec_variability=None):
        self.tracks = tracks
        self.M = len(tracks)
        self.cfg = cfg
        self.rw = rw
        self.dt_exec = float(dt_exec_s)
        self.dt_ho = float(dt_ho_s)
        self.dt = float(tracks[0].dt_dec)
        self.n_step = min(t.n_step for t in tracks)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.K = tracks[0].top_idx.shape[1] - 2
        self.a_dim = self.K + 2
        self.s_dim = 6 * self.K + 13
        self.snr_min = cfg.snr_min_db if snr_min_db is None else snr_min_db
        # [CHOICE] per-handover execution jitter (lognormal, mean = dt_exec,
        # sigma chosen so the 95th percentile is ~1.6x the mean). The paper
        # gives a single dt_exec per scheme; jitter is what lets a *timing*
        # criterion produce any failures at all once targets are chosen well.
        self.dt_exec_sigma = (0.30 if dt_exec_variability is None
                              else float(dt_exec_variability))
        self.reset()

    # ------------------------------------------------------------- helpers
    def _slot_of(self, tr, k, pid):
        """Column index at which `pid` currently sits, or -1.

        Columns are re-sorted by elevation every step, so a PASS ID is the only
        stable identity. Tracking the serving link by column silently swaps the
        serving satellite each step -- which is what made the first version of
        this env report 0% success and zero link losses at once.
        """
        if pid < 0 or k >= tr.n_step:
            return -1
        hit = np.flatnonzero(tr.top_idx[k] == pid)
        return int(hit[0]) if hit.size else -1

    def _cand_slots(self, tr, k, serving_pid):
        """Up to K candidate columns, excluding the serving satellite."""
        cols = [c for c in range(tr.top_idx.shape[1])
                if tr.top_idx[k, c] >= 0 and tr.top_idx[k, c] != serving_pid]
        return cols[:self.K]

    def _serving_elev(self, tr, k, pid):
        """True elevation of the serving pass -- NOT its rank in the columns."""
        return tr.elev_of(pid, k) if tr.is_live(pid, k) else 0.0

    # ------------------------------------------------------------------ obs
    def _obs(self, k):
        K = self.K
        out = np.zeros((self.M, self.s_dim), dtype=np.float64)
        for i, tr in enumerate(self.tracks):
            if k >= tr.n_step:
                continue
            s_pid = self.serving_pid[i]
            cands = self._cand_slots(tr, k, s_pid)
            feats = np.zeros(5 * K)
            for j, c in enumerate(cands):
                e = float(tr.top_elev[k, c])
                feats[5 * j:5 * j + 5] = [e, float(tr.top_rate[k, c]),
                                          np.clip(tr.top_rem[k, c], 0, 600) / 600.0,
                                          float(snr_from_elev_db(
                                              np.array([e]), self.cfg)[0]) / 40.0,
                                          0.0]
            live = tr.is_live(s_pid, k)
            s_elev = tr.elev_of(s_pid, k) if live else 0.0
            s_rem = (max(0.0, float(tr.t_exit[s_pid]) - k * self.dt)
                     if live else 0.0)
            s_snr = (float(snr_from_elev_db(np.array([s_elev]), self.cfg)[0])
                     if live else 0.0)
            s_rate = tr.rate_of(s_pid, k) if live else 0.0
            serving = np.array([s_elev, s_rate, np.clip(s_rem, 0, 600) / 600.0,
                                s_snr / 40.0, 0.0])
            misc = np.array([np.clip(s_rem, 0, 600) / 600.0, s_elev])
            term = np.array([tr.lat_rad, tr.lon_rad, tr.speed_kn / 40.0,
                             (k * self.dt) / tr.horizon_s,
                             min(1.0, self.n_handover[i] / 20.0)])
            mask = np.zeros(K + 1)
            mask[:len(cands)] = 1.0
            mask[K] = 1.0 if live else 0.0
            out[i] = np.concatenate([feats, serving, misc, term, mask])
        return out

    # ---------------------------------------------------------------- reset
    def reset(self, start_k=0, ep_len=None):
        """Reset. `start_k` randomises the phase of the pass cycle.

        Without a randomised start every episode replays the identical
        trajectory (the geometry is deterministic), so the replay buffer would
        see one path and the "learning curve" would measure epsilon-decay alone.
        """
        M = self.M
        self.start_k = int(start_k)
        self.k = self.start_k
        self.ep_len = int(ep_len) if ep_len else (self.n_step - 1 - self.start_k)
        # the serving satellite is column 0 (highest elevation) at the start
        self.serving_pid = [
            int(t.top_idx[self.start_k, 0])
            if self.start_k < t.n_step and t.top_idx[self.start_k, 0] >= 0 else -1
            for t in self.tracks]
        self.n_handover = np.zeros(M, dtype=np.int64)
        self.n_ok = np.zeros(M, dtype=np.int64)
        self.n_fail = np.zeros(M, dtype=np.int64)
        self.n_lost = np.zeros(M, dtype=np.int64)
        self.pending = [None] * M     # (t_init, serving_pid, target_pid, exec_t)
        self.ep_reward = np.zeros(M, dtype=np.float64)
        return self._obs(self.k)

    # ----------------------------------------------------------------- step
    def step(self, actions):
        """Advance one decision interval. Returns obs, reward, done, info."""
        k = self.k
        t = k * self.dt
        rew = np.zeros(self.M, dtype=np.float64)

        for i, tr in enumerate(self.tracks):
            if k >= tr.n_step:
                continue
            s_pid = self.serving_pid[i]

            # --- 0. resolve any handover whose execution time has arrived ---
            pend = self.pending[i]
            if pend is not None and t + _EPS >= pend[3]:
                t_init, s_pid0, tgt_pid, t_exec = pend
                self.pending[i] = None
                tgt_slot = self._slot_of(tr, k, tgt_pid)
                alive = tr.is_live(tgt_pid, k)
                # the criterion is literal: the transfer must COMPLETE before
                # the serving link dies -- dt_exec < t_exit - t_init
                budget_ok = (s_pid0 >= 0) and (t_exec < tr.t_exit[s_pid0])
                ok = bool(budget_ok and alive)
                self.n_handover[i] += 1
                if ok:
                    self.n_ok[i] += 1
                    self.serving_pid[i] = tgt_pid
                    rew[i] += self.rw.alpha_quality * np.tanh(
                        max(tr.elev_of(tgt_pid, k), 0.0) * 3.0)
                else:
                    self.n_fail[i] += 1
                    rew[i] += self.rw.terminal_penalty
                    self.serving_pid[i] = -1
                s_pid = self.serving_pid[i]

            # --- 1. link lost? -------------------------------------------
            # Liveness is judged on the serving pass's own geometry, not on
            # whether it still occupies a candidate column.
            if not tr.is_live(s_pid, k):
                if self.pending[i] is None:
                    self.n_lost[i] += 1
                    rew[i] += self.rw.terminal_penalty
                # emergency re-acquire: best remaining candidate
                cands = self._cand_slots(tr, k, -1)
                self.serving_pid[i] = (int(tr.top_idx[k, cands[0]])
                                       if cands else -1)
                s_pid = self.serving_pid[i]
            if not tr.is_live(s_pid, k):
                continue
            s_slot = self._slot_of(tr, k, s_pid)

            # --- 2. act ---------------------------------------------------
            a = int(actions[i])
            cands = self._cand_slots(tr, k, s_pid)
            if a == self.K:                                   # HOLD
                pass
            elif a < self.K or a == self.K + 1:
                if a < self.K:
                    tgt_slot = cands[a] if a < len(cands) else -1
                else:                                          # longest remaining
                    tgt_slot = (max(cands, key=lambda c: tr.top_rem[k, c])
                                if cands else -1)
                if tgt_slot < 0:
                    rew[i] += self.rw.terminal_penalty        # invalid action
                    self.n_handover[i] += 1
                    self.n_fail[i] += 1
                elif self.pending[i] is None:
                    # The transfer starts NOW (at the decision instant) and
                    # takes dt_exec to complete. It is NOT deferred to the last
                    # moment before the boundary: deferring would measure the
                    # target's survival rather than the trigger's timing, and
                    # the paper's criterion is stated in terms of t_init.
                    # Delta_t_ho enters only as GBPT's trigger lead time, which
                    # the dt_dec = 100 ms decision grid already subsumes
                    # (dt_ho = 26.7 ms < dt_dec), so it is not simulated again.
                    exec_dt = float(self.rng.lognormal(
                        np.log(max(self.dt_exec, 1e-6)), self.dt_exec_sigma))
                    self.pending[i] = (t, s_pid, int(tr.top_idx[k, tgt_slot]),
                                       t + exec_dt)
                    rew[i] += (self.rw.alpha_stability
                               * self.rw.lam_stability * -1.0)

            # --- 3. keep-alive quality reward ----------------------------
            s_elev = tr.elev_of(s_pid, k)
            snr = float(snr_from_elev_db(np.array([s_elev]), self.cfg)[0])
            rew[i] += self.rw.alpha_quality * np.tanh(snr / self.cfg.snr_target_db)

        self.k += 1
        done = (self.k - self.start_k) >= self.ep_len or \
               self.k >= self.n_step - 1
        obs = (self._obs(self.k) if not done
               else np.zeros((self.M, self.s_dim), dtype=np.float64))
        self.ep_reward += rew
        info = {"success_rate": self.success_rate(),
                "handovers": int(self.n_handover.sum()),
                "lost": int(self.n_lost.sum())}
        return obs, rew, done, info

    # ----------------------------------------------------------- reporting
    def success_rate(self):
        tot = int(self.n_handover.sum())
        return (float(self.n_ok.sum()) / tot) if tot > 0 else float("nan")

    def per_terminal_success(self):
        out = []
        for i in range(self.M):
            out.append(self.n_ok[i] / self.n_handover[i]
                       if self.n_handover[i] > 0 else float("nan"))
        return np.array(out, dtype=float)
