# -*- coding: utf-8 -*-
"""Trained-DQN target selection for the GAP-MOH ablation arm.

WHY THIS FILE EXISTS. `policies.select_target`'s `"policy"` branch documents
itself as a [DEVIATION] stand-in:

    "[DEVIATION] The real MADRL-Std target comes from a policy network. In this
     scripted diagnostic it is stood in by the same rule as longest_remaining
     ... Stage 1 replaces it with the trained net."

This is that replacement. It adapts a `gapmoh.rl.nets.DQN` -- trained in
`HandoverEnv` -- to the candidate-array calling convention of
`sim.simulate_terminal`, so a LEARNED policy can be evaluated through the SAME
path that produces every other row of the comparison. Without this the DRL
refinement layer would be measured by a different simulator from the rows it is
compared against, and the ablation would compare two worlds rather than two
layers.

THE OBSERVATION IS REBUILT HERE, NOT IMPORTED. `HandoverEnv._obs` builds its
observation from a `TerminalTrack`; `sim._emit` has a candidate array instead.
The two must agree field for field or the net is fed a vector it was never
trained on -- a wrong answer with no error. `_obs` below therefore mirrors
`HandoverEnv._obs` in feature order, normalisation constants and ordering
convention, and the first call checks the assembled width against the net's
input dimension so a layout drift fails loudly instead of quietly.

DOMAIN SHIFT, STATED RATHER THAN HIDDEN.

  [DEVIATION] The env decides WHEN and WHOM at every 100 ms step. Under
      `sim.simulate_terminal` the WHEN is fixed by the scheme's trigger (GBPT
      fires at t_exit - dt_ho, CN:140) and only the WHOM is open. The evaluated
      policy is therefore used as a TARGET SELECTOR. That is what the manuscript
      claims the refinement layer does (CN:31: "作为框架之上的精化层优化目标选择
      与负载均衡"), so the restriction is faithful -- but the net was trained on
      the joint (when, whom) problem, so the HOLD action (a == K) has no meaning
      at a trigger instant already past the point of no return. HOLD is mapped
      to the highest-elevation candidate and COUNTED (`n_hold`), so the fraction
      of HOLD actions actually taken is reported rather than silently absorbed.

  [DEVIATION] `HandoverEnv` has no `require_rising` filter; `sim._emit` applies
      one before the policy ever sees the candidates, so the evaluated candidate
      set is a subset of the trained one. The filter is applied to BOTH ablation
      arms, so the comparison stays apples-to-apples even though the training
      distribution shifted.

  [CHOICE]    The observation's SNR features are clear-sky
      (`snr_from_elev_db`), matching the env, even when the caller passes a
      rain-attenuated SNR. Feeding rain into the net would be a distribution it
      never saw.

No Table 7 number appears in this file.
"""

import json

import numpy as np

from ..channel import snr_from_elev_db

__all__ = ["PolicySelector", "save_checkpoint", "load_checkpoint"]

SERV_SNR_NORM_DB = 40.0
REM_CLIP_S = 600.0
N_HO_CLIP = 20.0
SPEED_NORM_KN = 40.0

# Must match `build_track`/`TerminalTrack.rate_of`: ch = max(1, round(4.0/dt)).
RATE_WINDOW_S = 4.0


class PolicySelector:
    """A trained DQN in `select_target`'s calling convention.

    Call it as::

        j = policy_fn(cand, ce, crem, e_srv, s_srv, ctx)

    where `cand` are PASS ids of the admissible candidates (serving already
    excluded, rising filter already applied), `ce`/`crem` their elevations
    (rad) and remaining visibility (s), `e_srv`/`s_srv` the serving link's
    elevation and SNR, and `ctx` the per-instant context built by `sim._emit`.
    Returns an index INTO `cand`, or None.
    """

    def __init__(self, net, cfg, K=None):
        self.net = net
        self.cfg = cfg
        self.K = int(cfg.n_candidates if K is None else K)
        self.n_calls = 0
        self.n_hold = 0            # action K: no meaning at a fixed trigger
        self.n_out_of_range = 0    # action pointed at an absent candidate
        self._checked = False

    # ------------------------------------------------------------------ obs
    def _obs(self, cand, ce, crem, ce_rate, s_elev, s_rate, s_rem, ctx):
        """Mirror of `HandoverEnv._obs`, feature for feature.

        Layout (6K+13): 5*K candidate features (elevation, elevation rate,
        remaining/600, clear-sky SNR/40, load), 5 serving features, 2 serving
        scalars, 5 terminal scalars, K+1 validity mask. Candidates are placed in
        DESCENDING ELEVATION order, because that is how `HandoverEnv._cand_slots`
        hands them over -- the action index means "the a-th highest-elevation
        candidate", not "the a-th candidate as listed".
        """
        K = self.K
        order = np.argsort(ce)[::-1][:K]

        feats = np.zeros(5 * K)
        for j, ix in enumerate(order):
            e = float(ce[ix])
            feats[5 * j:5 * j + 5] = [
                e,
                float(ce_rate[ix]),
                float(np.clip(crem[ix], 0.0, REM_CLIP_S)) / REM_CLIP_S,
                float(snr_from_elev_db(np.array([e]), self.cfg)[0])
                / SERV_SNR_NORM_DB,
                0.0,
            ]
        # Same slot order as the env: elevation, elevation rate, remaining/600,
        # clear-sky SNR/40, load. The SNR is recomputed clear-sky from the
        # elevation rather than taken from the caller, which may have applied
        # rain attenuation the net was never trained on.
        serving = np.array([
            float(s_elev), float(s_rate),
            float(np.clip(s_rem, 0.0, REM_CLIP_S)) / REM_CLIP_S,
            float(snr_from_elev_db(np.array([s_elev]), self.cfg)[0])
            / SERV_SNR_NORM_DB,
            0.0,
        ])
        misc = np.array([float(np.clip(s_rem, 0.0, REM_CLIP_S)) / REM_CLIP_S,
                         float(s_elev)])
        term = np.array([float(ctx["lat_rad"]), float(ctx["lon_rad"]),
                         float(ctx["speed_kn"]) / SPEED_NORM_KN,
                         float(ctx["t"]) / float(ctx["horizon_s"]),
                         min(1.0, float(ctx["n_ho"]) / N_HO_CLIP)])
        mask = np.zeros(K + 1)
        mask[:order.size] = 1.0
        # `TerminalTrack.is_live`: above the mask AND not yet past t_exit. The
        # remaining-time term is what carries the second half of that here,
        # since the selector has no direct handle on the serving pass table.
        mask[K] = 1.0 if (s_elev > np.deg2rad(self.cfg.min_elev_deg)
                          and s_rem > 0.0) else 0.0

        obs = np.concatenate([feats, serving, misc, term, mask])
        if not self._checked:
            want = int(self.net.q.dims[0])
            if obs.size != want:
                raise ValueError(
                    f"observation layout drift: built {obs.size} features, "
                    f"network wants {want}. `PolicySelector._obs` no longer "
                    f"matches `HandoverEnv._obs`; fix one to match the other "
                    f"before trusting any number this policy produces.")
            self._checked = True
        return obs, order

    # ----------------------------------------------------------------- call
    def __call__(self, cand, ce, crem, e_srv, s_srv, ctx):
        self.n_calls += 1
        cand = np.asarray(cand)
        if cand.size == 0:
            return None
        ce = np.asarray(ce, dtype=float)
        crem = np.asarray(crem, dtype=float)

        elev_of = ctx["elev_of_pass"]
        t = float(ctx["t"])
        ce_rate = np.array([
            (float(elev_of(int(p), t + RATE_WINDOW_S)) - float(ce[i]))
            / RATE_WINDOW_S for i, p in enumerate(cand)])
        s_rate = ((float(elev_of(int(ctx["serving_pass"]), t + RATE_WINDOW_S))
                   - float(e_srv)) / RATE_WINDOW_S)

        obs, order = self._obs(cand, ce, crem, ce_rate, float(e_srv), s_rate,
                               float(ctx["serving_remaining_s"]), ctx)
        a = int(np.argmax(self.net.q.forward(obs[None, :])[0]))

        if a < self.K:
            if a >= order.size:
                # the net asked for a candidate that does not exist at this
                # instant; the mask said so, but a policy can still pick it.
                self.n_out_of_range += 1
                return int(order[0])
            return int(order[a])
        if a == self.K:
            # HOLD is undefined at a trigger instant: the trigger has already
            # fired, so there is nothing left to wait for. Fall back to the
            # highest-elevation candidate and record the event.
            self.n_hold += 1
            return int(order[0])
        # a == K + 1: longest remaining, over the same top-K the env used.
        return int(order[int(np.argmax(crem[order]))])

    def stats(self):
        return {"n_calls": self.n_calls, "n_hold": self.n_hold,
                "n_out_of_range": self.n_out_of_range,
                "hold_fraction": (self.n_hold / self.n_calls
                                  if self.n_calls else float("nan"))}


def save_checkpoint(path, net, meta=None):
    """Persist a trained DQN as a `.npz` of bare arrays (no pickle)."""
    arrays = {"dims": np.asarray(net.q.dims, dtype=np.int64)}
    for i, (W, b, g, be) in enumerate(zip(net.q.Ws, net.q.bs,
                                          net.q.gammas, net.q.betas)):
        arrays[f"W{i}"] = W
        arrays[f"b{i}"] = b
        if g is not None:
            arrays[f"g{i}"] = g
            arrays[f"be{i}"] = be
    arrays["meta"] = np.asarray(json.dumps(meta or {}, ensure_ascii=False))
    np.savez(path, **arrays)
    return path


def load_checkpoint(path, cfg, K=None, rng=None):
    """Rebuild a `PolicySelector` from `save_checkpoint`. Returns (sel, meta)."""
    from .nets import DQN

    z = np.load(path, allow_pickle=False)
    dims = [int(x) for x in z["dims"]]
    rng = rng if rng is not None else np.random.default_rng(0)
    net = DQN(dims[0], dims[-1], hidden=tuple(dims[1:-1]), rng=rng)
    for i in range(len(dims) - 1):
        net.q.Ws[i] = np.asarray(z[f"W{i}"], dtype=np.float64)
        net.q.bs[i] = np.asarray(z[f"b{i}"], dtype=np.float64)
        if f"g{i}" in z:
            net.q.gammas[i] = np.asarray(z[f"g{i}"], dtype=np.float64)
            net.q.betas[i] = np.asarray(z[f"be{i}"], dtype=np.float64)
    net.qt.copy_from(net.q)
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    return PolicySelector(net, cfg, K=K), meta
